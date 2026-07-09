"""MedSegDecoder: SAM3 medical-image segmentation as a bio_qwen3vl decoder.

Conceptual mapping
------------------
Same role as RNALMDecoder, but heavyweight:

    LLM hidden states ──► (extract user-turn span) ──► proj ──► text_embeds
                                                                    │
                                                                    ▼
                          SAM3.forward(pixel_values=..., text_embeds=...) ──► (cls, box, mask)

Key design point — the text_embeds fed to SAM3 are NOT pure text condition.
They are the **figure-text joint hidden states** of the user's last turn:
the span includes
    <|vision_start|><|image_pad|>×N<|vision_end|> + the textual task query.
After ~30+ self-attention layers in Qwen3-VL, image_pad-position hidden
states already encode "task-aware visual semantics" and text-position hidden
states encode "image-grounded language". SAM3 then runs cross-attention from
its OWN multi-scale Hiera features to this joint representation — late
fusion across two visual pathways.

This is enforced by the ``user_text_mask`` tensor produced by the collator.
The mask is derived from Qwen3-VL's native image token IDs, so med_seg keeps
the Qwen tokenizer/model embedding table unchanged.
"""

from typing import Any, Dict, List, Optional, Tuple

import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_PROFILE_ENABLED = os.environ.get("BIO_MED_SEG_PROFILE", "0").lower() in {"1", "true", "yes", "on"}
_PROFILE_EVERY = int(os.environ.get("BIO_MED_SEG_PROFILE_EVERY", "10"))


class _StageTimer:
    """Lightweight CUDA-aware stage timer.

    On first call after a step starts, primes a fresh tracker. ``mark(name)``
    records elapsed wall time since the previous mark using torch.cuda.Event
    (sync-on-stop, no global synchronize). At step end, ``flush()`` prints a
    single ranked breakdown line like:

        [med_seg-prof step=120] qwen=3142.1 slice=0.4 proj=0.2 sam3=841.7 ...

    Disabled unless BIO_MED_SEG_PROFILE=1.
    """
    def __init__(self):
        self.step = 0
        self.events: List[Tuple[str, "torch.cuda.Event"]] = []
        self.cpu_marks: List[Tuple[str, float]] = []

    def reset(self):
        self.events = []
        self.cpu_marks = []

    def mark(self, name: str):
        if not _PROFILE_ENABLED:
            return
        if torch.cuda.is_available():
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.events.append((name, ev))
        else:
            self.cpu_marks.append((name, time.perf_counter()))

    def flush(self, prefix: str = ""):
        if not _PROFILE_ENABLED:
            return
        self.step += 1
        if self.step % _PROFILE_EVERY != 0:
            self.reset()
            return
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            self.reset()
            return
        parts = []
        if torch.cuda.is_available() and self.events:
            torch.cuda.synchronize()
            for i in range(1, len(self.events)):
                name = self.events[i][0]
                ms = self.events[i - 1][1].elapsed_time(self.events[i][1])
                parts.append(f"{name}={ms:.1f}")
        elif self.cpu_marks:
            for i in range(1, len(self.cpu_marks)):
                name = self.cpu_marks[i][0]
                ms = (self.cpu_marks[i][1] - self.cpu_marks[i - 1][1]) * 1000
                parts.append(f"{name}={ms:.1f}")
        if parts:
            # Use print + flush to bypass any logger filtering and ensure
            # the line lands in the redirected stdout/log file even when
            # transformers/HF Trainer mute the logging hierarchy.
            print(
                f"[med_seg-prof {prefix + ' ' if prefix else ''}step={self.step}] "
                + " ".join(parts),
                flush=True,
            )
        self.reset()


_PROF = _StageTimer()

from qwenvl.registry.base import BaseDecoder
from qwenvl.registry.registry import ComponentRegistry

logger = logging.getLogger(__name__)


@ComponentRegistry.register_decoder("med_seg_sam3_decoder")
class MedSegDecoder(BaseDecoder):
    """Wraps a SAM3 model as a bio_qwen3vl decoder.

    The decoder takes Qwen3-VL hidden states + extras (a SECOND pixel_values
    tensor sized for SAM3, prompt boxes, user-turn mask, GT targets) and
    returns the SAM3 ModelOutput. ``compute_loss`` runs Hungarian matching
    + multi-task loss (cls focal + L1 + GIoU + mask focal/dice).

    Marker: ``REQUIRES_EXTRAS = True`` — the ModalityRouter uses this to
    skip the decoder on ranks that have no med-seg samples (SAM3 cannot
    produce a meaningful zero-cost dummy forward — its image branch needs
    real pixels).
    """

    REQUIRES_EXTRAS = True

    def __init__(
        self,
        llm_hidden_size: int,
        config: Optional[Any] = None,
        sam3_model: Optional[nn.Module] = None,
        **_,
    ):
        super().__init__()
        self.config = config
        self.llm_hidden_size = int(llm_hidden_size)
        self.sam3_text_dim = int(getattr(config, "sam3_text_dim", 256))
        # Projection bridge Qwen → SAM3.
        # The default (``proj_mlp=True``) is a 2-layer MLP with LayerNorm
        # and GELU. The previous single Linear (4096 → 256) compressed
        # the Qwen joint hidden state by 16× without any non-linearity,
        # bottlenecking fine-grained semantic conditioning for L3
        # tasks (PanNuke / amos22 / kits23 / MSD).
        proj_mlp = bool(getattr(config, "proj_mlp", True))
        proj_hidden_mult = int(getattr(config, "proj_hidden_mult", 2))
        if proj_mlp:
            hidden = self.sam3_text_dim * max(1, proj_hidden_mult)
            self.proj = nn.Sequential(
                nn.LayerNorm(self.llm_hidden_size),
                nn.Linear(self.llm_hidden_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.sam3_text_dim),
            )
        else:
            self.proj = nn.Linear(self.llm_hidden_size, self.sam3_text_dim, bias=False)
        # SAM3 is loaded by the trainer (huge weights, may need from_pretrained
        # outside __init__) and attached here. ``set_sam3`` is provided so
        # the same MedSegDecoder skeleton can be created early and SAM3
        # injected after the rest of the model is built.
        self.sam3 = sam3_model

        self.loss_w = {
            "cls": float(getattr(config, "loss_w_cls", 1.0)),
            "bbox_l1": float(getattr(config, "loss_w_bbox_l1", 5.0)),
            "bbox_giou": float(getattr(config, "loss_w_bbox_giou", 2.0)),
            "mask_focal": float(getattr(config, "loss_w_mask_focal", 2.0)),
            "mask_dice": float(getattr(config, "loss_w_mask_dice", 2.0)),
            "mask_dice_high": float(getattr(config, "loss_w_mask_dice_high", 0.0)),
            "mask_semantic": float(getattr(config, "loss_w_mask_semantic", 0.0)),
            "meta_ce": float(getattr(config, "loss_w_meta_ce", 0.0)),
        }
        self.meta_num_classes = int(getattr(config, "meta_num_classes", 15))
        self.meta_classifier = nn.Sequential(
            nn.LayerNorm(self.sam3_text_dim),
            nn.Linear(self.sam3_text_dim, self.meta_num_classes),
        )
        if self.loss_w["meta_ce"] <= 0.0:
            for p in self.meta_classifier.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # SAM3 attachment
    # ------------------------------------------------------------------
    def set_sam3(self, sam3_model: nn.Module):
        """Inject the loaded Sam3Model after construction."""
        self.sam3 = sam3_model

    @property
    def output_size(self) -> int:
        # Single-class binary detection (matches sam_uni reference).
        return 1

    # ------------------------------------------------------------------
    # Hidden-state span extraction (figure-text joint region)
    # ------------------------------------------------------------------
    def _slice_user_turn_hidden(
        self,
        hidden_states: Tensor,        # [B, L, D_llm]
        user_text_mask: Tensor,        # [B, L] bool
    ) -> Tuple[Tensor, Tensor]:
        """Pull out the user-turn span (image_pad block + text query) per sample,
        right-pad to a common length, and return (hid_pad, attn_pad).

        The mask covers the entire span — image_pad positions included — so
        SAM3 receives a figure-text joint representation, not just text.
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device
        embeds_list: List[Tensor] = []
        max_t = 1
        for i in range(B):
            hi = hidden_states[i][user_text_mask[i].bool()]
            if hi.numel() == 0:
                # Fallback: keep the last token to avoid empty input
                hi = hidden_states[i][-1:]
            embeds_list.append(hi)
            max_t = max(max_t, hi.shape[0])

        hid_pad = torch.zeros((B, max_t, D), dtype=hidden_states.dtype, device=device)
        attn_pad = torch.zeros((B, max_t), dtype=torch.long, device=device)
        for i in range(B):
            ti = embeds_list[i].shape[0]
            hid_pad[i, :ti] = embeds_list[i]
            attn_pad[i, :ti] = 1
        return hid_pad, attn_pad

    # ------------------------------------------------------------------
    # decode → compute_loss
    # ------------------------------------------------------------------
    def decode(
        self,
        hidden_states: Tensor,
        pixel_values_sam3: Optional[Tensor] = None,
        input_boxes: Optional[Tensor] = None,
        input_boxes_labels: Optional[Tensor] = None,
        user_text_mask: Optional[Tensor] = None,
        targets: Optional[Dict] = None,
        **extras,
    ):
        """Run SAM3 with Qwen-derived figure-text joint hidden as text_embeds.

        Returns the raw Sam3 ModelOutput (with ``.pred_logits``,
        ``.pred_boxes``, ``.pred_masks``).
        """
        if self.sam3 is None:
            raise RuntimeError(
                "MedSegDecoder.sam3 is not set. Call decoder.set_sam3(...) "
                "after loading the Sam3 model."
            )
        if user_text_mask is None:
            raise ValueError(
                "MedSegDecoder.decode requires `user_text_mask` (bool [B, L]) "
                "covering the user-turn figure-text span."
            )
        if pixel_values_sam3 is None:
            raise ValueError(
                "MedSegDecoder.decode requires `pixel_values_sam3` (SAM3 image input)."
            )

        _PROF.mark("decode_in")

        # Ensure mask aligns with hidden_states length (LLM may have truncated)
        L = hidden_states.shape[1]
        if user_text_mask.shape[1] != L:
            if user_text_mask.shape[1] > L:
                user_text_mask = user_text_mask[:, :L]
            else:
                pad = L - user_text_mask.shape[1]
                user_text_mask = F.pad(user_text_mask, (0, pad), value=False)

        hid_pad, attn_pad = self._slice_user_turn_hidden(hidden_states, user_text_mask)
        _PROF.mark("slice")

        # Project Qwen hidden → SAM3 text dim. ``self.proj`` is either a
        # single Linear (legacy) or an MLP — pull dtype from the first
        # parameter rather than ``.weight`` so both shapes work.
        proj_dtype = next(self.proj.parameters()).dtype
        sam3_text_embeds = self.proj(hid_pad.to(proj_dtype))
        meta_logits = None
        if self.loss_w["meta_ce"] > 0.0:
            attn_f = attn_pad.to(device=sam3_text_embeds.device, dtype=sam3_text_embeds.dtype)
            pooled_text = (
                sam3_text_embeds * attn_f.unsqueeze(-1)
            ).sum(dim=1) / attn_f.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            meta_dtype = next(self.meta_classifier.parameters()).dtype
            meta_logits = self.meta_classifier(pooled_text.to(meta_dtype))
        # Stash for the NaN probe in compute_loss (helps locate which
        # stage first goes bad: proj output vs SAM3 internal forward).
        self._last_text_embeds = sam3_text_embeds.detach()
        # Probe Qwen-side hidden too — if the joint hidden span is bad,
        # the proj output will be bad regardless of proj weights.
        self._last_qwen_hidden_slice = hid_pad.detach()
        _PROF.mark("proj")

        sam3_kwargs: Dict[str, Any] = {
            "pixel_values": pixel_values_sam3,
            "input_ids": None,
            "attention_mask": attn_pad,
            "text_embeds": sam3_text_embeds,
            "return_dict": True,
        }
        if input_boxes is not None and input_boxes.numel() > 0:
            sam3_kwargs["input_boxes"] = input_boxes
        if input_boxes_labels is not None and input_boxes_labels.numel() > 0:
            sam3_kwargs["input_boxes_labels"] = input_boxes_labels

        out = self.sam3(**sam3_kwargs)
        if meta_logits is not None:
            try:
                setattr(out, "meta_logits", meta_logits)
            except Exception:
                pass
        _PROF.mark("sam3_fwd")
        return out
