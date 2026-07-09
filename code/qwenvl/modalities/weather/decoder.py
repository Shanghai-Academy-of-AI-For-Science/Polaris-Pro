"""Weather decoder: Polaris meteo head + regression-style loss.

Adapts ``PolarisMeteoHead`` to the bio_qwen3vl ``ModalityRouter``.  The
decoder is invoked by ``compute_decoder_losses`` via the custom
``compute_loss_from_hidden`` hook (see ``qwenvl/registry/modality_router.py``):
that path bypasses the standard CE-on-decode-logits flow used by RNA /
protein / mol since meteo prediction is a regression problem on a 5D
tensor field.

Inputs needed at loss time live in two places:

* ``hidden_states`` (LLM output) — selects the meteo-token positions.
* ``self.encoder._step_cache`` — set by ``WeatherEncoder.forward`` and
  contains ``condition_embed`` / ``patch_embed_pre_swin`` /
  ``meteo_values`` / ``targets`` / ``lead_hours``.

The encoder reference is wired up in ``register_modality`` so the decoder
does not have to be passed it through every forward.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .internal.polaris_attention import precompute_freqs_cis
from .internal.polaris_layers import AdaLN, DoubleDeconvHead, remove_small_scales
from .internal.polaris_swin import SwinBlock
from .internal.helpers import round_to_multiple

logger = logging.getLogger(__name__)


class WeatherDecoder(nn.Module):
    """Polaris meteo head + regression loss path.

    The original Polaris ``PolarisMeteoHead.forward`` is wrapped in
    ``compute_loss_from_hidden`` so it can pull the matching encoder cache
    and produce a scalar loss directly from LLM hidden states.
    """

    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        # Hold the encoder reference in a 1-element list so PyTorch's
        # automatic Module-attribute registration does NOT pull the
        # encoder's parameters in as decoder children (which would
        # duplicate them under both encoders.weather.* and decoders.weather.*
        # in the state_dict).
        self._encoder_ref = [encoder]

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.qwenvl_dim = config.qwenvl_dim
        self.patch_size = config.patch_size
        self.upper_chans = config.upper_chans
        self.lower_chans = config.lower_chans
        self.decoder_depth = config.decoder_depth

        if isinstance(config.image_size, int):
            in_h = in_w = config.image_size
        else:
            in_h, in_w = config.image_size
        if self.patch_size == 1:
            swin_H = in_h // 2 * 2
            swin_W = in_w
        else:
            swin_H = in_h // self.patch_size
            swin_W = in_w // self.patch_size
        self._swin_HW = (swin_H, swin_W)

        max_seq_len = swin_H * swin_W
        freqs_cos, freqs_sin = precompute_freqs_cis(self.hidden_size // self.num_heads, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self._rope_buffers_checked = False

        # Cross-modal back projector: LLM hidden → swin hidden.
        # Trained at ``mm_projector_lr`` (matched on substring "mlp_qwen2swin").
        self.mlp_qwen2swin = nn.Sequential(
            nn.Linear(self.qwenvl_dim, self.qwenvl_dim),
            nn.GELU(),
            nn.Linear(self.qwenvl_dim, self.hidden_size),
        )

        if config.embed_mode == "add":
            self.embed_dim = config.hidden_size
        elif config.embed_mode == "cat":
            self.embed_dim = round_to_multiple(config.hidden_size // len(config.embed_types)) * len(config.embed_types)
        else:
            raise ValueError(f"Invalid embed_mode: {config.embed_mode}")

        self.decoder_layers = nn.ModuleList()
        for i in range(self.decoder_depth):
            blk = SwinBlock(
                dim=self.hidden_size, embed_dim=self.embed_dim, num_heads=self.num_heads,
                input_size=(swin_H, swin_W), window_size=config.window_size,
                shift_size=0 if i % 2 == 0 else config.window_size // 2,
                mlp_ratio=config.mlp_ratio, attn_type=config.attn_type,
                mask_type=config.mask_type, norm_type=config.norm_type,
                ffn_type=config.ffn_type, n_kv_heads=config.n_kv_heads,
                attn_implementation="eager",
            )
            self.decoder_layers.append(blk)

        self.norm_layer = AdaLN(self.hidden_size, embed_dim=self.embed_dim)
        self.pred_layer = DoubleDeconvHead(
            in_chans=self.hidden_size, upper_chans=self.upper_chans,
            lower_chans=self.lower_chans, patch_size=self.patch_size,
        )

        self.gradient_checkpointing = False

    # Provide the same context-injection API the encoder has, so the
    # training entry can call it on the decoder too without checking
    # which side actually owns the buffers.
    def inject_meteorological_context(self, *args, **kwargs):
        # The encoder is the canonical owner of mean/std/nanmask/...
        # We ignore here to avoid duplicate state.
        return

    def _ensure_finite_rope_buffers(self) -> None:
        """Rebuild the RoPE tables if the non-persistent buffers came up broken.

        ``freqs_cos`` / ``freqs_sin`` are ``persistent=False`` (not in the
        checkpoint) and are meant to be filled at ``__init__``. Under a
        meta-device ``from_pretrained`` load they can be ``to_empty``'d to
        uninitialised memory. A valid table has ``cos[0]==1`` and ``sin[0]==0``
        for every column — checking row 0 catches garbage that happens to be
        finite (large floats) which an isfinite-only check would miss.
        """
        if self._rope_buffers_checked:
            return
        with torch.no_grad():
            row0 = self.freqs_cos[0].detach().to(torch.float32)
            row0_sin = self.freqs_sin[0].detach().to(torch.float32)
        needs_recompute = (
            not torch.isfinite(self.freqs_cos).all()
            or not torch.isfinite(self.freqs_sin).all()
            or not torch.allclose(row0, torch.ones_like(row0), atol=1e-5)
            or not torch.allclose(row0_sin, torch.zeros_like(row0_sin), atol=1e-5)
        )
        if needs_recompute:
            logger.warning(
                "[weather] broken decoder RoPE buffers detected "
                "(nan/inf/zeros); recomputing freqs_cos/freqs_sin"
            )
            max_seq_len = int(self._swin_HW[0] * self._swin_HW[1])
            freqs_cos, freqs_sin = precompute_freqs_cis(
                self.hidden_size // self.num_heads, max_seq_len,
            )
            with torch.no_grad():
                self.freqs_cos.copy_(freqs_cos.to(self.freqs_cos.device, self.freqs_cos.dtype))
                self.freqs_sin.copy_(freqs_sin.to(self.freqs_sin.device, self.freqs_sin.dtype))
        self._rope_buffers_checked = True

    # ------------------------------------------------------------------
    # Loss computation (the only ModalityRouter-callable entry point)
    # ------------------------------------------------------------------

    def predict_from_hidden(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Decode LLM hidden states into a meteo prediction tensor.

        Pure inference: no loss, no NaN-guard side effects.  Reusable by
        the rollout engine, which needs to call this multiple times within
        one optimizer step (warm-up steps + selected step).

        Returns ``meteo_output`` of shape ``[B, C, H, W]`` (the ``pred_layer``
        currently emits a single-frame prediction; rollout stitches frames
        externally), or ``None`` if no weather pad tokens are present.
        """
        encoder = self._encoder_ref[0]
        cache = getattr(encoder, "_step_cache", None) or {}
        if not cache:
            return None

        self._ensure_finite_rope_buffers()

        condition_embed = cache["condition_embed"]
        patch_embed_post_swin = cache["patch_embed_post_swin"]
        meteo_values = cache["meteo_values"]
        lead_hours = cache["lead_hours"]
        input_size = cache["input_size"]
        T_in = cache["T_in"]

        input_ids = kwargs.get("input_ids")
        weather_pad_id = self._weather_pad_id_from_config(kwargs)
        if input_ids is None or weather_pad_id is None:
            raise RuntimeError(
                "WeatherDecoder.predict_from_hidden: input_ids / weather pad id missing."
            )

        meteo_mask = (input_ids == weather_pad_id)
        B, _, D = hidden_states.shape
        L_per_sample = int(meteo_mask.sum(dim=1)[0].item())
        if L_per_sample == 0:
            return None
        meteo_states = hidden_states[meteo_mask].view(B, L_per_sample, D)

        h = self.mlp_qwen2swin(meteo_states)
        if patch_embed_post_swin is not None:
            h = h + patch_embed_post_swin

        # Skip gradient checkpointing under torch.no_grad() (e.g. rollout
        # warm-up) — checkpoint() warns and offers no benefit when the
        # outputs don't require grad.
        use_gc = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for blk in self.decoder_layers:
            if use_gc:
                h = checkpoint(
                    blk, h, condition_embed, self.freqs_cos, self.freqs_sin,
                    use_reentrant=False,
                )
            else:
                h = blk(h, condition_embed, self.freqs_cos, self.freqs_sin)

        h = self.norm_layer(h, condition_embed)
        h = rearrange(h, "n (h w) c -> n c h w", h=input_size[0] // self.patch_size // 2 * 2)
        meteo_output = self.pred_layer(
            h,
            residual=meteo_values[:, -1],
            input_size=input_size,
            lead_hour=condition_embed.new_tensor([0.0]) if lead_hours is None else lead_hours,
            target_frames=T_in - 1,
        )
        return meteo_output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weather_pad_id_from_config(kwargs) -> Optional[int]:
        """Extract ``<|weather_pad|>`` id.

        We rely on the model upstream to pass it under the key
        ``__weather_pad_id__`` (set by the model wrapper before calling
        ``compute_decoder_losses``).  Falls back to None when missing.
        """
        return kwargs.get("__weather_pad_id__")
