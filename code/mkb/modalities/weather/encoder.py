"""Weather encoder: Polaris Swin-ViT + meteo merger (projector to LLM hidden).

Contract:

* ``forward(input_ids, attention_mask, **extra_kwargs) -> (latent, mask)``.
  ``input_ids`` / ``attention_mask`` are placeholders sized to the number of
  meteo tokens; the real meteorological tensors arrive through ``extra_kwargs``
  (``meteo_values``, ``times``, ``lead_hours``, …) under the ``weather_`` prefix.

* The encoder pre-computes ``condition_embed`` and the patch embed and stores
  them in ``self._step_cache`` for the matching ``WeatherDecoder`` to reuse.

* Mean / std / nanmask buffers (from ``inject_meteorological_context``) are
  stored on the encoder so ``unnormalize`` / lat-weighting / channel masking
  run without any external wrapper.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .internal.polaris_attention import precompute_freqs_cis
from .internal.polaris_layers import (
    AdaLN,
    CubeEmbedConv,
    LayerNorm,
    PatchEmbedConv,
    TimestepEmbed,
)
from .internal.polaris_swin import SwinBlock
from .internal.helpers import round_to_multiple

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as _MergerNorm
except Exception:  # pragma: no cover
    _MergerNorm = nn.LayerNorm

logger = logging.getLogger(__name__)


class PolarisMeteoPatchMerger(nn.Module):
    """Cross-modal projector ('merger'): Swin hidden → LLM hidden.

    Same arithmetic as the original Polaris ``PolarisMeteoPatchMerger``;
    serves as the modality projector for the ``ModalityRouter`` (an
    ``IdentityProjector`` is registered separately so the router can scatter
    its output directly).
    """

    def __init__(self, dim: int, context_dim: int) -> None:
        super().__init__()
        self.hidden_size = context_dim
        self.ln_q = _MergerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        x = self.mlp(x)
        x = x.view(bsz, seqlen, -1)
        return x


class WeatherEncoder(nn.Module):
    """Polaris-style Swin encoder + meteo merger.

    The forward signature is dictated by ``ModalityRouter.encode_and_project``:
    it takes ``input_ids`` / ``attention_mask`` (here used only to derive a
    batch size; their real meaning is conveyed by the placeholder pad
    tokens scattered into the LLM input embeddings) and a number of
    keyword-only extras forwarded by the collator under the ``weather_``
    prefix (already stripped by the router).
    """

    # Exposed so the router can route per-modality state_dict keys cleanly.
    is_image_like = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.encoder_depth = config.encoder_depth
        self.num_heads = config.num_heads
        self.const_chans = config.const_chans

        self.patch_embed = CubeEmbedConv(
            in_chans=config.in_chans,
            out_chans=self.hidden_size,
            in_frames=config.in_frames,
            norm_func=None,
            flatten=True,
            patch_size=self.patch_size,
        )

        if self.const_chans > 0:
            self.const_embed = PatchEmbedConv(
                self.const_chans, self.hidden_size, config.patch_size,
                norm_func=LayerNorm, flatten=True,
            )

        self.embed_mode = config.embed_mode
        self.embed_types = list(config.embed_types)
        self.embed_freq = config.embed_freq
        self._init_time_embeddings()

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

        self.encoder_layers = nn.ModuleList()
        for i in range(self.encoder_depth):
            blk = SwinBlock(
                dim=self.hidden_size, embed_dim=self.embed_dim, num_heads=self.num_heads,
                input_size=(swin_H, swin_W), window_size=self.window_size,
                shift_size=0 if i % 2 == 0 else self.window_size // 2,
                mlp_ratio=config.mlp_ratio, attn_type=config.attn_type,
                mask_type=config.mask_type, norm_type=config.norm_type,
                ffn_type=config.ffn_type, n_kv_heads=config.n_kv_heads,
                attn_implementation="eager",
            )
            self.encoder_layers.append(blk)

        # Cross-modal merger: Swin hidden → LLM hidden (qwenvl_dim).
        # `ModalityRouter` registers an IdentityProjector for `weather` since
        # the upscaling already happens here.
        self.meteo_merger = PolarisMeteoPatchMerger(
            dim=config.qwenvl_dim, context_dim=self.hidden_size,
        )

        max_seq_len = swin_H * swin_W
        freqs_cos, freqs_sin = precompute_freqs_cis(self.hidden_size // self.num_heads, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.gradient_checkpointing = False

        # ── Lazy data buffers (mean/std/nanmask/const/weight/...) ───
        # Populated by ``inject_meteorological_context``.  Held in fp32
        # CPU tensors and copied to device on demand to keep memory low.
        self._polaris_data_loaded = False
        self.channels: List[str] = []
        self.indices: Dict[str, List[int]] = {}
        self.coords: Dict[str, list] = {}
        self._fp32_master: Dict[str, torch.Tensor] = {}

        # Per-forward cache shared with the matching WeatherDecoder.
        # Cleared at the start of every encoder forward to avoid leaking
        # graph-disconnected tensors across steps.
        self._step_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # RoPE buffer self-heal
    # ------------------------------------------------------------------

    def _ensure_rope_buffers(self) -> None:
        """Rebuild the RoPE tables if the non-persistent buffers came up broken.

        ``freqs_cos`` / ``freqs_sin`` are ``persistent=False`` (not in the
        checkpoint) and are meant to be filled at ``__init__``. Under a
        meta-device ``from_pretrained`` load they can be ``to_empty``'d to
        uninitialised memory. A valid table has ``cos[0]==1`` and ``sin[0]==0``
        for every column — checking row 0 catches garbage that is finite (large
        floats) which an isfinite-only check would miss.
        """
        with torch.no_grad():
            row0 = self.freqs_cos[0].detach().to(torch.float32)
            row0_sin = self.freqs_sin[0].detach().to(torch.float32)
        needs_recompute = (
            not torch.isfinite(self.freqs_cos).all()
            or not torch.isfinite(self.freqs_sin).all()
            or not torch.allclose(row0, torch.ones_like(row0), atol=1e-5)
            or not torch.allclose(row0_sin, torch.zeros_like(row0_sin), atol=1e-5)
        )
        if not needs_recompute:
            return
        head_dim = self.hidden_size // self.num_heads
        max_seq_len = self.freqs_cos.shape[0]
        new_cos, new_sin = precompute_freqs_cis(head_dim, max_seq_len)
        with torch.no_grad():
            self.freqs_cos.copy_(new_cos.to(self.freqs_cos.device, self.freqs_cos.dtype))
            self.freqs_sin.copy_(new_sin.to(self.freqs_sin.device, self.freqs_sin.dtype))

    # ------------------------------------------------------------------
    # Time-conditioning
    # ------------------------------------------------------------------

    def _init_time_embeddings(self):
        if self.embed_mode == "add":
            self.embed_dim = self.hidden_size
        elif self.embed_mode == "cat":
            self.embed_dim = round_to_multiple(self.hidden_size // len(self.embed_types)) * len(self.embed_types)
        else:
            raise ValueError(f"Invalid embed_mode: {self.embed_mode}")

        # Optional dropout on time-embedding outputs (anti-overfit for
        # ``hour`` / ``doy``).  Read from config; default 0.0 keeps the
        # legacy zero-dropout behaviour identical to upstream Polaris.
        time_embed_dropout = float(getattr(self.config, "time_embed_dropout", 0.0))
        for k in self.embed_types:
            embed_layer = TimestepEmbed(
                self.embed_dim, frequency=self.embed_freq,
                is_periodic=(k in ["hour", "doy"]), sinusoidal=True,
                dropout=time_embed_dropout,
            )
            self.add_module(f"{k}_embed", embed_layer)

    def _forward_embedding(self, conds: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.embed_mode == "add":
            embed = 0
            for k in self.embed_types:
                embedding = getattr(self, f"{k}_embed")
                embed = embed + embedding(conds[k])
            return embed
        # cat
        embeds = [getattr(self, f"{k}_embed")(conds[k]) for k in self.embed_types]
        return torch.cat(embeds, dim=1)

    # ------------------------------------------------------------------
    # Meteo data buffer management
    # ------------------------------------------------------------------

    def inject_meteorological_context(self, channels, indices, coords, buffers):
        """Inject ERA5 mean/std/nanmask buffers onto this encoder.

        Stores fp32 tensors on CPU; ``_get_fp32_const`` copies to device on
        demand.  Mirrors the original Polaris API so external loaders
        (``polaris_data_utils.load_meteorological_buffers``) work unchanged.
        """
        self.channels = channels
        self.indices = indices
        self.coords = coords
        for k, v in buffers.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v).float()
            elif torch.is_tensor(v):
                t = v.detach().float()
            else:
                setattr(self, k, v)
                continue
            self._fp32_master[k] = t.cpu()
        self._polaris_data_loaded = True

    def _get_fp32_const(self, name: str, device: torch.device) -> torch.Tensor:
        if not self._polaris_data_loaded or name not in self._fp32_master:
            raise RuntimeError(
                f"WeatherEncoder data not loaded! Call inject_meteorological_context() "
                f"before using '{name}'."
            )
        return self._fp32_master[name].to(device=device, dtype=torch.float32, non_blocking=True)

    # ------------------------------------------------------------------
    # Input pre-processing (replicates Polaris reset_input / nan handling)
    # ------------------------------------------------------------------

    def _reset_output(self, x: torch.Tensor) -> torch.Tensor:
        if "nanmask" in self._fp32_master:
            nanmask = self._fp32_master["nanmask"].to(x.device)[: x.shape[-3]]
            x = x * (~nanmask)
        if "channel_mask" in self._fp32_master:
            cmask = self._fp32_master["channel_mask"].to(x.device)[: x.shape[-3]]
            x = x * cmask
        return x

    def _reset_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reset_output(x)
        accumid = self.indices.get("accumid", [])
        if len(accumid) > 0:
            if torch.is_grad_enabled():
                x = x.clone()
            x[:, :, accumid] = 0
        return x

    # ------------------------------------------------------------------
    # Conditioning factory (hour, doy, step, lead_hour, optional const)
    # ------------------------------------------------------------------

    def get_condition(
        self,
        t: int,
        times,
        lead_hour,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        times = pd.DatetimeIndex(times)
        B = len(times)

        if isinstance(lead_hour, torch.Tensor):
            lead_np = lead_hour.detach().to(dtype=torch.float32, device="cpu").numpy()
        else:
            lead_np = np.asarray(lead_hour, dtype=np.float32)

        if lead_np.ndim == 0:
            lead_np = np.full((B,), float(lead_np), dtype=np.float32)
        elif lead_np.size == 1 and B > 1:
            lead_np = np.full((B,), float(lead_np.reshape(-1)[0]), dtype=np.float32)
        elif lead_np.size != B:
            raise ValueError(f"lead_hour size mismatch: got {lead_np.size}, expected {B}")

        cur_times = times + pd.to_timedelta(lead_np * t, unit="h")
        used_times = cur_times

        hour_np = used_times.hour.values.astype(np.float32) / 24.0
        doy_np = (np.minimum(365, used_times.dayofyear.values).astype(np.float32) / 365.0)

        hour = torch.tensor(hour_np, dtype=dtype, device=device)
        doy = torch.tensor(doy_np, dtype=dtype, device=device)

        Tmax = getattr(self.config, "max_rollout_steps", 200)
        step_scalar = float(np.log1p(float(t)) / np.log1p(float(Tmax)))
        step = torch.full((B,), step_scalar, dtype=dtype, device=device)
        if self.training:
            step = step + torch.empty((B,), device=device, dtype=dtype).uniform_(-0.005, 0.005)

        lead_hour_t = torch.tensor(lead_np, dtype=dtype, device=device)

        conds = dict(hour=hour, doy=doy, step=step, lead_hour=lead_hour_t)
        if "const" in self._fp32_master:
            conds["const"] = self._get_fp32_const("const", device).type(dtype)
        return conds

    # ------------------------------------------------------------------
    # Encoder forward (ModalityRouter contract)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meteo_values: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        times: Optional[Any] = None,
        lead_hours: Optional[torch.Tensor] = None,
        polaris_task: Optional[Any] = None,
        channel_mask: Optional[torch.Tensor] = None,
        step_idx: int = 0,
        **_unused,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode meteorological inputs.

        Args:
            input_ids / attention_mask: dummy placeholders sized
                ``[B, swin_H * swin_W]``.  Required by the router signature
                but not used computationally.
            meteo_values: ``[B, T_in, C, H, W]`` float32 (smuggled as int32 by
                the collator and reinterpreted upstream) — already on device.
            times: pandas timestamps for the current batch (one per sample).
            lead_hours: ``[B]`` int / float — forecast lead hours.
            channel_mask: optional ``[B, C, 1, 1]`` channel mask.
            step_idx: int rollout step index (0 for single-step training).

        Returns:
            latent: ``[B, swin_H * swin_W, qwenvl_dim]`` after meteo_merger.
            latent_mask: None.
        """
        if meteo_values is None:
            raise ValueError("WeatherEncoder.forward requires `meteo_values` in extras.")

        # Rebuild RoPE freqs if the checkpoint load left them uninitialised
        # (non-persistent buffers under a meta-device load come up as garbage).
        self._ensure_rope_buffers()

        # NaN→zero, fp32 normalisation
        if meteo_values.dtype == torch.int32:
            meteo_values = meteo_values.contiguous().view(torch.float32)
        meteo_values = meteo_values.contiguous().to(torch.float32)
        nanmask_local = torch.isnan(meteo_values[-1, -1])
        if torch.any(nanmask_local):
            self._fp32_master["nanmask"] = nanmask_local.cpu()
        meteo_values = torch.nan_to_num(meteo_values)
        meteo_values = self._reset_input(meteo_values)
        if channel_mask is not None:
            self._fp32_master["channel_mask"] = channel_mask[0].cpu()

        device = meteo_values.device
        # Compute conditions & feed encoder. Use the encoder's compute dtype
        # (e.g. bf16) — meteo_values is fp32 here due to the upstream
        # NaN/normalisation cast, but the embedding MLPs run in bf16 and
        # would otherwise see a dtype mismatch in F.linear.
        conds = self.get_condition(step_idx, times, lead_hours, device=device, dtype=self._embed_dtype())

        meteo_compute = meteo_values.to(self._embed_dtype())
        meteo_feature = self.patch_embed(meteo_compute, conds["lead_hour"])
        patch_embed = meteo_feature

        if self.const_chans > 0 and "const" in conds:
            const = conds["const"].to(device=device, dtype=meteo_compute.dtype)
            meteo_feature = meteo_feature + self.const_embed(const)

        condition_embed = self._forward_embedding(conds)

        # Skip gradient checkpointing under torch.no_grad() — e.g. rollout
        # warm-up steps — otherwise PyTorch emits a per-block "None of the
        # inputs have requires_grad=True" warning and `checkpoint` adds
        # pointless bookkeeping for a path that won't backprop anyway.
        use_gc = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for blk in self.encoder_layers:
            if use_gc:
                meteo_feature = checkpoint(
                    blk, meteo_feature, condition_embed, self.freqs_cos, self.freqs_sin,
                    use_reentrant=False,
                )
            else:
                meteo_feature = blk(meteo_feature, condition_embed, self.freqs_cos, self.freqs_sin)

        # `meteo_feature` is the post-swin encoder output (Polaris's
        # `meteo_embeds`).  The matching head uses it as the skip connection
        # *after* mlp_qwen2swin, so cache before applying the merger.
        meteo_embeds_post_swin = meteo_feature

        merged = self.meteo_merger(meteo_feature)  # → [B, L, qwenvl_dim]

        # Cache for the decoder.  We deliberately keep references to all
        # the intermediate tensors needed by the head so the decoder can
        # reuse them without recomputing the encoder side.
        # ``patch_embed_post_swin`` matches the Polaris ``meteo_embeds``
        # variable used in ``polaris_head(meteo_feature_patch_embed=...)``
        # under the use_language=True flow — i.e. the encoder swin output.
        self._step_cache = {
            "condition_embed": condition_embed,
            "patch_embed_post_swin": meteo_embeds_post_swin,
            "meteo_values": meteo_values,
            "targets": targets,
            "times": times,
            "lead_hours": lead_hours,
            "input_size": meteo_values.shape[-2:],
            "T_in": meteo_values.shape[1],
        }

        # Mask is None — every meteo token is valid; the router will infer
        # the count from the pad-token mask.
        return merged, None

    def _embed_dtype(self) -> torch.dtype:
        # Canonical encoder compute dtype (handles bf16 / fp32 uniformly).
        for p in self.parameters():
            return p.dtype
        return torch.float32

    @torch.no_grad()
    def unnormalize(self, x: torch.Tensor, fill_type: str = "zero") -> torch.Tensor:
        """Undo channel-wise mean/std normalisation. fp32 throughout."""
        mean = self._get_fp32_const("mean", x.device)
        std = self._get_fp32_const("std", x.device)
        x = x.to(torch.float32) * std + mean
        logid = self.indices.get("logid", [])
        if len(logid) > 0:
            v = x[:, :, logid].clamp(min=0, max=7)
            x[:, :, logid] = torch.expm1(v)
        return self._reset_output(x)
