"""Shared transformer-decoder building blocks for AR modality decoders.

Used by ``ProteinARDecoder`` and ``MolARDecoder`` (and any future bio AR
decoder following the same Flamingo-style "frozen LLM + per-layer cross-
attention" pattern).  Keeping these in one place avoids drift between
modalities and makes architectural improvements (e.g. flash-attn switch,
new normalisation) propagate uniformly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RotaryEmbedding1D(nn.Module):
    """1-D rotary positional embedding for the AR decoder's self-attention.

    ``forward(x)`` accepts ``x`` of shape ``[B, heads, L, head_dim]`` and
    returns the same shape with rotation applied.  Cache is built lazily
    on the first forward call and recomputed any time the cache shape,
    device, or finite-ness diverges from the current input.

    History note: ``inv_freq`` / ``cos_cached`` / ``sin_cached`` used to
    be ``register_buffer(..., persistent=False)``. transformers >=5.0's
    ``from_pretrained`` materialization path can leave such buffers
    pointing at uninitialized memory (NaN/Inf), which then poisons the
    entire encoder/decoder. Plain Python attributes built lazily on the
    target device avoid that path entirely.
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self._dim = int(dim)
        self._init_max_len = int(max_len)
        self._cos_cache: Tensor | None = None
        self._sin_cache: Tensor | None = None

    def _ensure_cache(self, seq_len: int, device: torch.device) -> None:
        cache = self._cos_cache
        need_rebuild = (
            cache is None
            or cache.shape[0] < seq_len
            or cache.device != device
            or not torch.isfinite(cache).all()
        )
        if not need_rebuild:
            return
        target_len = max(seq_len, self._init_max_len)
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, self._dim, 2, dtype=torch.float32, device=device) / self._dim)
        )
        t = torch.arange(target_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[2]
        self._ensure_cache(seq_len, x.device)
        cos = self._cos_cache[:seq_len].to(x.dtype)
        sin = self._sin_cache[:seq_len].to(x.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return x * cos + self._rotate_half(x) * sin


class BioDecoderLayer(nn.Module):
    """Pre-norm transformer decoder block: causal self-attn → cross-attn → FFN.

    Shared by Protein / Mol AR decoders.  ``forward`` takes the decoder
    hidden ``x``, the projected LLM key/value ``llm_kv``, a ``RotaryEmbedding1D``
    instance, and an optional KV padding mask (True = attend).
    """

    def __init__(self, dim: int, heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        if dim % heads != 0:
            raise ValueError(
                f"decoder_hidden_size={dim} must be divisible by num_heads={heads}"
            )

        # Self-attention (causal, RoPE applied to q/k)
        self.norm1 = nn.LayerNorm(dim)
        self.self_q = nn.Linear(dim, dim, bias=True)
        self.self_k = nn.Linear(dim, dim, bias=True)
        self.self_v = nn.Linear(dim, dim, bias=True)
        self.self_o = nn.Linear(dim, dim, bias=True)

        # Cross-attention (Q from this decoder, K/V from LLM hidden)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=True)
        self.cross_k = nn.Linear(dim, dim, bias=True)
        self.cross_v = nn.Linear(dim, dim, bias=True)
        self.cross_o = nn.Linear(dim, dim, bias=True)

        # FFN
        self.norm3 = nn.LayerNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, L, D] → [B, heads, L, head_dim]."""
        B, L, _ = x.shape
        return x.view(B, L, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """[B, heads, L, head_dim] → [B, L, D]."""
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.heads * self.head_dim)

    def forward(
        self,
        x: Tensor,                              # [B, Lq, D]
        llm_kv: Tensor,                         # [B, Lk, D]  (already kv_proj'd)
        rope: RotaryEmbedding1D,
        kv_padding_mask: Optional[Tensor] = None,  # [B, Lk] — True means VALID
    ) -> Tensor:
        # ---- Self-attention with causal mask + RoPE ----
        h = self.norm1(x)
        q = self._split_heads(self.self_q(h))
        k = self._split_heads(self.self_k(h))
        v = self._split_heads(self.self_v(h))
        q = rope(q)
        k = rope(k)
        # SDPA's is_causal=True builds the causal mask internally without
        # materializing a [Lq, Lq] tensor — important when Lq approaches 16K.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = self._merge_heads(attn_out)
        x = x + self.dropout(self.self_o(attn_out))

        # ---- Cross-attention to LLM hidden ----
        h = self.norm2(x)
        kv = self.norm_kv(llm_kv)
        cq = self._split_heads(self.cross_q(h))
        ck = self._split_heads(self.cross_k(kv))
        cv = self._split_heads(self.cross_v(kv))
        # SDPA bool-mask convention: True = element participates in attention.
        # We pass the mask following the SAME convention (True = valid).
        # Shape [B, 1, 1, Lk] is a strided view that SDPA broadcasts to
        # [B, heads, Lq, Lk] internally without materializing the full
        # INT32-sized mask.
        attn_mask = None
        if kv_padding_mask is not None:
            attn_mask = kv_padding_mask[:, None, None, :]
        cattn_out = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=attn_mask)
        cattn_out = self._merge_heads(cattn_out)
        x = x + self.dropout(self.cross_o(cattn_out))

        # ---- FFN ----
        x = x + self.ffn(self.norm3(x))
        return x


__all__ = ["RotaryEmbedding1D", "BioDecoderLayer"]
