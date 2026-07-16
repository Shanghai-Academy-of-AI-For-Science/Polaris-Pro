"""
RNA encoder: RNAConvFormer — lightweight conv-stem + Transformer + latent resampler.

~50-60 M parameters with default settings (hidden=512, 8 layers).
This is the canonical implementation used by ModalityRouter.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class _RotaryEmbedding1D(nn.Module):
    """Simple 1-D rotary position embedding for the RNA encoder.

    Caches are built lazily on the first forward call and recomputed any
    time the cache shape, dtype, or finite-ness diverges from the current
    input tensor. This survives transformers >=5.0's ``from_pretrained``
    materialization path, which can leave ``persistent=False`` buffers
    holding garbage memory if the buffer was registered before the module
    moved to ``meta`` device.
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self._dim = int(dim)
        self._init_max_len = int(max_len)
        # No tensors created in __init__ — transformers >=5.0 may move the
        # whole module to ``meta`` device during from_pretrained, which
        # turns any tensor created here into a data-less meta tensor that
        # cannot be moved to GPU later. Compute inv_freq fresh on the
        # input device every time the cache needs rebuilding.
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
        # Build inv_freq directly on the target device in fp32 — survives
        # whatever from_pretrained / .to(dtype) did to the module.
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, self._dim, 2, dtype=torch.float32, device=device) / self._dim)
        )
        t = torch.arange(target_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [target_len, dim]
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: Tensor):
        """Apply RoPE to ``x`` of shape [B, heads, L, head_dim]."""
        seq_len = x.shape[2]
        self._ensure_cache(seq_len, x.device)
        cos = self._cos_cache[:seq_len].to(x.dtype)
        sin = self._sin_cache[:seq_len].to(x.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return x * cos + self._rotate_half(x) * sin


class _DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1-D convolution: depthwise → pointwise."""

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.pw = nn.Conv1d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor, mask: Tensor | None = None):
        """x: [B, L, D], mask: [B, L] (1 = keep, 0 = pad)."""
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        h = x.transpose(1, 2)  # [B, D, L]
        h = self.pw(self.act(self.dw(h)))
        h = h.transpose(1, 2)  # [B, L, D]
        return self.norm(x + h)


class _RNATransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE self-attention."""

    def __init__(self, dim: int, heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        ffn_dim = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        rope: _RotaryEmbedding1D,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """x: [B, L, D], key_padding_mask: [B, L] (True = ignore)."""
        h = self.norm1(x)
        B, L, D = h.shape
        heads = self.attn.num_heads
        head_dim = D // heads
        q = h @ self.attn.in_proj_weight[:D].T + self.attn.in_proj_bias[:D]
        k = h @ self.attn.in_proj_weight[D : 2 * D].T + self.attn.in_proj_bias[D : 2 * D]
        v = h @ self.attn.in_proj_weight[2 * D :].T + self.attn.in_proj_bias[2 * D :]
        q = q.view(B, L, heads, head_dim).transpose(1, 2)
        k = k.view(B, L, heads, head_dim).transpose(1, 2)
        v = v.view(B, L, heads, head_dim).transpose(1, 2)
        q = rope(q)
        k = rope(k)
        attn_mask = None
        if key_padding_mask is not None:
            # ``key_padding_mask`` follows the nn.MultiheadAttention convention
            # (True = pad). Build an additive float mask in q.dtype so SDPA
            # adds a bf16-safe finite negative number on padded positions
            # rather than -inf. Newer PyTorch versions route bool attn_mask
            # to the memory-efficient SDPA backend, which under bf16 can
            # return NaN for the entire output even when each query row has
            # at least one valid key (observed in 2.x; same code worked on
            # earlier torch). A finite -1e4 keeps softmax numerically stable.
            attn_mask = torch.zeros(
                key_padding_mask.size(0), 1, 1, key_padding_mask.size(1),
                dtype=q.dtype, device=q.device,
            )
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :], -1e4,
            )
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.attn.out_proj(attn_out)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class _LatentResampler(nn.Module):
    """Perceiver-style cross-attention that compresses L encoder tokens into K latent tokens."""

    def __init__(self, dim: int, num_latent_tokens: int = 16, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(1, num_latent_tokens, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        ffn_dim = dim * 4
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, encoder_out: Tensor, key_padding_mask: Tensor | None = None):
        B = encoder_out.shape[0]
        q = self.norm_q(self.latent_queries.expand(B, -1, -1))
        kv = self.norm_kv(encoder_out)
        h, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        h = q + h
        h = h + self.ffn(h)
        return h


class RNAConvFormer(nn.Module):
    """Lightweight RNA encoder: token embed + conv stem + Transformer + latent resampler.

    ~50-60 M parameters with default settings (hidden=512, 8 layers).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.rna_encoder_hidden_size
        self.hidden_size = dim

        self.tok_embed = nn.Embedding(config.rna_vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Embedding(config.rna_max_seq_length, dim)

        self.conv_stem = nn.ModuleList([
            _DepthwiseSeparableConv1d(dim, config.conv_kernel_size),
            _DepthwiseSeparableConv1d(dim, config.conv_kernel_size),
        ])

        self.rope = _RotaryEmbedding1D(dim // config.num_attention_heads, config.rna_max_seq_length)
        self.layers = nn.ModuleList([
            _RNATransformerBlock(dim, config.num_attention_heads, ffn_mult=4, dropout=config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

        self.resampler = _LatentResampler(
            dim, config.num_latent_tokens, config.num_attention_heads, config.dropout,
        )

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, rna_input_ids: Tensor, rna_attention_mask: Tensor):
        """
        Args:
            rna_input_ids:      [B, L]  char-level token ids
            rna_attention_mask: [B, L]  1 = real token, 0 = padding
        Returns:
            latent:             [B, K, D]  compressed latent tokens
            latent_mask:        [B, K]     all-ones mask (fixed length)
        """
        device = rna_input_ids.device
        B, L = rna_input_ids.shape

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        # Guard against sequences exceeding pos_embed table size
        if L > self.pos_embed.num_embeddings:
            if not getattr(self, "_pos_clamp_warned", False):
                import logging as _logging
                _logging.getLogger("RNAConvFormer").warning(
                    "Input length L=%d exceeds pos_embed cap=%d; clamping. "
                    "Residues beyond position %d share one position embedding "
                    "(silent position aliasing). Train with a larger "
                    "--rna_max_seq_length / --dna_max_seq_length to avoid this.",
                    L, self.pos_embed.num_embeddings,
                    self.pos_embed.num_embeddings - 1,
                )
                self._pos_clamp_warned = True
            pos_ids = pos_ids.clamp(max=self.pos_embed.num_embeddings - 1)
        x = self.tok_embed(rna_input_ids) + self.pos_embed(pos_ids)

        pad_mask_bool = rna_attention_mask == 0

        for conv in self.conv_stem:
            x = conv(x, rna_attention_mask)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, self.rope, pad_mask_bool, use_reentrant=False)
            else:
                x = layer(x, self.rope, key_padding_mask=pad_mask_bool)

        x = self.final_norm(x)

        latent = self.resampler(x, key_padding_mask=pad_mask_bool)

        K = latent.shape[1]
        latent_mask = torch.ones(B, K, dtype=rna_attention_mask.dtype, device=device)
        return latent, latent_mask


# Backward-compatible alias
Qwen3VLRNAEncoder = RNAConvFormer
