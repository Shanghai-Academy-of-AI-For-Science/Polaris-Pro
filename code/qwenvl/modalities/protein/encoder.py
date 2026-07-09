"""
Protein encoders.

1. **ProteinEncoder** — ESM2 backbone + perceiver resampler. The backbone
   produces per-residue embeddings; the resampler compresses them into K
   fixed latent tokens.
2. **ProteinConvFormer** — lightweight self-contained encoder (conv stem +
   Transformer + perceiver resampler), no ESM dependency.

Selected by ``protein_backbone_name`` in config (``"esm2_*"`` → ProteinEncoder,
``"convformer"`` → ProteinConvFormer).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


# ============================================================================
# Perceiver Resampler
# ============================================================================

class _LatentSelfAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN block used by ``_LatentResampler`` to
    iteratively refine latent tokens after the initial cross-attention.

    Mirrors the main Transformer block but operates on the K-token latent
    sequence, so attention cost is O(K²) — cheap even with K=64.
    """

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

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class _LatentResampler(nn.Module):
    """Perceiver-style cross-attention that compresses L encoder tokens into K latent tokens.

    With ``num_layers=1`` (default), the resampler runs a single cross-attn
    + FFN block — backward-compatible with all existing checkpoints.
    With ``num_layers>=2``, the cross-attn is followed by ``num_layers-1``
    self-attention refinement blocks (Flamingo / Perceiver-IO style), giving
    the latent tokens iterative passes to absorb residue-level information.
    Useful when K is small relative to sequence length (e.g. K=64 over a
    2048-residue protein → 32× compression).
    """

    def __init__(
        self,
        dim: int,
        num_latent_tokens: int = 32,
        heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
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
        # Optional iterative self-attention refinement on the K latent tokens.
        n_refine = max(0, int(num_layers) - 1)
        self.refine_layers = nn.ModuleList([
            _LatentSelfAttnBlock(dim, heads, ffn_mult=4, dropout=dropout)
            for _ in range(n_refine)
        ])
        # Set externally by ProteinConvFormer / training script when memory
        # is tight (long sequences + K=64 + multi-layer perceiver).
        self.gradient_checkpointing = False

    def forward(self, encoder_out: Tensor, key_padding_mask: Tensor | None = None):
        B = encoder_out.shape[0]
        q = self.norm_q(self.latent_queries.expand(B, -1, -1))
        kv = self.norm_kv(encoder_out)
        h, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        h = q + h
        h = h + self.ffn(h)
        # Iterative self-attention refinement
        for layer in self.refine_layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return h


# ============================================================================
# Protein Encoder (ESM2 + Resampler)
# ============================================================================

class ProteinEncoder(nn.Module):
    """Protein encoder: ESM2 backbone + perceiver resampler.

    The backbone is set externally via ``set_backbone()``. Forward matches the
    ModalityRouter contract: encoder(input_ids, attention_mask) -> (latent, mask).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.protein_encoder_hidden_size

        self.backbone: nn.Module | None = None
        self._backbone_frozen_cached: bool | None = None

        # Aim for head_dim ≈ 64 but fall back to the largest divisor of
        # ``hidden`` so MultiheadAttention's ``embed_dim % num_heads == 0``
        # check passes for non-multiple-of-64 sizes (e.g. ESM2-35M's 480).
        hidden = config.protein_encoder_hidden_size
        n_heads = max(1, hidden // 64)
        while n_heads > 1 and hidden % n_heads != 0:
            n_heads -= 1
        self.resampler = _LatentResampler(
            dim=config.protein_encoder_hidden_size,
            num_latent_tokens=config.num_latent_tokens,
            heads=n_heads,
            dropout=getattr(config, "dropout", 0.1),
            num_layers=getattr(config, "num_resampler_layers", 1),
        )
        self.resampler.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def set_backbone(self, backbone: nn.Module):
        """Set the protein backbone (ESM2). Called from register_modality."""
        self.backbone = backbone
        self._backbone_frozen_cached = None  # Reset cache on backbone change

    def _is_backbone_frozen(self) -> bool:
        """Check if backbone has no trainable params (cached for performance)."""
        if self._backbone_frozen_cached is None:
            self._backbone_frozen_cached = not any(
                p.requires_grad for p in self.backbone.parameters()
            )
        return self._backbone_frozen_cached

    def forward(self, protein_input_ids: Tensor, protein_attention_mask: Tensor):
        """
        Args:
            protein_input_ids:      [B, L] ESM token ids (with <cls> and <eos>)
            protein_attention_mask: [B, L] 1 = real token, 0 = padding
        Returns:
            latent:      [B, K, D] compressed latent tokens
            latent_mask: [B, K]    all-ones mask (fixed length)
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded. Call set_backbone() first.")

        device = protein_input_ids.device
        B = protein_input_ids.shape[0]

        sequence_id = protein_attention_mask.bool()

        # Memory optimization: skip autograd graph for frozen backbone
        if self._is_backbone_frozen():
            with torch.no_grad():
                output = self.backbone(
                    sequence_tokens=protein_input_ids,
                    sequence_id=sequence_id,
                )
                embeddings = output.embeddings.detach()
        else:
            output = self.backbone(
                sequence_tokens=protein_input_ids,
                sequence_id=sequence_id,
            )
            embeddings = output.embeddings  # [B, L, d_model]

        # Attention mask for resampler: True = pad (to ignore)
        pad_mask_bool = protein_attention_mask == 0

        latent = self.resampler(embeddings, key_padding_mask=pad_mask_bool)

        K = latent.shape[1]
        latent_mask = torch.ones(B, K, dtype=protein_attention_mask.dtype, device=device)
        return latent, latent_mask


ProteinESMCEncoder = ProteinEncoder


# ============================================================================
# ProteinConvFormer — lightweight self-contained encoder
# ============================================================================

class _RotaryEmbedding1D(nn.Module):
    """Simple 1-D rotary position embedding.

    Caches are built lazily on the first forward call and recomputed any
    time the cache shape, dtype, or finite-ness diverges from the current
    input tensor. This survives transformers >=5.0's ``from_pretrained``
    materialization path, which can leave ``persistent=False`` buffers
    holding garbage memory if the buffer was registered before the module
    moved to ``meta`` device.
    """

    def __init__(self, dim: int, max_len: int = 2048):
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


class _TransformerBlock(nn.Module):
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
            # SDPA bool-mask convention: True = element participates in
            # attention (the OPPOSITE of nn.MultiheadAttention's
            # key_padding_mask, which uses True = ignore). Our incoming
            # ``key_padding_mask`` follows the nn.MultiheadAttention
            # convention (True = pad), so invert before broadcasting.
            # Shape [B, 1, 1, L] is a strided view that SDPA broadcasts to
            # [B, heads, L_q, L] internally without materializing a dense
            # tensor whose linear index would overflow INT32 once
            # B*heads*L*L > 2^31 (e.g. L>=8192 with multi-head batches).
            attn_mask = (~key_padding_mask)[:, None, None, :]
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.attn.out_proj(attn_out)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# Hand-crafted physicochemical features for the 20 standard amino acids.
# Used by ProteinConvFormer when ``use_physicochemical_embed=True`` to
# inject inductive bias absent from a randomly-initialized token embedding.
# All values are public biochemistry constants (Kyte-Doolittle hydropathy,
# side-chain charge at pH 7, side-chain volume Å³, Grantham polarity index,
# aromaticity flag, sulfur-containing flag).  Values are normalized to ~unit
# scale so the projection layer doesn't have to compensate for skewed
# magnitudes.
#
# Layout: [num_features=6] per row.  Rows are placed at the *encoder* token
# id corresponding to each amino acid (see ``_SEQUENCE_VOCAB`` in
# ``processor.py``).  Special tokens (<cls>=0, <pad>=1, <eos>=2, <unk>=3,
# <mask>=32) and ambiguous AAs (X=24, B=25, U=26, Z=27, O=28) get all-zero
# features so their embedding is determined entirely by the learned
# ``tok_embed`` lookup.
_PROT_PHYSICO_NUM_FEATURES = 6


def _build_physicochemical_table(vocab_size: int):
    """Build the [vocab_size, 6] physicochemical lookup tensor.

    Imports the encoder vocab list lazily (avoids circular imports during
    package init) and writes one row per standard amino acid.
    """
    from .processor import _SEQUENCE_VOCAB
    # Standard AA features. Column order: hydropathy, charge, volume, polarity, aromatic, sulfur.
    # Hydropathy: Kyte-Doolittle scale / 4.5 (so values fall in [-1, 1]).
    # Charge:     +1 / -1 / 0 at pH 7.
    # Volume:     side-chain volume (Å³) / 230 (max ~Trp 228).
    # Polarity:   Grantham polarity / 13 (max ~Lys 12.3).
    # Aromatic:   {F, W, Y, H} = 1.
    # Sulfur:     {C, M} = 1.
    aa_feats = {
        # AA: [hydropathy, charge, volume, polarity, aromatic, sulfur]
        "A": [ 1.8 / 4.5,  0.0,  88.6 / 230.0,  8.1 / 13.0, 0.0, 0.0],
        "R": [-4.5 / 4.5, +1.0, 173.4 / 230.0, 10.5 / 13.0, 0.0, 0.0],
        "N": [-3.5 / 4.5,  0.0, 114.1 / 230.0, 11.6 / 13.0, 0.0, 0.0],
        "D": [-3.5 / 4.5, -1.0, 111.1 / 230.0, 13.0 / 13.0, 0.0, 0.0],
        "C": [ 2.5 / 4.5,  0.0, 108.5 / 230.0,  5.5 / 13.0, 0.0, 1.0],
        "E": [-3.5 / 4.5, -1.0, 138.4 / 230.0, 12.3 / 13.0, 0.0, 0.0],
        "Q": [-3.5 / 4.5,  0.0, 143.8 / 230.0, 10.5 / 13.0, 0.0, 0.0],
        "G": [-0.4 / 4.5,  0.0,  60.1 / 230.0,  9.0 / 13.0, 0.0, 0.0],
        "H": [-3.2 / 4.5,  0.0, 153.2 / 230.0, 10.4 / 13.0, 1.0, 0.0],
        "I": [ 4.5 / 4.5,  0.0, 166.7 / 230.0,  5.2 / 13.0, 0.0, 0.0],
        "L": [ 3.8 / 4.5,  0.0, 166.7 / 230.0,  4.9 / 13.0, 0.0, 0.0],
        "K": [-3.9 / 4.5, +1.0, 168.6 / 230.0, 11.3 / 13.0, 0.0, 0.0],
        "M": [ 1.9 / 4.5,  0.0, 162.9 / 230.0,  5.7 / 13.0, 0.0, 1.0],
        "F": [ 2.8 / 4.5,  0.0, 189.9 / 230.0,  5.2 / 13.0, 1.0, 0.0],
        "P": [-1.6 / 4.5,  0.0, 112.7 / 230.0,  8.0 / 13.0, 0.0, 0.0],
        "S": [-0.8 / 4.5,  0.0,  89.0 / 230.0,  9.2 / 13.0, 0.0, 0.0],
        "T": [-0.7 / 4.5,  0.0, 116.1 / 230.0,  8.6 / 13.0, 0.0, 0.0],
        "W": [-0.9 / 4.5,  0.0, 227.8 / 230.0,  5.4 / 13.0, 1.0, 0.0],
        "Y": [-1.3 / 4.5,  0.0, 193.6 / 230.0,  6.2 / 13.0, 1.0, 0.0],
        "V": [ 4.2 / 4.5,  0.0, 140.0 / 230.0,  5.9 / 13.0, 0.0, 0.0],
    }
    table = torch.zeros(vocab_size, _PROT_PHYSICO_NUM_FEATURES, dtype=torch.float32)
    for aa, feats in aa_feats.items():
        if aa in _SEQUENCE_VOCAB:
            idx = _SEQUENCE_VOCAB.index(aa)
            table[idx] = torch.tensor(feats, dtype=torch.float32)
    return table


# Default vocab size, derived from processor._SEQUENCE_VOCAB so the two stay
# in sync automatically. Hard-coding is fragile: a previous bug had this set
# to 31 while the tokenizer emitted IDs up to 32, crashing the embedding.
from .processor import _SEQUENCE_VOCAB as _PROTEIN_SEQ_VOCAB
_PROTEIN_CONVFORMER_VOCAB_SIZE = len(_PROTEIN_SEQ_VOCAB)


class ProteinConvFormer(nn.Module):
    """Lightweight protein encoder: token embed + conv stem + Transformer + latent resampler.

    Architecturally symmetric with ``RNAConvFormer``, fully self-contained
    with no ESM3 dependency.  ~50–100 M parameters depending on config.

    Forward signature matches the ModalityRouter contract:
        encoder(input_ids, attention_mask) -> (latent [B, K, D], mask [B, K])
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.protein_encoder_hidden_size
        self.hidden_size = dim

        # vocab_size persists in config (Qwen3VLProteinConfig.protein_convformer_vocab_size,
        # defaults to len(_SEQUENCE_VOCAB)) so checkpoints round-trip correctly even
        # if the tokenizer vocab grows later.
        vocab_size = config.protein_convformer_vocab_size
        max_seq_len = getattr(config, "protein_max_seq_length", 2048)

        self.tok_embed = nn.Embedding(vocab_size, dim, padding_idx=1)  # pad=1 in _SEQUENCE_VOCAB
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Optional physicochemical-feature embedding: inject
        # hand-crafted residue properties so the random-init encoder has a
        # head start before SFT. Disabled when the config lacks the flag
        # (i.e. when loading an old checkpoint that wasn't trained with it).
        self.use_physicochemical_embed = bool(
            getattr(config, "use_physicochemical_embed", False)
        )
        if self.use_physicochemical_embed:
            self.physico_proj = nn.Linear(_PROT_PHYSICO_NUM_FEATURES, dim, bias=False)
            self.register_buffer(
                "physico_table",
                _build_physicochemical_table(vocab_size),
                persistent=False,
            )
        else:
            self.physico_proj = None

        conv_kernel = getattr(config, "conv_kernel_size", 7)
        self.conv_stem = nn.ModuleList([
            _DepthwiseSeparableConv1d(dim, conv_kernel),
            _DepthwiseSeparableConv1d(dim, conv_kernel),
        ])

        n_heads = getattr(config, "num_attention_heads", max(1, dim // 64))
        n_layers = getattr(config, "num_encoder_layers", 12)
        dropout = getattr(config, "dropout", 0.1)

        self.rope = _RotaryEmbedding1D(dim // n_heads, max_seq_len)
        self.layers = nn.ModuleList([
            _TransformerBlock(dim, n_heads, ffn_mult=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

        # Multi-layer perceiver resampler: default 1 keeps backward
        # compat with older checkpoints; new training scripts pass
        # ``num_resampler_layers=2`` for better residue-level fidelity at
        # high compression ratios (e.g. K=64 over 2048 residues).
        n_resampler_layers = getattr(config, "num_resampler_layers", 1)
        self.resampler = _LatentResampler(
            dim,
            config.num_latent_tokens,
            n_heads,
            dropout,
            num_layers=n_resampler_layers,
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

    def forward(self, protein_input_ids: Tensor, protein_attention_mask: Tensor):
        """
        Args:
            protein_input_ids:      [B, L] char-level token ids (from ProteinTokenizer)
            protein_attention_mask: [B, L] 1 = real token, 0 = padding
        Returns:
            latent:      [B, K, D] compressed latent tokens
            latent_mask: [B, K]    all-ones mask (fixed length)
        """
        device = protein_input_ids.device
        B, L = protein_input_ids.shape

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        # Guard against sequences exceeding pos_embed table size
        if L > self.pos_embed.num_embeddings:
            if not getattr(self, "_pos_clamp_warned", False):
                import logging as _logging
                _logging.getLogger("ProteinConvFormer").warning(
                    "Input length L=%d exceeds pos_embed cap=%d; clamping. "
                    "Residues beyond position %d share one position embedding "
                    "(silent position aliasing). Train with a larger "
                    "--protein_max_seq_length to avoid this.",
                    L, self.pos_embed.num_embeddings,
                    self.pos_embed.num_embeddings - 1,
                )
                self._pos_clamp_warned = True
            pos_ids = pos_ids.clamp(max=self.pos_embed.num_embeddings - 1)
        x = self.tok_embed(protein_input_ids) + self.pos_embed(pos_ids)

        # Add physicochemical-property embedding (zero rows for special /
        # ambiguous AAs, so this only contributes for the 20 standard AAs).
        if self.physico_proj is not None:
            physico = self.physico_table[protein_input_ids]  # [B, L, 6]
            x = x + self.physico_proj(physico.to(x.dtype))

        pad_mask_bool = protein_attention_mask == 0

        for conv in self.conv_stem:
            x = conv(x, protein_attention_mask)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, self.rope, pad_mask_bool, use_reentrant=False)
            else:
                x = layer(x, self.rope, key_padding_mask=pad_mask_bool)

        x = self.final_norm(x)

        latent = self.resampler(x, key_padding_mask=pad_mask_bool)

        K = latent.shape[1]
        latent_mask = torch.ones(B, K, dtype=protein_attention_mask.dtype, device=device)
        return latent, latent_mask
