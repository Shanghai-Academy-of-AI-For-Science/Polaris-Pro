"""Mol AR decoder for text → SMILES generation.

Operates on the whole-atom SMILES vocab from
:mod:`qwenvl.modalities.mol.tokenizer`.

Architecture (configurable via the AR-specific hyperparams in
``Qwen3VLMolConfig``):

* AA-token-style embedding (size ``mol_vocab_size`` from tokenizer/config) → 768-d hidden
* 6-layer pre-norm transformer
* Causal self-attention with RoPE (per-layer ``RotaryEmbedding1D``)
* Per-layer cross-attention to LLM hidden states (``kv_proj`` 3584→768
  applied once, cached at prefill time for inference speedup)
* Linear LM head into the mol vocab

Inference
---------
The decoder runs an independent autoregressive loop in mol-vocab space
(:meth:`generate_smiles`) — it does NOT scatter into the LLM vocab.
SMILES generation is therefore decoupled from the LLM tokenizer; the
LLM only contributes the conditioning hidden states via cross-attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qwenvl.registry.base import BaseDecoder
from qwenvl.registry.registry import ComponentRegistry
from qwenvl.registry.decoder_blocks import RotaryEmbedding1D, BioDecoderLayer

from .tokenizer import (
    MOL_VOCAB_SIZE,
    MOL_PAD_ID,
    MOL_CLS_ID,
    MOL_SEP_ID,
)


@ComponentRegistry.register_decoder("mol_ar_decoder")
class MolARDecoder(BaseDecoder):
    """Autoregressive SMILES decoder with cross-attention to a frozen LLM.

    Operates on the whole-atom SMILES vocab.  Designed for the
    "frozen-LLM" setting where the LLM cannot itself learn SMILES
    syntax (branches, ring closures, stereo) — this decoder takes over
    that responsibility while still benefitting from the LLM's text
    understanding via cross-attention.
    """

    def __init__(
        self,
        llm_hidden_size: int = 3584,
        decoder_hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_mult: int = 4,
        mol_vocab_size: int = MOL_VOCAB_SIZE,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        cls_id: int = MOL_CLS_ID,
        **kwargs,
    ):
        super().__init__()
        self._output_size = mol_vocab_size
        self.cls_id = cls_id
        self.llm_hidden_size = llm_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.tok_embed = nn.Embedding(mol_vocab_size, decoder_hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        # Compress LLM hidden 3584 → 768 once (per forward) so cross-attn
        # KV cost stays bounded.  LayerNorm at the output stabilises early
        # training when the decoder is still random.
        self.kv_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size),
        )

        self.rope = RotaryEmbedding1D(
            decoder_hidden_size // num_heads, max_seq_length
        )
        self.layers = nn.ModuleList([
            BioDecoderLayer(decoder_hidden_size, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(decoder_hidden_size)
        self.head = nn.Linear(decoder_hidden_size, mol_vocab_size, bias=False)

        # Cross-attn KV cache for inference — populated by the model
        # wrapper at the prefill step, cleared between ``generate()`` calls.
        self._cached_llm_kv: Optional[Tensor] = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def output_size(self) -> int:
        return self._output_size

    def reset_generation_cache(self):
        """Drop the cached LLM KV.  Called at the start/end of each
        ``generate_smiles()`` call to prevent prompt-leak between samples."""
        self._cached_llm_kv = None

    @torch.no_grad()
    def generate_smiles(
        self,
        hidden_states_full: Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """Generate a SMILES token id sequence in mol-vocab space.

        The decoder runs its own AR loop conditioned on the LLM hidden
        states; output is a ``[B, T]`` tensor of mol-vocab ids (excluding
        the seeded ``<cls>``, including the terminal ``<sep>`` if produced).

        Args:
            hidden_states_full: ``[B, L_llm, llm_hidden]`` LLM hidden states
                covering the full prompt — used as the cross-attn key/value
                source (cached after the first call until ``reset_generation_cache``).
            max_new_tokens: Hard cap on generated tokens (excluding cls).
            do_sample / temperature / top_k / top_p: Standard sampling
                controls.  Greedy if ``do_sample`` is False.
        """
        device = hidden_states_full.device
        B = hidden_states_full.shape[0]

        # Cache the cross-attn KV once for the whole AR loop.
        if self._cached_llm_kv is None:
            self._cached_llm_kv = self.kv_proj(hidden_states_full).detach()
        llm_kv = self._cached_llm_kv

        # Seed with <cls>; outputs go straight into mol-vocab id buffer.
        cur = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        out_ids: list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            x = self.tok_embed(cur)
            for layer in self.layers:
                x = layer(x, llm_kv, self.rope, kv_padding_mask=None)
            x = self.final_norm(x)
            logits = self.head(x[:, -1, :])  # [B, V_mol]

            if do_sample:
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    logits = torch.where(logits < v[:, -1:], torch.full_like(logits, float("-inf")), logits)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    keep = cum - probs <= top_p
                    keep[:, 0] = True  # always keep top-1
                    sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_id = logits.argmax(dim=-1)

            # Once a sample is finished, force <pad> so out_ids is rectangular.
            next_id = torch.where(finished, torch.full_like(next_id, MOL_PAD_ID), next_id)
            out_ids.append(next_id)
            finished = finished | (next_id == MOL_SEP_ID)
            if finished.all():
                break

            cur = torch.cat([cur, next_id.unsqueeze(-1)], dim=1)

        return torch.stack(out_ids, dim=1) if out_ids else torch.empty(
            (B, 0), dtype=torch.long, device=device
        )


__all__ = ["MolARDecoder", "MOL_VOCAB_SIZE"]
