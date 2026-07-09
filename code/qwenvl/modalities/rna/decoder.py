"""
RNA decoder head: a small linear layer that maps LLM hidden states
to the RNA char vocab (8 classes: pad/cls/A/U/G/C/N/sep).

Registered as a BaseDecoder so the framework can discover it via
ModalitySpec.decoder_cls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qwenvl.registry.base import BaseDecoder
from qwenvl.registry.registry import ComponentRegistry

RNA_VOCAB_SIZE = 8

# RNA char tokenizer ID <-> LLM nucleotide token ID
# Qwen3-VL tokenizer: A=32, C=34, G=38, U=52, N=45 (single-char tokens).
RNA_CHAR_TO_LLM = {2: 32, 3: 52, 4: 38, 5: 34, 6: 45}   # A->32, U->52, G->38, C->34, N->45
LLM_TO_RNA_CHAR = {v: k for k, v in RNA_CHAR_TO_LLM.items()}
RNA_CHAR_SEP_ID = 7


@ComponentRegistry.register_decoder("rna_lm_decoder")
class RNALMDecoder(BaseDecoder):
    """Linear decoder head for RNA sequence generation.

    Maps LLM hidden states to a small (8-class) RNA char vocabulary.
    During inference the logits are expanded back to the full LLM vocab
    via ``logits_to_vocab_space``.
    """

    def __init__(self, llm_hidden_size: int = 3584, rna_vocab_size: int = RNA_VOCAB_SIZE, **kwargs):
        super().__init__()
        self.head = nn.Linear(llm_hidden_size, rna_vocab_size, bias=False)
        self._output_size = rna_vocab_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def logits_to_vocab_space(
        self,
        hidden_states: Tensor,
        full_vocab_size: int,
        eos_token_id: int = None,
        **kwargs,
    ) -> Tensor:
        """Map 8-dim RNA logits to full LLM vocab for autoregressive generation.

        Uses pre-built index tensors and scatter to avoid creating a full
        vocab-size tensor from scratch each call.

        Note (DNA generation): this decoder is shared by the DNA modality
        via the ModalityRouter alias. Internally T is normalized to U during
        teacher-forcing, so for DNA tasks the model emits 'U' tokens. Callers
        that need DNA strings should post-process the decoded text with
        ``str.replace("U", "T")``.
        """
        rna_logits = self.head(hidden_states)
        B, L, _ = rna_logits.shape

        # Build index mapping once (cached on first call)
        if not hasattr(self, "_vocab_map_src_ids"):
            src_ids = list(RNA_CHAR_TO_LLM.keys())
            tgt_ids = list(RNA_CHAR_TO_LLM.values())
            if eos_token_id is not None:
                src_ids.append(RNA_CHAR_SEP_ID)
                tgt_ids.append(eos_token_id)
            self._vocab_map_src_ids = src_ids
            self._vocab_map_tgt_ids = tgt_ids
            self._vocab_map_eos = eos_token_id
        elif eos_token_id != self._vocab_map_eos:
            # EOS changed (rare) — rebuild
            src_ids = list(RNA_CHAR_TO_LLM.keys())
            tgt_ids = list(RNA_CHAR_TO_LLM.values())
            if eos_token_id is not None:
                src_ids.append(RNA_CHAR_SEP_ID)
                tgt_ids.append(eos_token_id)
            self._vocab_map_src_ids = src_ids
            self._vocab_map_tgt_ids = tgt_ids
            self._vocab_map_eos = eos_token_id

        src_ids = self._vocab_map_src_ids
        tgt_ids = self._vocab_map_tgt_ids

        # Only allocate the full tensor; use vectorized indexing instead of loop
        logits = torch.full(
            (B, L, full_vocab_size), float("-inf"),
            device=rna_logits.device, dtype=rna_logits.dtype,
        )
        logits[..., tgt_ids] = rna_logits[..., src_ids]
        return logits
