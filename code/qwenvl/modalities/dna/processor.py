"""
DNA data processor: char-level tokenizer for DNA sequences (A/T/G/C/N).

Differs from ``RNACharTokenizer`` in two ways:
  * native T character (vocab id 3) — no T→U normalization;
  * any unexpected U (rare in DNA datasets) is mapped to N as a fallback.

The tokenizer matches the embedding layer of ``DNAConvFormer`` (vocab_size=8).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from qwenvl.registry.base import BaseProcessor
from qwenvl.registry.registry import ComponentRegistry


_DNA_CHAR_TO_ID = {
    "<pad>": 0, "<cls>": 1, "A": 2, "T": 3, "G": 4, "C": 5, "N": 6, "<sep>": 7,
}
_DNA_ID_TO_CHAR = {v: k for k, v in _DNA_CHAR_TO_ID.items()}

# Native DNA → Qwen3-VL LLM single-char token ids (verified in data_processor
# PROTEIN_TOKEN_DICT: T=51).  Used to render the assistant-side ground truth
# for teacher forcing (the same ids the LM emits at generation).
DNA_TOKEN_DICT = {"A": 32, "C": 34, "G": 38, "T": 51, "N": 45}
_LLM_TO_DNA_CHAR = {32: 2, 51: 3, 38: 4, 34: 5, 45: 6}
_DNA_CHAR_SEP_ID = 7


class DNACharTokenizer:
    """Minimal char-level tokenizer for DNA sequences (A/T/G/C/N).

    Produces ``[CLS] base_1 base_2 ... base_L [SEP]`` id sequences,
    compatible with ``DNAConvFormer``'s embedding layer. Unknown chars
    (including stray ``U``) fall back to ``N``.
    """

    pad_token_id = _DNA_CHAR_TO_ID["<pad>"]
    cls_token_id = _DNA_CHAR_TO_ID["<cls>"]
    sep_token_id = _DNA_CHAR_TO_ID["<sep>"]
    vocab = _DNA_CHAR_TO_ID

    def __call__(
        self,
        sequences: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 2048,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        all_ids: List[List[int]] = []
        for seq in sequences:
            ids = [self.cls_token_id]
            for ch in seq.upper():
                # No T→U normalization. U is rare in DNA datasets; treat
                # as the unknown-base fallback ``N`` to stay robust.
                if ch == "U":
                    ch = "N"
                ids.append(_DNA_CHAR_TO_ID.get(ch, _DNA_CHAR_TO_ID["N"]))
            ids.append(self.sep_token_id)
            if truncation and len(ids) > max_length:
                ids = ids[: max_length - 1] + [self.sep_token_id]
            all_ids.append(ids)

        if padding:
            max_len = max(len(ids) for ids in all_ids)
            masks = []
            for ids in all_ids:
                pad_len = max_len - len(ids)
                masks.append([1] * len(ids) + [0] * pad_len)
                ids.extend([self.pad_token_id] * pad_len)
        else:
            masks = [[1] * len(ids) for ids in all_ids]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": all_ids, "attention_mask": masks}


@ComponentRegistry.register_processor("dna_char_processor")
class DNACharProcessor(BaseProcessor):
    """BaseProcessor adapter for DNA char-level tokenization."""

    def __init__(self, max_length: int = 2048, num_latent_tokens: int = 16, **kwargs):
        self._tokenizer = DNACharTokenizer()
        self._max_length = max_length
        self._num_latent_tokens = num_latent_tokens

    def process_input(
        self,
        raw_input: Any,
        **kwargs,
    ) -> Dict[str, Tensor]:
        if isinstance(raw_input, str):
            raw_input = [raw_input]
        return self._tokenizer(
            raw_input,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

    def build_placeholder(
        self,
        raw_input: Any,
        is_output: bool = False,
        **kwargs,
    ) -> Tuple[str, Optional[Any]]:
        if is_output:
            seq = raw_input if isinstance(raw_input, str) else raw_input[0]
            valid_chars = set("ATCGN")
            real_token_ids = []
            for ch in seq.upper():
                if ch == "U":
                    ch = "N"
                if ch not in valid_chars:
                    raise ValueError(f"Invalid DNA character: {ch}")
                real_token_ids.append(DNA_TOKEN_DICT[ch])
            placeholder = "<|image_pad|>" * len(seq)
            return placeholder, real_token_ids
        else:
            K = self._num_latent_tokens
            placeholder = "<|vision_start|>" + "<|image_pad|>" * K + "<|vision_end|>"
            return placeholder, None

    @property
    def modality_name(self) -> str:
        return "dna"
