"""
RNA data processor: char-level tokenizer and chat-template helpers.

Registered as a BaseProcessor so the framework can use it via ModalitySpec.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from mkb.registry.base import BaseProcessor
from mkb.registry.registry import ComponentRegistry

_RNA_CHAR_TO_ID = {
    "<pad>": 0, "<cls>": 1, "A": 2, "U": 3, "G": 4, "C": 5, "N": 6, "<sep>": 7,
}
_RNA_ID_TO_CHAR = {v: k for k, v in _RNA_CHAR_TO_ID.items()}

# Qwen3-VL tokenizer single-char IDs: A=32, C=34, G=38, U=52, N=45
# (verified via tokenizer.encode("N", add_special_tokens=False) -> [45]).
# N represents an unknown/ambiguous nucleotide and appears in real RNA/DNA
# datasets; mapping it through the same path keeps T→U normalization while
# allowing N in both encoder inputs and assistant outputs.
RNA_TOKEN_DICT = {"A": 32, "C": 34, "G": 38, "U": 52, "N": 45}
_LLM_TO_RNA_CHAR = {32: 2, 52: 3, 38: 4, 34: 5, 45: 6}
_RNA_CHAR_SEP_ID = 7


class RNACharTokenizer:
    """Minimal char-level tokenizer for RNA sequences (A/U/G/C/N).

    Produces ``[CLS] base_1 base_2 ... base_L [SEP]`` id sequences,
    compatible with ``RNAConvFormer``'s embedding layer.
    """

    pad_token_id = _RNA_CHAR_TO_ID["<pad>"]
    cls_token_id = _RNA_CHAR_TO_ID["<cls>"]
    sep_token_id = _RNA_CHAR_TO_ID["<sep>"]
    vocab = _RNA_CHAR_TO_ID

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
                if ch == "T":
                    ch = "U"
                ids.append(_RNA_CHAR_TO_ID.get(ch, _RNA_CHAR_TO_ID["N"]))
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


@ComponentRegistry.register_processor("rna_char_processor")
class RNACharProcessor(BaseProcessor):
    """BaseProcessor adapter for RNA/DNA char-level tokenization."""

    def __init__(self, max_length: int = 2048, num_latent_tokens: int = 16, **kwargs):
        self._tokenizer = RNACharTokenizer()
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
            valid_chars = set("AUCGT")
            real_token_ids = []
            for ch in seq.upper():
                if ch not in valid_chars:
                    raise ValueError(f"Invalid RNA character: {ch}")
                if ch == "T":
                    ch = "U"
                real_token_ids.append(RNA_TOKEN_DICT[ch])
            placeholder = "<|image_pad|>" * len(seq)
            return placeholder, real_token_ids
        else:
            K = self._num_latent_tokens
            placeholder = "<|vision_start|>" + "<|image_pad|>" * K + "<|vision_end|>"
            return placeholder, None

    @property
    def modality_name(self) -> str:
        return "rna"
