"""
Protein data processor: pure-Python character-level tokenizer for protein sequences.

Replaces the Rust-based EsmSequenceTokenizer (PreTrainedTokenizerFast) with a
simple Python dict lookup. The ESMC tokenizer uses BPE with NO merges on a
31-token amino acid vocabulary, making it trivially reproducible without any
Rust dependency. This eliminates fork() deadlocks when used inside DataLoader
worker processes (the Rust tokenizers library has known issues with forking).
"""

from typing import Any, Dict, List

import torch

# Must match SEQUENCE_VOCAB in encoder.py exactly.
_SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

_TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(_SEQUENCE_VOCAB)}
_CLS_ID = _TOKEN_TO_ID["<cls>"]   # 0
_PAD_ID = _TOKEN_TO_ID["<pad>"]   # 1
_EOS_ID = _TOKEN_TO_ID["<eos>"]   # 2
_UNK_ID = _TOKEN_TO_ID["<unk>"]   # 3


class ProteinTokenizer:
    """Pure-Python character-level tokenizer for protein sequences.

    Produces ``[CLS] residue_1 residue_2 ... residue_L [EOS]`` token
    sequences compatible with the ESMC model's embedding layer.

    Functionally equivalent to EsmSequenceTokenizer but implemented
    entirely in Python to avoid Rust tokenizer fork() issues.
    """

    def __init__(self):
        self.pad_token_id = _PAD_ID

    def _encode_one(self, seq: str) -> List[int]:
        """Encode a single protein sequence to token IDs."""
        body = [_TOKEN_TO_ID.get(c, _UNK_ID) for c in seq]
        return [_CLS_ID] + body + [_EOS_ID]

    def __call__(
        self,
        sequences: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 2048,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Tokenize a batch of protein sequences.

        Args:
            sequences: List of amino acid strings.
            padding: Pad to max length in batch.
            truncation: Truncate to max_length.
            max_length: Maximum sequence length (including special tokens).
                Training/inference always pass an explicit value derived from
                ``DataArguments.max_bio_seq_length``; this 2048 default is for
                ad-hoc REPL use only.
            return_tensors: "pt" for PyTorch tensors.

        Returns:
            Dict with "input_ids" and "attention_mask".
        """
        all_ids: List[List[int]] = []
        for seq in sequences:
            tokens = self._encode_one(seq)
            if truncation and len(tokens) > max_length:
                # Keep CLS at start, truncate body, append EOS
                tokens = tokens[:max_length - 1] + [_EOS_ID]
            all_ids.append(tokens)

        if padding:
            max_len = max(len(ids) for ids in all_ids)
            masks = []
            for ids in all_ids:
                pad_len = max_len - len(ids)
                masks.append([1] * len(ids) + [0] * pad_len)
                ids.extend([_PAD_ID] * pad_len)
        else:
            masks = [[1] * len(ids) for ids in all_ids]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": all_ids, "attention_mask": masks}
