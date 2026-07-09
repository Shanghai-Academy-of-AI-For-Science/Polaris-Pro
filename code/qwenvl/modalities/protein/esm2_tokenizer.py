"""ESM2-compatible protein tokenizer.

Mirrors the structure of :class:`qwenvl.modalities.protein.processor.ProteinTokenizer`
but uses the **ESM2 vocabulary and id ordering** as defined by fair-esm's
``Alphabet.from_architecture("ESM-1b")`` (which ESM2 inherits) and confirmed by
``transformers.AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")``.

Why pure Python instead of HF ``AutoTokenizer``: the Rust-based tokenizers
library has known fork() deadlocks when used inside DataLoader workers.  The
ESM2 vocab is fixed (33 tokens, no merges) so a Python dict lookup is the
simplest correct implementation.
"""

from typing import Any, Dict, List

import torch


# ESM2 vocabulary (33 tokens).  Order MUST match
# ``transformers.EsmTokenizer.vocab`` for facebook/esm2_*; fair-esm names the
# slot at id=31 ``<null_1>`` and the same convention is used in the HF port.
_ESM2_VOCAB: List[str] = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-",
    "<null_1>",
    "<mask>",
]

_TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(_ESM2_VOCAB)}
_CLS_ID = _TOKEN_TO_ID["<cls>"]   # 0
_PAD_ID = _TOKEN_TO_ID["<pad>"]   # 1
_EOS_ID = _TOKEN_TO_ID["<eos>"]   # 2
_UNK_ID = _TOKEN_TO_ID["<unk>"]   # 3


class Esm2Tokenizer:
    """Pure-Python char-level tokenizer for ESM2.

    Functionally equivalent to ``transformers.EsmTokenizer`` for protein
    sequences (no BPE merges), but avoids the Rust ``tokenizers`` dependency
    so it is safe inside forked DataLoader workers.
    """

    def __init__(self):
        self.pad_token_id = _PAD_ID

    def _encode_one(self, seq: str) -> List[int]:
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
        all_ids: List[List[int]] = []
        for seq in sequences:
            tokens = self._encode_one(seq)
            if truncation and len(tokens) > max_length:
                tokens = tokens[: max_length - 1] + [_EOS_ID]
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
