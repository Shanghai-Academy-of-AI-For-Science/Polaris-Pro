"""Input-building helpers for the inference CLI.

  * ``RNACharTokenizer`` + its char<->id maps and ``RNA_NUM_LATENT_TOKENS``
  * ``extract_epi_cell_line`` (EPI task-tag → cell-line suffix)
  * the ``ea_*`` enhancer-activity binned-label family + ``rewrite_ea_binned_instruction``
"""

import os
import re
import bisect
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any

import torch

logger = logging.getLogger(__name__)

# RNA encoder (RNAConvFormer) compresses a variable-length sequence into this
# many latent tokens fed to the LLM.
RNA_NUM_LATENT_TOKENS = 16


# ---------------------------------------------------------------------------
# Enhancer-activity (EA) binned-label helpers
# ---------------------------------------------------------------------------
# Some DNA regression tasks (enhancer activity) can be posed either as a raw
# float or as an N-way bucket id. Binned mode maps the float to a stable bucket
# id so training and eval share exactly the same binning. Edges + per-bin
# centers come from a JSON file in ``qwenvl/data/assets/ea_bin_edges_<N>.json``.

_EA_BIN_CACHE: Dict[int, Dict[str, Any]] = {}

# Per-sample EA mode marker: lets float and binned EA samples coexist in one
# batch without a global env switch.
EA_LABEL_MODE_KEY = "_ea_label_mode"


def ea_label_mode() -> str:
    """Read the EA_LABEL_MODE env once; default ``"float"`` keeps legacy
    2-decimal behaviour."""
    return os.environ.get("EA_LABEL_MODE", "float").strip()


def ea_n_bins(mode: Optional[str] = None) -> Optional[int]:
    """Parse ``binned_<N>`` and return N; ``None`` for any non-binned mode."""
    if mode is None:
        mode = ea_label_mode()
    if not mode.startswith("binned_"):
        return None
    try:
        return int(mode.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def ea_n_bins_for_sample(item: Dict[str, Any]) -> Optional[int]:
    """Per-sample EA bin count.

    Resolution order:
      1. ``item[EA_LABEL_MODE_KEY]`` if the loader stamped one (e.g. from
         ``dna_EA_bin100`` registry entry).
      2. ``$EA_LABEL_MODE`` env var — legacy fallback, applies globally so
         older configs that only set the env still work.
    Returns ``None`` for "float" mode or anything we can't parse.
    """
    mode = item.get(EA_LABEL_MODE_KEY)
    if not mode:
        mode = ea_label_mode()
    return ea_n_bins(mode)


def _ea_default_edges_path(n_bins: int) -> str:
    # Packaged next to this file: qwenvl/data/assets/ea_bin_edges_<N>.json
    here = Path(__file__).resolve().parent
    return str(here / "assets" / f"ea_bin_edges_{n_bins}.json")


def ea_load_bin_table(n_bins: int) -> Dict[str, Any]:
    """Load (and cache) the bin-edges JSON for ``n_bins``.

    Lookup order:
      1. ``$EA_BIN_EDGES_JSON`` (full path, takes precedence)
      2. ``qwenvl/data/assets/ea_bin_edges_<N>.json`` (packaged)
    """
    if n_bins in _EA_BIN_CACHE:
        return _EA_BIN_CACHE[n_bins]
    path = os.environ.get("EA_BIN_EDGES_JSON") or _ea_default_edges_path(n_bins)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("n_bins", 0)) != n_bins:
        raise ValueError(
            f"EA bin-edges file {path} has n_bins={payload.get('n_bins')} "
            f"but EA_LABEL_MODE requested {n_bins}"
        )
    _EA_BIN_CACHE[n_bins] = payload
    return payload


def ea_value_to_bin(val: float, n_bins: int) -> int:
    """Map a raw EA float label to its bin id ``[0, n_bins-1]``."""
    edges = ea_load_bin_table(n_bins)["edges"]  # length n_bins + 1
    interior = edges[1:-1]                      # length n_bins - 1
    idx = bisect.bisect_right(interior, val)
    if idx < 0:
        idx = 0
    elif idx >= n_bins:
        idx = n_bins - 1
    return idx


def ea_format_bin(bin_id: int, n_bins: int) -> str:
    """Zero-padded bucket id, width = digit count of (n_bins - 1)."""
    width = max(2, len(str(n_bins - 1)))
    return f"{bin_id:0{width}d}"


def ea_binned_output_instruction(n_bins: int) -> str:
    """User-facing EA bucket instruction shared by training and inference."""
    top = n_bins - 1
    width = max(2, len(str(top)))
    return (
        f"Output a {width}-digit bucket id from "
        f"{0:0{width}d} (lowest activity) to {top:0{width}d} "
        f"(highest activity). Higher numbers indicate higher activity."
    )


def ea_binned_system_prompt(n_bins: int) -> str:
    """System prompt for EA binned mode.

    The source EA data asks for a floating-point value. In binned mode the
    assistant target is a bucket id, so the system instruction must be
    rewritten too; otherwise system/user/target disagree.
    """
    top = n_bins - 1
    width = max(2, len(str(top)))
    return (
        "You are a DNA sequence analysis expert. Read the DNA sequence and "
        "the question carefully. "
        f"Respond with a single {width}-digit bucket id from "
        f"{0:0{width}d} (lowest activity) to {top:0{width}d} "
        "(highest activity) only. Higher bucket ids indicate higher enhancer "
        "activity. Do not add units, explanations, reasoning, punctuation, "
        "or any additional text."
    )


def rewrite_ea_binned_instruction(text: str, n_bins: int, role: str) -> str:
    """Rewrite EA float instructions into bucket-id instructions."""
    if role == "system":
        return ea_binned_system_prompt(n_bins)
    if role != "user":
        return text

    replacement = ea_binned_output_instruction(n_bins)
    rewritten = re.sub(
        r"Answer\s+with\s+(?:a\s+)?(?:single\s+)?"
        r"(?:floating-point|float)\s+number\.?",
        replacement,
        text,
        flags=re.IGNORECASE,
    )
    if rewritten == text and replacement not in text:
        rewritten = f"{text.rstrip()} {replacement}"
    return rewritten


def ea_bin_to_value(bin_id: int, n_bins: int) -> float:
    """Reverse lookup used at eval time: bin id -> representative float."""
    table = ea_load_bin_table(n_bins)
    centers = table["centers"]
    if not 0 <= bin_id < len(centers):
        raise ValueError(f"bin_id {bin_id} out of range [0, {len(centers)})")
    return float(centers[bin_id])


def extract_epi_cell_line(task: Optional[str]) -> Optional[str]:
    """Return the cell-line suffix from an EPI task tag, if present."""
    if not task:
        return None
    prefix = "promoter_enhancer_interaction-"
    if task.startswith(prefix):
        cell = task[len(prefix):].strip()
        return cell or None
    return None


# ---------------------------------------------------------------------------
# Built-in character-level RNA tokenizer (RNAConvFormer, vocab_size=8)
# ---------------------------------------------------------------------------
_RNA_CHAR_TO_ID = {"<pad>": 0, "<cls>": 1, "A": 2, "U": 3, "G": 4, "C": 5, "N": 6, "<sep>": 7}
_RNA_ID_TO_CHAR = {v: k for k, v in _RNA_CHAR_TO_ID.items()}


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
    ) -> Dict[str, torch.Tensor]:
        all_ids: List[List[int]] = []
        for seq in sequences:
            ids = [self.cls_token_id]
            for ch in seq.upper():
                if ch == "T":
                    ch = "U"
                ids.append(_RNA_CHAR_TO_ID.get(ch, _RNA_CHAR_TO_ID["N"]))
            ids.append(self.sep_token_id)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length - 1] + [self.sep_token_id]
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
