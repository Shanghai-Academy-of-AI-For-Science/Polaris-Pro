"""
SpecialTokenManager: manages per-modality special tokens.

Handles adding new tokens to the tokenizer, resizing model embeddings,
and smart-initializing new token embeddings from existing semantically
similar tokens (e.g. ``<|rna_start|>`` from ``<|vision_start|>``).

Design principle: each modality MUST have its own start/end/pad tokens.
The model uses the pad token ID to locate where to scatter modality
embeddings in the forward pass.  Reusing vision tokens for non-vision
modalities is no longer supported by default.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .modality_spec import ModalitySpec

logger = logging.getLogger(__name__)

# ── Well-known Qwen3-VL built-in token IDs ──────────────────────────
QWEN_VISION_START_ID = 151652  # <|vision_start|>
QWEN_VISION_END_ID = 151653    # <|vision_end|>
QWEN_IMAGE_PAD_ID = 151655     # <|image_pad|>
QWEN_VIDEO_PAD_ID = 151656     # <|video_pad|>

# ── Canonical modality token strings ────────────────────────────────
# These are the default token names for each modality.  All new modalities
# should follow this naming convention: <|{name}_start|>, <|{name}_end|>,
# <|{name}_pad|>.
MODALITY_TOKEN_DEFS = {
    "rna": {
        "start": "<|rna_start|>",
        "end": "<|rna_end|>",
        "pad": "<|rna_pad|>",
    },
    "dna": {
        "start": "<|dna_start|>",
        "end": "<|dna_end|>",
        "pad": "<|dna_pad|>",
    },
    "protein": {
        "start": "<|protein_start|>",
        "end": "<|protein_end|>",
        "pad": "<|protein_pad|>",
    },
    "mol": {
        "start": "<|mol_start|>",
        "end": "<|mol_end|>",
        "pad": "<|mol_pad|>",
    },
    # Runtime contact_map data support has been removed, but these tokens stay
    # in the canonical layout as dormant reserved slots. Keeping them in the
    # default registration order prevents later modalities (weather and
    # bio_seq_pad) from shifting IDs.
    "contact_map": {
        "start": "<|contact_start|>",
        "end": "<|contact_end|>",
        "pad": "<|contact_pad|>",
    },
    "weather": {
        "start": "<|weather_start|>",
        "end": "<|weather_end|>",
        "pad": "<|weather_pad|>",
    },
}

# Canonical shared layout. Do not filter this by active modalities: even
# single-modality runs must reserve earlier dormant slots so token IDs stay
# identical across RNA/DNA/protein/mol/weather checkpoints.
DEFAULT_REGISTERED_MODALITIES: List[str] = [
    "rna",
    "dna",
    "protein",
    "mol",
    "contact_map",
    "weather",
]

# Placeholder for bio sequence OUTPUT tokens (RNA/DNA generation).
# Used ONLY as a temporary placeholder during tokenization — gets replaced
# with actual nucleotide token IDs before training.  Kept separate from
# the modality input pad tokens so that the rope function / model forward
# never confuse output placeholders with encoder input positions.
BIO_SEQ_OUTPUT_PAD = "<|bio_seq_pad|>"

CANONICAL_TOKEN_IDS = {
    "<|rna_start|>": 151669,
    "<|rna_end|>": 151670,
    "<|rna_pad|>": 151671,
    "<|dna_start|>": 151672,
    "<|dna_end|>": 151673,
    "<|dna_pad|>": 151674,
    "<|protein_start|>": 151675,
    "<|protein_end|>": 151676,
    "<|protein_pad|>": 151677,
    "<|mol_start|>": 151678,
    "<|mol_end|>": 151679,
    "<|mol_pad|>": 151680,
    "<|contact_start|>": 151681,
    "<|contact_end|>": 151682,
    "<|contact_pad|>": 151683,
    "<|weather_start|>": 151684,
    "<|weather_end|>": 151685,
    "<|weather_pad|>": 151686,
    BIO_SEQ_OUTPUT_PAD: 151687,
}


def _collect_package_token_defs() -> Dict[str, Dict[str, str]]:
    """Collect TOKEN_DEFS from all modality packages under qwenvl.modalities.*."""
    import importlib
    import pkgutil
    collected: Dict[str, Dict[str, str]] = {}
    try:
        import qwenvl.modalities as _root
        for _imp, pkg_name, is_pkg in pkgutil.iter_modules(_root.__path__):
            if not is_pkg:
                continue
            try:
                mod = importlib.import_module(f"qwenvl.modalities.{pkg_name}")
            except Exception:
                continue
            pkg_defs = getattr(mod, "TOKEN_DEFS", None)
            if isinstance(pkg_defs, dict):
                collected.update(pkg_defs)
    except Exception:
        pass
    return collected


def get_modality_tokens(name: str) -> Dict[str, str]:
    """Return the canonical token strings for a modality name.

    Checks the hardcoded MODALITY_TOKEN_DEFS first, then falls back to
    TOKEN_DEFS exported by modality packages, and finally generates
    default names from the modality name.
    """
    if name in MODALITY_TOKEN_DEFS:
        return dict(MODALITY_TOKEN_DEFS[name])
    # Try package-level TOKEN_DEFS (lazy, not cached — called rarely)
    pkg_defs = _collect_package_token_defs()
    if name in pkg_defs:
        return dict(pkg_defs[name])
    return {
        "start": f"<|{name}_start|>",
        "end": f"<|{name}_end|>",
        "pad": f"<|{name}_pad|>",
    }


def get_all_bio_special_tokens() -> List[str]:
    """Return a flat list of all bio-modality special tokens."""
    tokens = []
    for name in MODALITY_TOKEN_DEFS:
        defs = get_modality_tokens(name)
        tokens.extend(defs.values())
    tokens.append(BIO_SEQ_OUTPUT_PAD)
    return tokens


def get_all_known_modality_names() -> List[str]:
    """Return all modality names known to this codebase.

    Combines the canonical names in :data:`MODALITY_TOKEN_DEFS` with names
    declared by per-modality packages under ``qwenvl.modalities.*``.  Used to
    keep the tokenizer / embedding shape *identical* across runs regardless of
    which modalities are actually enabled — so that any bio_qwen3vl checkpoint
    is cross-loadable with any other.
    """
    seen = []
    for name in MODALITY_TOKEN_DEFS.keys():
        if name not in seen:
            seen.append(name)
    for name in _collect_package_token_defs().keys():
        if name not in seen:
            seen.append(name)
    return seen


class SpecialTokenManager:
    """Manages registration and initialization of modality special tokens.

    After calling ``register_all_modality_tokens`` or individual
    ``register_modality_tokens``, the new tokens are:
    1. Added to the tokenizer.
    2. Model embeddings resized.
    3. New embeddings warm-started from existing vision tokens.
    """

    def __init__(self, tokenizer, model: nn.Module):
        self.tokenizer = tokenizer
        self.model = model
        self._registered_tokens: Dict[str, int] = {}
        # After registration, maps modality_name -> {start: id, end: id, pad: id}
        self.modality_token_ids: Dict[str, Dict[str, int]] = {}

    def _assert_fits_unused_slots(self, tokenizer_vocab_size: int) -> int:
        """Verify new tokens fit into Qwen3-VL's existing padded unused slots.

        Qwen3-VL ships with ``embed_tokens`` / ``lm_head`` padded to a multiple
        of 128 — for the 8B model that means 151936 rows while
        ``len(tokenizer)`` is 151669, leaving 267 unused slots.  Adding bio
        tokens up to that limit lands them on rows the base model has never
        looked up (no forward, no gradient), so registration is a pure no-op
        on the original Qwen-VL behaviour.

        Resizing the embedding to grow past the original padded size is
        explicitly disallowed: it would change the lm_head softmax
        denominator and the matrix alignment expected by the base ckpt's
        bf16 paths.  If you ever need more bio tokens than fit,
        rethink the modality lineup — don't enable resizing.
        """
        model_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        if tokenizer_vocab_size > model_vocab_size:
            raise RuntimeError(
                f"Bio-token registration would require resizing the LLM "
                f"embedding from {model_vocab_size} to {tokenizer_vocab_size} "
                f"(+{tokenizer_vocab_size - model_vocab_size} rows). Resizing "
                f"is disabled to keep Qwen3-VL's original lm_head intact. "
                f"The canonical shared bio-token layout reserves IDs "
                f"151669-151687, so use base Qwen3-VL or a checkpoint whose "
                f"embedding/lm_head already covers that layout."
            )
        return model_vocab_size

    def _assert_canonical_ids(self, tokens: List[str]) -> None:
        mismatches = []
        for tok in tokens:
            expected = CANONICAL_TOKEN_IDS.get(tok)
            if expected is None:
                continue
            actual = self.tokenizer.convert_tokens_to_ids(tok)
            if actual != expected:
                mismatches.append(f"{tok}: expected {expected}, got {actual}")
        if mismatches:
            raise RuntimeError(
                "Bio special-token IDs do not match the canonical reserved "
                "layout. Start from base Qwen3-VL or a checkpoint trained "
                "with the shared 151669-151687 layout; do not continue from "
                "an old single-modality tokenizer. Mismatches: "
                + "; ".join(mismatches)
            )

    # ------------------------------------------------------------------
    # Bulk registration
    # ------------------------------------------------------------------

    def register_all_modality_tokens(
        self,
        modality_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Register tokens for multiple modalities at once.

        Args:
            modality_names: List of modality names to register.
                If None, registers the full canonical reserved layout in
                DEFAULT_REGISTERED_MODALITIES. Do not pass only active
                modalities when starting from base, because that would shift
                token IDs.

        Returns:
            Dict mapping modality_name -> {start: id, end: id, pad: id}.
        """
        if modality_names is None:
            modality_names = list(DEFAULT_REGISTERED_MODALITIES)

        # Collect all new tokens
        all_new_tokens = []
        requested_tokens = []
        for name in modality_names:
            defs = get_modality_tokens(name)
            for tok in defs.values():
                requested_tokens.append(tok)
                if tok not in self.tokenizer.get_vocab() and tok not in all_new_tokens:
                    all_new_tokens.append(tok)
        # Always include the output sequence placeholder
        requested_tokens.append(BIO_SEQ_OUTPUT_PAD)
        if BIO_SEQ_OUTPUT_PAD not in self.tokenizer.get_vocab() and BIO_SEQ_OUTPUT_PAD not in all_new_tokens:
            all_new_tokens.append(BIO_SEQ_OUTPUT_PAD)

        # Add them in one batch (efficient – only one resize)
        if all_new_tokens:
            existing = list(
                self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
            )
            combined = existing + [t for t in all_new_tokens if t not in existing]
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": combined}
            )
            if num_added > 0:
                tokenizer_vocab_size = len(self.tokenizer)
                model_vocab_size = self._assert_fits_unused_slots(tokenizer_vocab_size)
                logger.info(
                    f"[TokenManager] Added {num_added} bio-modality tokens "
                    f"into Qwen3-VL's unused slots, tokenizer size now "
                    f"{tokenizer_vocab_size}, model vocab size unchanged at "
                    f"{model_vocab_size}"
                )
                self._smart_init_batch(all_new_tokens)

        self._assert_canonical_ids(requested_tokens)

        # Build token-id maps
        result = {}
        for name in modality_names:
            defs = get_modality_tokens(name)
            ids = {}
            for role, tok in defs.items():
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                ids[role] = tid
                self._registered_tokens[tok] = tid
            self.modality_token_ids[name] = ids
            result[name] = ids

        # Store the output placeholder token ID
        bio_output_pad_id = self.tokenizer.convert_tokens_to_ids(BIO_SEQ_OUTPUT_PAD)
        self._registered_tokens[BIO_SEQ_OUTPUT_PAD] = bio_output_pad_id
        result["_bio_seq_output_pad"] = {"pad": bio_output_pad_id}

        logger.info(f"[TokenManager] Registered modalities: {[k for k in result if not k.startswith('_')]}")
        for name, ids in result.items():
            if name.startswith("_"):
                logger.info(f"  {name}: pad={ids['pad']}")
            else:
                core = ", ".join(
                    f"{k}={ids[k]}" for k in ("start", "end", "pad") if k in ids
                )
                logger.info(f"  {name}: {core}")

        return result

    # ------------------------------------------------------------------
    # Single-modality registration
    # ------------------------------------------------------------------

    def register_modality_tokens(
        self,
        spec: ModalitySpec,
    ) -> Dict[str, int]:
        """Add start/end/pad tokens for a single modality.

        Returns:
            Dict mapping role (start/end/pad) -> token ID.
        """
        tokens_to_add = []
        for tok in (spec.start_token, spec.end_token, spec.pad_token):
            if tok and tok not in self.tokenizer.get_vocab():
                tokens_to_add.append(tok)

        if tokens_to_add:
            existing = list(
                self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
            )
            combined = existing + [t for t in tokens_to_add if t not in existing]
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": combined}
            )
            if num_added > 0:
                tokenizer_vocab_size = len(self.tokenizer)
                model_vocab_size = self._assert_fits_unused_slots(tokenizer_vocab_size)
                logger.info(
                    f"[TokenManager] Added {num_added} tokens for '{spec.name}' "
                    f"into Qwen3-VL's unused slots, tokenizer size now "
                    f"{tokenizer_vocab_size}, model vocab size unchanged at "
                    f"{model_vocab_size}"
                )
                self._smart_init_batch(tokens_to_add)

        token_map = {}
        for role, tok in [("start", spec.start_token), ("end", spec.end_token), ("pad", spec.pad_token)]:
            if tok:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                token_map[role] = tid
                self._registered_tokens[tok] = tid

        self.modality_token_ids[spec.name] = token_map
        return token_map

    # ------------------------------------------------------------------
    # Smart initialization
    # ------------------------------------------------------------------

    def _smart_init_batch(self, new_tokens: List[str]):
        """Initialize new token embeddings from semantically similar existing tokens.

        Strategy (warm start for faster learning):
        - *_start tokens: copy from <|vision_start|> embedding
        - *_end tokens:   copy from <|vision_end|> embedding
        - *_pad tokens:   copy from <|image_pad|> embedding

        This gives the model a significant head start because:
        1. The attention patterns for "start/end a modality segment" are
           structurally identical to the vision case.
        2. The pad tokens serve the same role (placeholder for encoder
           features) as <|image_pad|>.
        3. This is much better than random init because the model already
           "knows" how to handle vision_start/vision_end boundaries.

        Additionally, we also initialize the lm_head (output embeddings)
        for the new tokens in the same way so that the model can both
        read and predict them from the start.
        """
        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None:
            return

        weight = input_embeddings.weight.data

        # Reference tokens for initialization
        reference_map = {
            "start": QWEN_VISION_START_ID,
            "end": QWEN_VISION_END_ID,
            "pad": QWEN_IMAGE_PAD_ID,
        }

        for tok in new_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid >= weight.shape[0]:
                continue

            # Determine which reference to use
            tok_lower = tok.lower()
            ref_id = None
            if "start" in tok_lower:
                ref_id = reference_map["start"]
            elif "end" in tok_lower:
                ref_id = reference_map["end"]
            elif "pad" in tok_lower:
                ref_id = reference_map["pad"]

            if ref_id is not None and ref_id < weight.shape[0]:
                weight[tid] = weight[ref_id].clone()
                logger.info(
                    f"[TokenManager] input_embed: '{tok}' (id={tid}) "
                    f"<- reference id={ref_id}"
                )
            else:
                weight[tid] = weight[:min(1000, weight.shape[0])].mean(dim=0)
                logger.info(
                    f"[TokenManager] input_embed: '{tok}' (id={tid}) <- embedding mean"
                )

        # Also initialize lm_head (output embeddings) if it is NOT tied
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings is not input_embeddings:
            out_weight = output_embeddings.weight.data
            for tok in new_tokens:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid >= out_weight.shape[0]:
                    continue
                tok_lower = tok.lower()
                ref_id = None
                if "start" in tok_lower:
                    ref_id = reference_map["start"]
                elif "end" in tok_lower:
                    ref_id = reference_map["end"]
                elif "pad" in tok_lower:
                    ref_id = reference_map["pad"]
                if ref_id is not None and ref_id < out_weight.shape[0]:
                    out_weight[tid] = out_weight[ref_id].clone()

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_token_id(self, token: str) -> Optional[int]:
        """Look up a registered token's ID."""
        if token in self._registered_tokens:
            return self._registered_tokens[token]
        vocab = self.tokenizer.get_vocab()
        return vocab.get(token)

    def get_pad_token_id(self, modality_name: str) -> Optional[int]:
        """Return the pad token ID for a given modality."""
        ids = self.modality_token_ids.get(modality_name)
        if ids:
            return ids.get("pad")
        return None

    def get_modality_token_ids(self, name_or_spec) -> Dict[str, Optional[int]]:
        """Return {start, end, pad} token IDs for a modality."""
        if isinstance(name_or_spec, str):
            return self.modality_token_ids.get(name_or_spec, {})
        spec = name_or_spec
        return self.modality_token_ids.get(spec.name, {})

    def get_all_pad_token_ids(self) -> Dict[str, int]:
        """Return {modality_name: pad_token_id} for all registered modalities."""
        return {
            name: ids["pad"]
            for name, ids in self.modality_token_ids.items()
            if "pad" in ids
        }
