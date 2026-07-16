#!/usr/bin/env python
"""
Unified inference script for Monkey King Bang (MKB).

Supported tasks (matching training format):
  - classification   : RNA/DNA/mol input  -> text label (0/1)
  - regression        : RNA+DNA/mol input -> score
  - description       : RNA input     -> free-form text
  - generation        : RNA input(s)  -> RNA output sequence
  - mol_property_*    : mol input     -> property prediction

Input JSON(L) format is identical to training:
{
    "rna": ["SEQ1", ...],            # optional
    "dna": ["SEQ1", ...],            # optional
    "mol": ["SMILES1", ...],         # optional
    "image": ["path.jpg"],           # optional
    "conversations": [
        {"from": "human", "value": "<mol>\n...prompt..."},
        {"from": "gpt",  "value": ""}   # leave empty for inference
    ],
    "task": "mol_property_classification"  # optional, for logging
}

Usage:
  # Single sample (inline)
  python inference.py --model_path /path/to/ckpt \
      --rna "AUGCAUGC" \
      --prompt "<rna>\nWhat family does this RNA belong to?"

  # Batch from file
  python inference.py --model_path /path/to/ckpt \
      --input_file samples.jsonl \
      --output_file results.jsonl

  # Interactive chat
  python inference.py --model_path /path/to/ckpt --chat
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
from PIL import Image

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
except ImportError:  # transformers compatibility shim
    from transformers import LogitsProcessor, LogitsProcessorList
from mkb.models import Qwen3VLForConditionalGeneration
from mkb.processor_compat import load_auto_processor_compat
from mkb.data.inference_helpers import (
    RNACharTokenizer,
    RNA_NUM_LATENT_TOKENS,
    _RNA_CHAR_TO_ID,
    _RNA_ID_TO_CHAR,
    extract_epi_cell_line,
    ea_n_bins_for_sample,
    rewrite_ea_binned_instruction,
)
from mkb.registry.token_manager import BIO_SEQ_OUTPUT_PAD, MODALITY_TOKEN_DEFS

_RNA_TOKENS = MODALITY_TOKEN_DEFS["rna"]
_DNA_TOKENS = MODALITY_TOKEN_DEFS["dna"]
_PROTEIN_TOKENS = MODALITY_TOKEN_DEFS["protein"]
_MOL_TOKENS = MODALITY_TOKEN_DEFS["mol"]


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Apply vLLM/OpenAI-style presence penalty to generated tokens only."""

    def __init__(self, prompt_length: int, penalty: float):
        self.prompt_length = int(prompt_length)
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty <= 0 or input_ids.shape[1] <= self.prompt_length:
            return scores
        generated = input_ids[:, self.prompt_length:]
        for batch_i in range(generated.shape[0]):
            used = torch.unique(generated[batch_i])
            if used.numel() > 0:
                scores[batch_i, used] = scores[batch_i, used] - self.penalty
        return scores


def decode_mol_token_ids(token_ids: torch.Tensor) -> str:
    """Decode a 1-D / 2-D tensor of generated mol-vocab IDs into a SMILES string.

    Mol generation runs in the decoder's own SMILES vocab (see
    :mod:`mkb.modalities.mol.tokenizer`); each id maps directly to a
    SMILES token via ``MOL_ID_TO_TOKEN``.  ``<pad>`` / ``<cls>`` / ``<sep>``
    are dropped, and decoding stops at the first ``<sep>``.
    """
    from mkb.modalities.mol.tokenizer import MOL_ID_TO_TOKEN, MOL_SEP_ID

    if token_ids.dim() == 2:
        token_ids = token_ids[0]
    skip_tokens = {"<pad>", "<cls>"}
    out: List[str] = []
    for tid in token_ids.tolist():
        tid = int(tid)
        if tid == MOL_SEP_ID:
            break
        if 0 <= tid < len(MOL_ID_TO_TOKEN):
            tok = MOL_ID_TO_TOKEN[tid]
            if tok not in skip_tokens:
                out.append(tok)
    return "".join(out)


def _checkpoint_key_names(model_path: str) -> Optional[set]:
    """Return checkpoint tensor keys when they can be inspected cheaply."""
    root = Path(model_path)
    if not root.exists() or not root.is_dir():
        return None

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / index_name
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                weight_map = data.get("weight_map") or {}
                return set(weight_map.keys())
            except Exception:
                return None

    safetensors_files = sorted(root.glob("*.safetensors"))
    if safetensors_files:
        try:
            from safetensors import safe_open
        except Exception:
            return None
        keys = set()
        try:
            for path in safetensors_files:
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    keys.update(f.keys())
            return keys
        except Exception:
            return None

    return None


_MOL_DECODER_REQUIRED_SUFFIXES = (
    "tok_embed.weight",
    "kv_proj.0.weight",
    "layers.0.self_q.weight",
    "head.weight",
)
_MOL_DECODER_VOCAB_SUFFIXES = (
    "tok_embed.weight",
    "head.weight",
)


def _current_mol_vocab_size() -> int:
    from mkb.modalities.mol.tokenizer import MOL_VOCAB_SIZE
    return int(MOL_VOCAB_SIZE)


def _checkpoint_mol_config_vocab(model_path: str) -> Optional[int]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mol_config = data.get("mol_config")
    if not isinstance(mol_config, dict):
        return None
    value = mol_config.get("mol_vocab_size")
    return int(value) if value is not None else None


def _suffix_for_mol_decoder_key(key: str) -> Optional[str]:
    for suffix in _MOL_DECODER_REQUIRED_SUFFIXES:
        if key.endswith(f"decoders.mol.{suffix}"):
            return suffix
    return None


def _checkpoint_mol_tensor_shapes(
    model_path: str,
) -> Optional[Dict[str, Tuple[str, Tuple[int, ...]]]]:
    """Return shapes for required mol decoder tensors, or None if unreadable."""
    root = Path(model_path)
    if not root.exists() or not root.is_dir():
        return None

    def _record(out, key, shape):
        suffix = _suffix_for_mol_decoder_key(key)
        if suffix is not None:
            out[suffix] = (key, tuple(int(x) for x in shape))

    try:
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            from safetensors import safe_open
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = (json.load(f).get("weight_map") or {})
            out = {}
            wanted_by_shard = {}
            for key, shard in weight_map.items():
                if _suffix_for_mol_decoder_key(key) is not None:
                    wanted_by_shard.setdefault(shard, []).append(key)
            for shard, keys in wanted_by_shard.items():
                with safe_open(str(root / shard), framework="pt", device="cpu") as f:
                    for key in keys:
                        _record(out, key, f.get_tensor(key).shape)
            return out

        bin_index_path = root / "pytorch_model.bin.index.json"
        if bin_index_path.exists():
            with open(bin_index_path, "r", encoding="utf-8") as f:
                weight_map = (json.load(f).get("weight_map") or {})
            out = {}
            wanted_by_shard = {}
            for key, shard in weight_map.items():
                if _suffix_for_mol_decoder_key(key) is not None:
                    wanted_by_shard.setdefault(shard, []).append(key)
            for shard, keys in wanted_by_shard.items():
                state = torch.load(str(root / shard), map_location="cpu")
                for key in keys:
                    if key in state:
                        _record(out, key, state[key].shape)
            return out

        safetensors_files = sorted(root.glob("*.safetensors"))
        if safetensors_files:
            from safetensors import safe_open
            out = {}
            for path in safetensors_files:
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if _suffix_for_mol_decoder_key(key) is not None:
                            _record(out, key, f.get_tensor(key).shape)
            return out

        bin_files = sorted(root.glob("pytorch_model*.bin"))
        if bin_files:
            out = {}
            for path in bin_files:
                state = torch.load(str(path), map_location="cpu")
                for key, value in state.items():
                    if _suffix_for_mol_decoder_key(key) is not None:
                        _record(out, key, value.shape)
            return out
    except Exception:
        return None

    return None


def preflight_mol_checkpoint(
    model_path: str,
    fail_on_legacy_mol_decoder: bool = True,
) -> bool:
    """Strictly validate that a checkpoint uses the current mol vocab."""
    expected_vocab = _current_mol_vocab_size()

    config_vocab = _checkpoint_mol_config_vocab(model_path)
    if config_vocab != expected_vocab:
        raise RuntimeError(
            "[mol-preflight] checkpoint mol_config.mol_vocab_size must match "
            f"current tokenizer vocab. checkpoint={config_vocab!r}, "
            f"expected={expected_vocab}. Re-train mol with the current tokenizer."
        )

    keys = _checkpoint_key_names(model_path)
    if keys is None:
        raise RuntimeError(
            "[mol-preflight] could not inspect checkpoint tensor keys; refusing "
            "to load mol checkpoint because decoder vocab cannot be verified."
        )

    missing = [
        suffix for suffix in _MOL_DECODER_REQUIRED_SUFFIXES
        if not any(k.endswith(f"decoders.mol.{suffix}") for k in keys)
    ]
    if missing:
        has_any_mol_decoder = any("decoders.mol." in k for k in keys)
        raise RuntimeError(
            "[mol-preflight] checkpoint does not contain the required "
            f"MolARDecoder layout (missing: {', '.join(missing)}; "
            f"any_mol_decoder_keys={has_any_mol_decoder}). Re-train mol with "
            "the current code instead of loading a legacy/random decoder."
        )

    shapes = _checkpoint_mol_tensor_shapes(model_path)
    if shapes is None:
        raise RuntimeError(
            "[mol-preflight] could not inspect mol decoder tensor shapes; refusing "
            "to load mol checkpoint because decoder vocab cannot be verified."
        )
    missing_shapes = [s for s in _MOL_DECODER_VOCAB_SUFFIXES if s not in shapes]
    if missing_shapes:
        raise RuntimeError(
            "[mol-preflight] could not find mol decoder vocab tensor shapes for "
            f"{missing_shapes}; refusing to load this checkpoint."
        )
    bad_shapes = []
    for suffix in _MOL_DECODER_VOCAB_SUFFIXES:
        key, shape = shapes[suffix]
        if not shape or int(shape[0]) != expected_vocab:
            bad_shapes.append((key, shape))
    if bad_shapes:
        detail = "; ".join(f"{key}: shape={shape}" for key, shape in bad_shapes)
        raise RuntimeError(
            "[mol-preflight] mol decoder tensor vocab dimension must match "
            f"current tokenizer vocab={expected_vocab}. {detail}. Re-train mol "
            "with the current tokenizer."
        )

    print(
        "[mol-preflight] MolARDecoder vocab validation passed "
        f"(vocab={expected_vocab})."
    )
    return True


# ---------------------------------------------------------------------------
# Helpers – mirror the training-time _build_messages logic
# ---------------------------------------------------------------------------

def _rna_input_placeholder(num_latent_tokens: int, placeholder_tag: str = "<rna>") -> str:
    """Placeholder for an INPUT RNA/DNA sequence (encoded by RNAConvFormer)."""
    tok = _RNA_TOKENS if placeholder_tag == "<rna>" else _DNA_TOKENS
    return tok["start"] + tok["pad"] * num_latent_tokens + tok["end"]


def _protein_input_placeholder(num_latent_tokens: int) -> str:
    """Placeholder for an INPUT protein sequence (encoded by ESM3 + resampler)."""
    return _PROTEIN_TOKENS["start"] + _PROTEIN_TOKENS["pad"] * num_latent_tokens + _PROTEIN_TOKENS["end"]


def _mol_input_placeholder(num_latent_tokens: int) -> str:
    """Placeholder for an INPUT molecular SMILES (encoded by GNN + resampler)."""
    return _MOL_TOKENS["start"] + _MOL_TOKENS["pad"] * num_latent_tokens + _MOL_TOKENS["end"]


def build_inference_inputs(
    item: Dict[str, Any],
    processor,
    rna_tokenizer: Optional[RNACharTokenizer],
    protein_tokenizer=None,
    dna_tokenizer=None,
    num_latent_tokens: int = RNA_NUM_LATENT_TOKENS,
    num_dna_latent_tokens: int = RNA_NUM_LATENT_TOKENS,
    num_protein_latent_tokens: int = 32,
    num_mol_latent_tokens: int = 16,
    base_path: Path = Path(""),
    max_bio_seq_length: int = 4096,
    protein_max_residues: Optional[int] = None,
    rna_max_residues: Optional[int] = None,
    dna_max_residues: Optional[int] = None,
    mol_prompt_style: str = "prompt_only",
    mol_generation_slots: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Convert a single sample dict (training format) into model-ready tensors.

    This mirrors preprocess_qwen_visual / _build_messages but for inference
    (no labels, only user turns matter, gpt turn is left for generation).
    """
    if mol_prompt_style not in ("prompt_only", "train_slots"):
        raise ValueError(
            "mol_prompt_style must be one of {'prompt_only', 'train_slots'}, "
            f"got {mol_prompt_style!r}"
        )
    mol_generation_slots = max(1, int(mol_generation_slots or 1))

    # ---- extract media pools ----
    rnas = item.get("rna") or []
    if isinstance(rnas, str):
        rnas = [rnas]
    dnas = item.get("dna") or []
    if isinstance(dnas, str):
        dnas = [dnas]
    proteins = item.get("protein") or []
    if isinstance(proteins, str):
        proteins = [proteins]
    rna_cap = rna_max_residues if rna_max_residues is not None else max_bio_seq_length
    dna_cap = dna_max_residues if dna_max_residues is not None else max_bio_seq_length
    protein_cap = protein_max_residues if protein_max_residues is not None else max_bio_seq_length
    rnas = [s[: rna_cap - 2] for s in rnas]
    dnas = [s[: dna_cap - 2] for s in dnas]
    proteins = [s[: protein_cap - 2] for s in proteins]

    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    mols = item.get("mol") or []
    if isinstance(mols, str):
        mols = [mols]

    # Hard guard: contact_map modality has been removed.
    if item.get("contact_map"):
        raise ValueError(
            "contact_map modality has been removed; drop the 'contact_map' "
            "field and any '<contact_map>' placeholders from your data."
        )

    rna_pool = list(rnas)
    dna_pool = list(dnas)
    protein_pool = list(proteins)
    mol_pool = list(mols)
    image_pool = list(images)
    video_pool = list(videos)

    mol_smiles_list: List[str] = []  # SMILES consumed by <mol> placeholders

    # Sequences that need RNA encoder (input side only)
    input_sequences: List[str] = []
    # DNA sequences (independent encoder; no longer aliased to RNA)
    dna_input_sequences: List[str] = []
    # Protein sequences (separate encoder)
    protein_input_sequences: List[str] = []

    image_processor = getattr(processor, "image_processor", None)
    has_train_slot_target = False

    # ---- replacement helpers (same logic as training) ----
    def _replace_seq(text: str, tag: str, pool: list, is_assistant: bool) -> str:
        # Dispatches to the right input-sequences list based on the tag so
        # RNA and DNA flow through their own encoders.
        nonlocal input_sequences, dna_input_sequences
        # Replace only ONE occurrence — the outer _replace_all loop handles
        # interleaved ordering between different modality tags.
        if tag in text:
            if not pool:
                raise ValueError(f"More {tag} placeholders than sequences provided")
            seq = pool.pop(0)
            if is_assistant:
                text = text.replace(tag, "", 1)
            elif tag == "<dna>":
                dna_input_sequences.append(seq)
                text = text.replace(tag, _rna_input_placeholder(num_dna_latent_tokens, tag), 1)
            else:
                input_sequences.append(seq)
                text = text.replace(tag, _rna_input_placeholder(num_latent_tokens, tag), 1)
        return text

    def _replace_protein(text: str, is_assistant: bool) -> str:
        nonlocal protein_input_sequences
        # Replace only ONE occurrence — outer _replace_all controls ordering.
        if "<protein>" in text:
            if not protein_pool:
                raise ValueError("More <protein> placeholders than protein sequences provided")
            seq = protein_pool.pop(0)
            if is_assistant:
                text = text.replace("<protein>", "", 1)
            else:
                protein_input_sequences.append(seq)
                text = text.replace("<protein>", _protein_input_placeholder(num_protein_latent_tokens), 1)
        return text

    def _replace_mol(
        text: str,
        is_assistant: bool = False,
        is_last_assistant: bool = False,
    ) -> str:
        # Replace only ONE occurrence — outer _replace_all controls ordering.
        nonlocal has_train_slot_target
        if "<mol>" in text:
            is_mol_generation_target = (
                is_assistant
                and is_last_assistant
                and mol_prompt_style == "train_slots"
                and (sample_task == "mol_generation" or text.strip() == "<mol>")
            )
            if is_mol_generation_target:
                # Keep an assistant-side output window that matches the
                # training layout, but do not consume sample["mol"]. In eval
                # JSONL that field is the answer SMILES, so consuming it here
                # would leak the target into inference.
                has_train_slot_target = True
                return text.replace(
                    "<mol>", BIO_SEQ_OUTPUT_PAD * mol_generation_slots, 1
                )
            if not mol_pool:
                raise ValueError("More <mol> placeholders than mol sequences provided")
            smiles = mol_pool.pop(0)
            if is_assistant:
                # Generation target: don't build graph, just remove placeholder.
                # The turn will be skipped anyway (last assistant turn), but we
                # consume from pool to keep ordering correct.
                text = text.replace("<mol>", "", 1)
            else:
                # Input: build graph for GNN encoder
                mol_smiles_list.append(smiles)
                text = text.replace("<mol>", _mol_input_placeholder(num_mol_latent_tokens), 1)
        return text

    def _replace_all(
        text: str,
        is_assistant: bool,
        is_last_assistant: bool = False,
    ) -> str:
        # Hard guard: contact_map modality has been removed.
        if "<contact_map>" in text:
            raise ValueError(
                "contact_map modality has been removed; remove "
                "'<contact_map>' placeholders from your data."
            )
        while "<rna>" in text or "<dna>" in text or "<protein>" in text or "<mol>" in text:
            candidates = []
            for tag in ("<rna>", "<dna>", "<protein>", "<mol>"):
                pos = text.find(tag)
                if pos >= 0:
                    candidates.append((pos, tag))
            if not candidates:
                break
            _, first_tag = min(candidates, key=lambda x: x[0])
            if first_tag == "<rna>":
                text = _replace_seq(text, "<rna>", rna_pool, is_assistant)
            elif first_tag == "<dna>":
                text = _replace_seq(text, "<dna>", dna_pool, is_assistant)
            elif first_tag == "<protein>":
                text = _replace_protein(text, is_assistant)
            else:
                text = _replace_mol(text, is_assistant, is_last_assistant)
        return text

    # ---- build chat messages (only keep up to last user turn) ----
    messages = []
    conversations = item.get("conversations", [])

    # Find the index of the last assistant turn so we can skip it
    # (that's the response slot the model should generate).
    # Earlier assistant turns in multi-turn conversations are kept as context.
    last_assistant_idx = None
    for idx, turn in enumerate(conversations):
        if turn["from"] in ("gpt", "assistant"):
            last_assistant_idx = idx

    epi_cell_line = extract_epi_cell_line(item.get("task"))
    epi_injected = False

    sample_task = (item.get("task") or "").strip()
    # ``bool(dnas)`` checks the original list; dna_pool is mutated below.
    # Per-sample resolution: the caller may stamp ``_ea_label_mode`` on
    # each sample from --ea_label_mode, so the same checkpoint can be probed
    # in either mode without an env-var dance.
    _ea_n_bins = ea_n_bins_for_sample(item) if (bool(dnas) and sample_task == "EA") else None

    # Add standalone system prompt if present.
    system_prompt = item.get("system_prompt") or item.get("system")
    if system_prompt:
        text = system_prompt
        if _ea_n_bins is not None:
            text = rewrite_ea_binned_instruction(text, _ea_n_bins, "system")
        messages.append({"role": "system", "content": [{"type": "text", "text": text}]})

    for turn_idx, turn in enumerate(conversations):
        role_raw = turn["from"]
        role = "user" if role_raw == "human" else ("system" if role_raw == "system" else "assistant")
        text: str = turn["value"]
        is_assistant = role == "assistant"

        if role == "user" and epi_cell_line and not epi_injected:
            text = f"{text} Cell line: {epi_cell_line}."
            epi_injected = True

        # Same EA instruction rewrite as training-time _build_messages, so the
        # model sees identical system/user instructions at inference.
        if role in ("system", "user") and _ea_n_bins is not None:
            text = rewrite_ea_binned_instruction(text, _ea_n_bins, role)

        is_last_assistant = is_assistant and turn_idx == last_assistant_idx
        text = _replace_all(text, is_assistant, is_last_assistant)

        if (
            is_last_assistant
            and mol_prompt_style == "train_slots"
            and sample_task == "mol_generation"
            and not text.strip()
        ):
            # Some inference-formatted samples leave the final assistant turn
            # empty instead of using "<mol>".  Still give MolARDecoder the
            # same kind of assistant-side hidden-state window it saw in
            # training.
            text = BIO_SEQ_OUTPUT_PAD * mol_generation_slots
            has_train_slot_target = True

        # Skip the last assistant turn — the model should generate this response.
        # This handles both empty turns (correct inference format) and
        # non-empty turns with ground truth (training format used for eval).
        if is_last_assistant and not (
            mol_prompt_style == "train_slots" and BIO_SEQ_OUTPUT_PAD in text
        ):
            continue

        if role == "user" and ("<image>" in text or "<video>" in text):
            content = []
            parts = re.split(r"(<image>|<video>)", text)
            for seg in parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError("More <image> placeholders than images")
                    img_path = image_pool.pop(0)
                    abs_path = str((base_path / img_path).resolve()) if not Path(img_path).is_absolute() else img_path
                    content.append({"type": "image", "image": abs_path})
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError("More <video> placeholders than videos")
                    vid_path = video_pool.pop(0)
                    abs_path = str((base_path / vid_path).resolve()) if not Path(vid_path).is_absolute() else vid_path
                    content.append({"type": "video", "video": abs_path})
                elif seg:
                    content.append({"type": "text", "text": seg})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # ---- validate all pools consumed (same check as training) ----
    if rna_pool:
        raise ValueError(f"{len(rna_pool)} RNA sequence(s) unused (not consumed by <rna> placeholders)")
    if dna_pool:
        raise ValueError(f"{len(dna_pool)} DNA sequence(s) unused (not consumed by <dna> placeholders)")
    if protein_pool:
        raise ValueError(f"{len(protein_pool)} protein sequence(s) unused (not consumed by <protein> placeholders)")
    if mol_pool and not (has_train_slot_target and sample_task == "mol_generation"):
        raise ValueError(f"{len(mol_pool)} mol(s) unused (not consumed by <mol> placeholders)")
    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) unused (not consumed by <image> placeholders)")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) unused (not consumed by <video> placeholders)")

    # ---- tokenize via processor chat template ----
    # Disable Qwen3 thinking mode: training data has no <think> blocks,
    # so inference must match by not injecting <think> in the prompt.
    chat_kwargs = dict(
        tokenize=False,
        add_generation_prompt=not has_train_slot_target,
    )
    # The processor's apply_chat_template does NOT forward enable_thinking
    # to the tokenizer, so we temporarily patch the chat template.
    _patched = False
    _orig_template = None
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "chat_template"):
        _orig_template = processor.tokenizer.chat_template
        if _orig_template and "enable_thinking" in _orig_template:
            processor.tokenizer.chat_template = (
                "{%- set enable_thinking = false -%}\n" + _orig_template
            )
            _patched = True
    try:
        text_prompt = processor.apply_chat_template(messages, **chat_kwargs)
    finally:
        if _patched:
            processor.tokenizer.chat_template = _orig_template

    # Collect actual PIL images for processor
    pil_images = []
    for msg in messages:
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if part.get("type") == "image":
                    pil_images.append(Image.open(part["image"]).convert("RGB"))

    if pil_images:
        proc_out = processor(text=[text_prompt], images=pil_images, return_tensors="pt", padding=True)
    else:
        proc_out = processor(text=[text_prompt], return_tensors="pt", padding=True)

    result = dict(proc_out)

    # ---- RNA encoder inputs ----
    if input_sequences and rna_tokenizer is not None:
        rna_enc = rna_tokenizer(
            input_sequences, padding=True, truncation=True,
            max_length=rna_cap, return_tensors="pt",
        )
        result["rna_input_ids"] = rna_enc["input_ids"]
        result["rna_attention_mask"] = rna_enc["attention_mask"]
        K = num_latent_tokens
        result["rna_grid_thw"] = torch.tensor(
            [[K, 1, 1]] * len(input_sequences), dtype=torch.long,
        )

    # ---- DNA encoder inputs (independent encoder; native T) ----
    if dna_input_sequences and dna_tokenizer is not None:
        dna_enc = dna_tokenizer(
            dna_input_sequences, padding=True, truncation=True,
            max_length=dna_cap, return_tensors="pt",
        )
        result["dna_input_ids"] = dna_enc["input_ids"]
        result["dna_attention_mask"] = dna_enc["attention_mask"]
        K_dna = num_dna_latent_tokens
        result["dna_grid_thw"] = torch.tensor(
            [[K_dna, 1, 1]] * len(dna_input_sequences), dtype=torch.long,
        )

    # ---- protein encoder inputs ----
    if protein_input_sequences and protein_tokenizer is not None:
        prot_enc = protein_tokenizer(
            protein_input_sequences, padding=True, truncation=True,
            max_length=protein_cap, return_tensors="pt",
        )
        result["protein_input_ids"] = prot_enc["input_ids"]
        result["protein_attention_mask"] = prot_enc["attention_mask"]
        K_prot = num_protein_latent_tokens
        result["protein_grid_thw"] = torch.tensor(
            [[K_prot, 1, 1]] * len(protein_input_sequences), dtype=torch.long,
        )

    # ---- mol encoder inputs (SMILES -> PyG graph -> tensors) ----
    if mol_smiles_list:
        from mkb.modalities.mol.processor import smiles_to_graph
        from torch_geometric.data import Batch as PyGBatch

        graphs = []
        for smiles in mol_smiles_list:
            g = smiles_to_graph(smiles)
            if g is None:
                raise ValueError(f"Failed to process SMILES: {smiles}")
            graphs.append(g)
        mol_batch = PyGBatch.from_data_list(graphs)
        result["mol_input_ids"] = mol_batch.x
        result["mol_attention_mask"] = torch.ones(mol_batch.x.shape[0], dtype=torch.long)
        result["mol_edge_index"] = mol_batch.edge_index
        result["mol_edge_attr"] = mol_batch.edge_attr
        result["mol_edge_index_all"] = mol_batch.edge_index_all
        result["mol_batch_idx"] = mol_batch.batch
        result["mol_grid_thw"] = torch.tensor(
            [[num_mol_latent_tokens, 1, 1]] * len(mol_smiles_list), dtype=torch.long,
        )

    return result


# ---------------------------------------------------------------------------
# Decode generation output – handle RNA LM format
# ---------------------------------------------------------------------------

def decode_response(tokenizer, generated_ids: torch.Tensor) -> str:
    """Decode generated token ids, converting RNA token IDs back to sequence."""
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class BioQwen3VLInference:
    def __init__(
        self,
        model_path: str,
        processor_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "flash_attention_2",
        multi_gpu: bool = False,
        fail_on_legacy_mol_decoder: bool = False,
    ):
        print(f"Loading model from {model_path} ...")
        if fail_on_legacy_mol_decoder:
            preflight_mol_checkpoint(
                model_path,
                fail_on_legacy_mol_decoder=fail_on_legacy_mol_decoder,
            )
        if processor_path is None:
            processor_path = model_path
        if processor_path != model_path:
            print(f"Loading processor/tokenizer from {processor_path} ...")
        # transformers >=5.0 prints a per-tensor LOAD REPORT marking ckpt
        # keys without a matching module slot as UNEXPECTED. For non-medseg
        # eval the SAM3 sub-tree is intentionally absent from the model,
        # so dozens of `model.modality_router.decoders.med_seg.sam3.*`
        # keys come through. They're harmless but noisy — silence the
        # transformers logger for the duration of from_pretrained.
        import logging as _logging
        _logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)
        _logging.getLogger("transformers").setLevel(_logging.ERROR)

        if multi_gpu and torch.cuda.device_count() > 1:
            # Multi-GPU: use device_map="auto" for LLM backbone distribution,
            # then pin modality_router to the embed_tokens device so scatter_all
            # operates on a single device.
            print(f"Multi-GPU mode: distributing model across {torch.cuda.device_count()} GPUs")
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype, attn_implementation=attn_impl,
                    device_map="auto",
                )
            except Exception as e:
                print(f"flash_attention_2 unavailable, falling back to eager: {e}")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype, attn_implementation="eager",
                    device_map="auto",
                )
            # Pin modality router to the same device as embed_tokens so that
            # scatter_all's masked_scatter doesn't hit cross-device errors.
            embed_device = next(self.model.get_input_embeddings().parameters()).device
            router = getattr(self.model.model, "modality_router", None)
            if router is not None:
                router.to(embed_device)
                print(f"Modality router pinned to {embed_device}")
            self._input_device = embed_device
        else:
            # Single-device: simpler, no cross-device issues.
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype, attn_implementation=attn_impl,
                )
            except Exception as e:
                print(f"flash_attention_2 unavailable, falling back to eager: {e}")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype, attn_implementation="eager",
                )
            self.model = self.model.to(device)
            self._input_device = torch.device(device)
        self.model.eval()

        self.processor = load_auto_processor_compat(processor_path)
        self.tokenizer = self.processor.tokenizer
        # Inference is generation-only: left-pad so newly generated tokens
        # are appended on the right of each sequence. The batched collator
        # already left-pads explicitly, but set the tokenizer attribute for
        # any future caller that uses tokenizer(..., padding=True).
        self.tokenizer.padding_side = "left"

        # Register per-modality special tokens (ensures tokenizer knows them)
        from mkb.registry.token_manager import SpecialTokenManager
        token_mgr = SpecialTokenManager(self.processor.tokenizer, self.model)
        bio_token_ids = token_mgr.register_all_modality_tokens(["rna", "dna", "protein", "mol"])
        self.model.config.bio_token_ids = bio_token_ids

        # RNA tokenizer
        self.rna_tokenizer = None
        rna_cfg = getattr(self.model.config, "rna_config", None)
        router = getattr(self.model.model, "modality_router", None)
        has_rna = (rna_cfg is not None) or (router is not None and router.has_encoder("rna"))
        if has_rna:
            self.rna_tokenizer = RNACharTokenizer()
            print("RNA encoder detected")

        self.num_latent_tokens = (
            rna_cfg.num_latent_tokens if rna_cfg else RNA_NUM_LATENT_TOKENS
        )

        # DNA tokenizer (independent modality)
        self.dna_tokenizer = None
        dna_cfg = getattr(self.model.config, "dna_config", None)
        has_dna = (dna_cfg is not None) or (router is not None and router.has_encoder("dna"))
        if has_dna:
            from mkb.modalities.dna.processor import DNACharTokenizer
            self.dna_tokenizer = DNACharTokenizer()
            print("DNA encoder detected")
        self.num_dna_latent_tokens = (
            dna_cfg.num_latent_tokens if dna_cfg else RNA_NUM_LATENT_TOKENS
        )

        # Protein tokenizer — dispatch between ESMC/ESM3 (31-token) and
        # ESM2 (33-token) vocabs based on the configured backbone.
        self.protein_tokenizer = None
        protein_cfg = getattr(self.model.config, "protein_config", None)
        has_protein = (protein_cfg is not None) or (router is not None and router.has_encoder("protein"))
        if has_protein:
            from mkb.modalities.protein import make_protein_tokenizer
            backbone_name = getattr(protein_cfg, "protein_backbone_name", "") if protein_cfg else ""
            self.protein_tokenizer = make_protein_tokenizer(backbone_name)
            print(
                f"Protein encoder detected "
                f"(tokenizer: {type(self.protein_tokenizer).__name__}, "
                f"backbone: {backbone_name or 'default'})"
            )

        self.num_protein_latent_tokens = (
            protein_cfg.num_latent_tokens if protein_cfg else 32
        )

        # Mol modality detection
        mol_cfg = getattr(self.model.config, "mol_config", None)
        self.has_mol = (mol_cfg is not None) or (router is not None and router.has_encoder("mol"))
        if self.has_mol:
            print("Mol encoder detected")
        self.num_mol_latent_tokens = (
            mol_cfg.num_latent_tokens if mol_cfg else 16
        )
        self._mol_decoder_max_seq_length = (
            getattr(mol_cfg, "mol_decoder_max_seq_length", None) if mol_cfg else None
        )

        # Per-modality input-length cap. Read from ckpt config so inference
        # truncation matches what each encoder's pos_embed was trained on.
        # build_inference_inputs treats None as "fall back to max_bio_seq_length".
        self._protein_max_residues = (
            getattr(protein_cfg, "protein_max_seq_length", None) if protein_cfg else None
        )
        self._rna_max_residues = (
            getattr(rna_cfg, "rna_max_seq_length", None) if rna_cfg else None
        )
        self._dna_max_residues = (
            getattr(dna_cfg, "dna_max_seq_length", None) if dna_cfg else None
        )

        # Check if dedicated RNA decoder head is available
        self.has_rna_lm_head = False
        if hasattr(self.model, "rna_lm_head") and self.model.rna_lm_head is not None:
            self.has_rna_lm_head = True
        elif router is not None and "rna" in router.decoders:
            self.has_rna_lm_head = True
        if self.has_rna_lm_head:
            print("RNA decoder head detected")

        # Check if dedicated DNA decoder head is available
        self.has_dna_lm_head = False
        if router is not None and "dna" in router.decoders:
            self.has_dna_lm_head = True
        if self.has_dna_lm_head:
            print("DNA decoder head detected")

        # Check if dedicated protein decoder head is available
        self.has_protein_lm_head = False
        if router is not None and "protein" in router.decoders:
            self.has_protein_lm_head = True
        if self.has_protein_lm_head:
            print("Protein decoder head detected")

        # Check if dedicated mol decoder head is available
        self.has_mol_lm_head = False
        if router is not None and "mol" in router.decoders:
            self.has_mol_lm_head = True
        if self.has_mol_lm_head:
            print("Mol decoder head detected")
        self._mol_decoder_runtime_validated_strict = False
        self._validate_mol_decoder_runtime(fail_on_legacy_mol_decoder)

        # Import registered modalities so the registry is populated
        self._init_registry()
        print("Model ready.")

    def _init_registry(self):
        """Report registered modalities from ModalityRouter."""
        router = getattr(self.model.model, "modality_router", None)
        if router is not None:
            modalities = router.modality_names
            if modalities:
                print(f"ModalityRouter: registered modalities={modalities}")

    def _validate_mol_decoder_runtime(self, fail_on_legacy_mol_decoder: bool) -> None:
        """Validate the loaded mol decoder class/vocab after from_pretrained."""
        self._mol_decoder_vocab_size = None
        self._mol_tokenizer_vocab_size = None
        router = getattr(self.model.model, "modality_router", None)
        decoder = None
        if router is not None and hasattr(router, "decoders") and "mol" in router.decoders:
            decoder = router.decoders["mol"]
        if decoder is None:
            if fail_on_legacy_mol_decoder:
                raise RuntimeError(
                    "[mol-preflight] no mol decoder is registered on the loaded model."
                )
            return

        problems: List[str] = []
        if decoder.__class__.__name__ != "MolARDecoder":
            problems.append(
                f"decoder class is {decoder.__class__.__name__}, expected MolARDecoder"
            )
        try:
            from mkb.modalities.mol.tokenizer import MOL_SEP_ID, MOL_VOCAB_SIZE
            vocab_size = int(getattr(decoder, "output_size", 0))
            self._mol_decoder_vocab_size = vocab_size
            tokenizer_vocab_size = int(MOL_VOCAB_SIZE)
            self._mol_tokenizer_vocab_size = tokenizer_vocab_size
            if vocab_size <= int(MOL_SEP_ID):
                problems.append(
                    f"decoder vocab is {vocab_size}, too small for mol special tokens"
                )
            elif vocab_size != tokenizer_vocab_size:
                problems.append(
                    f"decoder vocab is {vocab_size}, tokenizer vocab is "
                    f"{tokenizer_vocab_size}; vocab must match exactly"
                )
        except Exception as exc:
            problems.append(f"could not validate mol vocab size: {exc}")

        if problems:
            message = "[mol-preflight] loaded mol decoder failed validation: " + "; ".join(problems)
            if fail_on_legacy_mol_decoder:
                raise RuntimeError(message)
            print(
                "[mol-preflight] skipped strict mol decoder validation for "
                f"non-mol inference: {'; '.join(problems)}"
            )
            return
        self._mol_decoder_runtime_validated_strict = True
        print(
            "[mol-preflight] Loaded MolARDecoder runtime validation passed "
            f"(vocab={self._mol_decoder_vocab_size})."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _is_mol_generation_item(item: Dict[str, Any]) -> bool:
        if item.get("task") == "mol_generation":
            return True
        convs = item.get("conversations") or []
        for turn in reversed(convs):
            if turn.get("from") in ("gpt", "assistant"):
                return (turn.get("value") or "").strip() == "<mol>"
        return False

    @torch.inference_mode()
    def generate_from_item(
        self,
        item: Dict[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        base_path: Path = Path(""),
        use_rna_head: Optional[bool] = None,
        use_protein_head: Optional[bool] = None,
        mol_prompt_style: str = "prompt_only",
    ) -> str:
        """Run generation on a single sample dict (training format).

        Args:
            use_rna_head: If True, use rna_lm_head for generation (RNA output).
                If None, auto-detect from task type ("generation" → True).
            use_protein_head: If True, use protein_lm_head for generation.
                If None, auto-detect from task/conversation.
        """
        if self._is_mol_generation_item(item):
            cap = getattr(self, "_mol_decoder_max_seq_length", None)
            if cap is not None and int(cap) > 0 and max_new_tokens > int(cap):
                print(
                    f"[mol-generation] max_new_tokens={max_new_tokens} exceeds "
                    f"mol_decoder_max_seq_length={int(cap)}; clipping."
                )
                max_new_tokens = int(cap)

        inputs = build_inference_inputs(
            item, self.processor, self.rna_tokenizer,
            protein_tokenizer=self.protein_tokenizer,
            dna_tokenizer=self.dna_tokenizer,
            num_latent_tokens=self.num_latent_tokens,
            num_dna_latent_tokens=self.num_dna_latent_tokens,
            num_protein_latent_tokens=self.num_protein_latent_tokens,
            num_mol_latent_tokens=self.num_mol_latent_tokens,
            base_path=base_path,
            protein_max_residues=self._protein_max_residues,
            rna_max_residues=self._rna_max_residues,
            dna_max_residues=self._dna_max_residues,
            mol_prompt_style=mol_prompt_style,
            mol_generation_slots=max_new_tokens,
        )
        device = self._input_device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Auto-detect generation mode:
        # Check task type and last assistant turn for modality markers
        is_rna_gen = use_rna_head
        is_protein_gen = use_protein_head

        is_mol_gen = None

        if is_rna_gen is None or is_protein_gen is None or is_mol_gen is None:
            task = item.get("task", "")
            # Check the last assistant turn for modality markers
            last_assistant_value = ""
            convs = item.get("conversations", [])
            for turn in reversed(convs):
                if turn.get("from") in ("gpt", "assistant"):
                    last_assistant_value = turn["value"].strip()
                    break

            # Determine mol generation (most specific task name)
            if is_mol_gen is None:
                is_mol_generation = (
                    task == "mol_generation"
                    or last_assistant_value == "<mol>"
                )
                is_mol_gen = self.has_mol_lm_head and is_mol_generation

            # Determine protein generation
            if is_protein_gen is None:
                is_protein_generation = (
                    task == "protein_generation"
                    or last_assistant_value == "<protein>"
                )
                is_protein_gen = self.has_protein_lm_head and is_protein_generation

            # RNA is the fallback for generic "generation" task
            if is_rna_gen is None:
                is_rna_generation = (
                    last_assistant_value == "<rna>"
                    or last_assistant_value == "<dna>"
                    or (task == "generation" and not is_protein_gen and not is_mol_gen)
                )
                is_rna_gen = self.has_rna_lm_head and is_rna_generation

        if is_mol_gen and not self.has_mol_lm_head:
            print(
                "[WARNING] Mol generation mode requested but mol_lm_head is not "
                "available in this checkpoint. Falling back to text decoder."
            )
            is_mol_gen = False

        if is_rna_gen and not self.has_rna_lm_head:
            print(
                "[WARNING] RNA generation mode requested but rna_lm_head is not "
                "available in this checkpoint. Falling back to text decoder."
            )
            is_rna_gen = False

        if is_protein_gen and not self.has_protein_lm_head:
            print(
                "[WARNING] Protein generation mode requested but protein_lm_head is not "
                "available in this checkpoint. Falling back to text decoder."
            )
            is_protein_gen = False

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        if repetition_penalty and abs(float(repetition_penalty) - 1.0) > 1e-8:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        input_len = inputs["input_ids"].shape[1]
        if presence_penalty and float(presence_penalty) > 0:
            gen_kwargs["logits_processor"] = LogitsProcessorList([
                PresencePenaltyLogitsProcessor(input_len, float(presence_penalty))
            ])

        # Mol generation does NOT ride the HF generate loop — its SMILES vocab
        # is fully decoupled from the LLM tokenizer.  We do one prompt
        # forward to get LLM hidden states, then let MolARDecoder run an
        # independent AR loop in mol-vocab space.
        if is_mol_gen:
            if not getattr(self, "_mol_decoder_runtime_validated_strict", False):
                self._validate_mol_decoder_runtime(True)
            self.model._mol_generation_mode = True
            try:
                mol_token_ids = self.model.generate_mol_smiles(
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    **inputs,
                )
            finally:
                self.model._mol_generation_mode = False
            try:
                from mkb.modalities.mol.tokenizer import MOL_SEP_ID
                self._last_mol_generation_finished = bool(
                    (mol_token_ids == MOL_SEP_ID).any().item()
                )
                self._last_mol_generation_token_count = int(mol_token_ids.shape[-1])
            except Exception:
                self._last_mol_generation_finished = None
                self._last_mol_generation_token_count = None
            return decode_mol_token_ids(mol_token_ids)

        # RNA / protein scatter their decoder logits into LLM vocab space and
        # ride the standard HF generate loop.
        if is_rna_gen:
            self.model._rna_generation_mode = True
        elif is_protein_gen:
            self.model._protein_generation_mode = True

        try:
            # Temporarily disable model_kwargs validation — modality-specific
            # kwargs (rna_input_ids, etc.) are consumed via **kwargs in
            # forward / prepare_inputs_for_generation, but HF's validator
            # doesn't recognize them as explicit signature parameters.
            _orig_validate = self.model._validate_model_kwargs
            self.model._validate_model_kwargs = lambda model_kwargs: None
            outputs = self.model.generate(**inputs, **gen_kwargs)
        finally:
            self.model._validate_model_kwargs = _orig_validate
            if is_rna_gen:
                self.model._rna_generation_mode = False
            elif is_protein_gen:
                self.model._protein_generation_mode = False

        generated = outputs[0, input_len:]

        if is_rna_gen:
            return self._decode_rna(generated)
        if is_protein_gen:
            return self._decode_protein(generated)
        return decode_response(self.tokenizer, generated)

    def _decode_rna(self, token_ids: torch.Tensor) -> str:
        """Decode LLM token IDs back to RNA sequence (A/U/G/C)."""
        llm_to_nuc = {32: "A", 34: "C", 38: "G", 52: "U"}
        chars = []
        for tid in token_ids.tolist():
            nuc = llm_to_nuc.get(tid)
            if nuc:
                chars.append(nuc)
        return "".join(chars)

    def _decode_protein(self, token_ids: torch.Tensor) -> str:
        """Decode LLM token IDs back to protein sequence (amino acids)."""
        llm_to_aa = {
            32: "A", 34: "C", 35: "D", 36: "E", 37: "F", 38: "G", 39: "H",
            40: "I", 42: "K", 43: "L", 44: "M", 45: "N", 47: "P", 48: "Q",
            49: "R", 50: "S", 51: "T", 53: "V", 54: "W", 56: "Y",
        }
        chars = []
        for tid in token_ids.tolist():
            aa = llm_to_aa.get(tid)
            if aa:
                chars.append(aa)
        return "".join(chars)

    # ------------------------------------------------------------------
    def generate_from_prompt(
        self,
        prompt: str,
        system: Optional[str] = None,
        rna: Optional[List[str]] = None,
        dna: Optional[List[str]] = None,
        protein: Optional[List[str]] = None,
        mol: Optional[List[str]] = None,
        image: Optional[List[str]] = None,
        task: Optional[str] = None,
        **gen_kwargs,
    ) -> str:
        """Convenience wrapper: build a sample dict from raw arguments."""
        conversations: List[Dict[str, str]] = []
        if system:
            conversations.append({"from": "system", "value": system})
        conversations.append({"from": "human", "value": prompt})
        conversations.append({"from": "gpt", "value": ""})
        item: Dict[str, Any] = {"conversations": conversations}
        if rna:
            item["rna"] = rna
        if dna:
            item["dna"] = dna
        if protein:
            item["protein"] = protein
        if mol:
            item["mol"] = mol
        if image:
            item["image"] = image
        if task:
            item["task"] = task
        return self.generate_from_item(item, **gen_kwargs)

    # ------------------------------------------------------------------
    def batch_inference(
        self,
        input_file: str,
        output_file: str,
        **gen_kwargs,
    ):
        """Process a JSONL / JSON file and write results."""
        path = Path(input_file)
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
        else:
            with open(path, "r", encoding="utf-8") as f:
                samples = json.load(f)

        # Resolve relative media paths against data_path or input file dir
        default_base = path.parent

        results = []
        for idx, sample in enumerate(samples):
            # Path("") is Path(".") and truthy, so `or default_base` would never
            # trigger; fall back only when data_path is actually empty/missing.
            base = Path(sample["data_path"]) if sample.get("data_path") else default_base
            task = sample.get("task", "unknown")
            try:
                response = self.generate_from_item(sample, base_path=base, **gen_kwargs)
            except Exception as e:
                print(f"[{idx}] Error ({task}): {e}")
                response = None

            result = {
                "id": idx,
                "task": task,
                "prompt": sample["conversations"][0]["value"] if sample.get("conversations") else "",
                "response": response,
            }
            # Preserve ground truth for evaluation
            if sample.get("conversations") and len(sample["conversations"]) > 1:
                gt = sample["conversations"][-1]["value"]
                result["ground_truth"] = gt
            results.append(result)

            if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
                print(f"  [{idx + 1}/{len(samples)}] task={task}  resp={str(response)[:80]}...")

        out_path = Path(output_file)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ok = sum(1 for r in results if r["response"] is not None)
        print(f"\nDone: {ok}/{len(results)} succeeded. Saved to {out_path}")

    # ------------------------------------------------------------------
    def chat(self, **gen_kwargs):
        """Interactive chat mode."""
        print("\n" + "=" * 50)
        print("Monkey King Bang (MKB) Interactive Chat")
        print("=" * 50)
        print("Commands:")
        print("  /rna SEQ          - add RNA sequence for next turn")
        print("  /dna SEQ          - add DNA sequence for next turn")
        print("  /protein SEQ      - add protein sequence for next turn")
        print("  /mol SMILES       - add molecule SMILES for next turn")
        print("  /image PATH       - add image for next turn")
        print("  /clear            - clear accumulated media")
        print("  /quit             - exit")
        print("=" * 50 + "\n")

        rna_seqs: List[str] = []
        dna_seqs: List[str] = []
        protein_seqs: List[str] = []
        mol_seqs: List[str] = []
        images: List[str] = []

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/clear":
                    rna_seqs, dna_seqs, protein_seqs, mol_seqs, images = [], [], [], [], []
                    print("  (cleared)")
                    continue
                elif cmd == "/rna" and arg:
                    rna_seqs.append(arg.upper())
                    print(f"  (added RNA, total={len(rna_seqs)})")
                    continue
                elif cmd == "/dna" and arg:
                    dna_seqs.append(arg.upper())
                    print(f"  (added DNA, total={len(dna_seqs)})")
                    continue
                elif cmd == "/protein" and arg:
                    protein_seqs.append(arg.upper())
                    print(f"  (added protein, total={len(protein_seqs)})")
                    continue
                elif cmd == "/mol" and arg:
                    mol_seqs.append(arg)
                    print(f"  (added mol SMILES, total={len(mol_seqs)})")
                    continue
                elif cmd == "/image" and arg:
                    images.append(arg)
                    print(f"  (added image, total={len(images)})")
                    continue
                else:
                    print("  Unknown command")
                    continue

            # Build the prompt – user must include <rna>/<dna>/<image> tags
            # in their prompt, matching training format. If they forget, remind them.
            response = self.generate_from_prompt(
                prompt=user_input,
                rna=rna_seqs if rna_seqs else None,
                dna=dna_seqs if dna_seqs else None,
                protein=protein_seqs if protein_seqs else None,
                mol=mol_seqs if mol_seqs else None,
                image=images if images else None,
                **gen_kwargs,
            )
            print(f"Assistant: {response}\n")

            # Clear media after each turn (one-shot, like training samples)
            rna_seqs, dna_seqs, protein_seqs, mol_seqs, images = [], [], [], [], []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sample_requests_mol_strict(sample: Dict[str, Any]) -> bool:
    """Return True when a CLI batch sample is a Mol-modality request."""
    if sample.get("mol"):
        return True
    if sample.get("task") == "mol_generation":
        return True
    for turn in sample.get("conversations") or []:
        value = turn.get("value") or ""
        if "<mol>" in value:
            return True
    return False


def _input_file_requests_mol_strict(input_file: Optional[str]) -> bool:
    """Best-effort scan used before model construction for CLI strict mode."""
    if not input_file:
        return False
    path = Path(input_file)
    try:
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if _sample_requests_mol_strict(json.loads(line)):
                        return True
            return False
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples = payload if isinstance(payload, list) else [payload]
        return any(_sample_requests_mol_strict(s) for s in samples if isinstance(s, dict))
    except Exception as exc:
        print(
            "[mol-preflight] could not scan input_file before model load "
            f"({exc}); keeping requested strict setting."
        )
        return True


def main():
    parser = argparse.ArgumentParser(description="Monkey King Bang (MKB) Inference")
    parser.add_argument("--model_path", type=str, required=True)

    # --- single sample mode ---
    parser.add_argument("--prompt", type=str, default=None,
                        help="User prompt (must contain <rna>/<dna>/<protein>/<mol>/<image> tags to match media)")
    parser.add_argument("--system", type=str, default=None,
                        help="System prompt. Each task has a specific one that constrains the output format; "
                             "see run_examples.sh for the per-task system prompts used to reproduce the benchmarks.")
    parser.add_argument("--rna", type=str, nargs="+", default=None,
                        help="RNA sequence(s)")
    parser.add_argument("--dna", type=str, nargs="+", default=None,
                        help="DNA sequence(s)")
    parser.add_argument("--protein", type=str, nargs="+", default=None,
                        help="Protein sequence(s)")
    parser.add_argument("--mol", type=str, nargs="+", default=None,
                        help="Molecule SMILES string(s)")
    parser.add_argument("--image", type=str, nargs="+", default=None,
                        help="Image file path(s)")

    # --- batch mode ---
    parser.add_argument("--input_file", type=str, default=None,
                        help="JSONL/JSON file with samples (training format)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL path")

    # --- chat mode ---
    parser.add_argument("--chat", action="store_true")

    # --- generation params ---
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding (overrides do_sample)")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--task", type=str, default=None,
                        choices=["generation", "protein_generation", "mol_generation", "classification", "regression", "description"],
                        help="Task type. Use 'generation' for RNA output, 'protein_generation' for protein output, "
                             "'mol_generation' for SMILES output. "
                             "Auto-detected from data in batch mode.")
    parser.add_argument("--use_rna_head", action="store_true",
                        help="Force RNA decoder head for generation (auto-detected from task field)")
    parser.add_argument("--use_protein_head", action="store_true",
                        help="Force protein decoder head for generation (auto-detected from task field)")
    parser.add_argument("--mol_prompt_style", type=str, default="prompt_only",
                        choices=["prompt_only", "train_slots"],
                        help="Mol generation prompt mode. prompt_only uses the "
                             "generation prompt consumed by generate_mol_smiles; "
                             "train_slots keeps assistant-side <|bio_seq_pad|> "
                             "slots for ablation.")
    parser.add_argument("--fail_on_legacy_mol_decoder", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Strictly validate MolARDecoder tensors and vocab before loading.")

    # --- device/dtype ---
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use device_map=auto to distribute model across multiple GPUs")

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    do_sample = not args.greedy and args.do_sample

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=do_sample,
        num_beams=args.num_beams,
        use_rna_head=args.use_rna_head if args.use_rna_head else None,
        use_protein_head=args.use_protein_head if args.use_protein_head else None,
        mol_prompt_style=args.mol_prompt_style,
    )

    strict_mol_on_init = bool(
        args.fail_on_legacy_mol_decoder
        and not args.chat
        and (
            args.task == "mol_generation"
            or bool(args.mol)
            or _input_file_requests_mol_strict(args.input_file)
        )
    )
    if args.fail_on_legacy_mol_decoder and not strict_mol_on_init:
        print(
            "[mol-preflight] initialization strict check is disabled because "
            "this CLI request is not statically a Mol request. Mol generation "
            "still validates the decoder at runtime."
        )

    inferencer = BioQwen3VLInference(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype_map[args.dtype],
        multi_gpu=args.multi_gpu,
        fail_on_legacy_mol_decoder=strict_mol_on_init,
    )

    # ---- chat mode ----
    if args.chat:
        inferencer.chat(**gen_kwargs)
        return

    # ---- batch mode ----
    if args.input_file:
        if args.output_file:
            out = args.output_file
        else:
            # Insert "_results" before the suffix; never collides with the input
            # (a suffix-less input still gets a distinct "_results.jsonl" name).
            in_path = Path(args.input_file)
            out = str(in_path.with_name(f"{in_path.stem}_results.jsonl"))
        inferencer.batch_inference(args.input_file, out, **gen_kwargs)
        return

    # ---- single sample mode ----
    if args.prompt is None:
        parser.print_help()
        print("\nError: provide --prompt, --input_file, or --chat")
        return

    response = inferencer.generate_from_prompt(
        prompt=args.prompt,
        system=args.system,
        rna=args.rna,
        dna=args.dna,
        protein=args.protein,
        mol=args.mol,
        image=args.image,
        task=args.task,
        **gen_kwargs,
    )
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
