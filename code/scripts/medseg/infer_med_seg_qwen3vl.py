#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text-grounding inference for BiomedParseData_unzip using the Qwen3-VL +
SAM3 (med_seg) training stack defined in this repo.

Architecture (matches mkb/modalities/med_seg/decoder.py):

    image+text → Qwen3-VL → user-turn JOINT hidden (image_pad block + text)
    image      → SAM3 backbone → multi-scale dense features
    joint hidden ─proj─►  text_embeds  ─cross-attn (inside SAM3)──►  cls/box/mask

Two input modes:
- Single image: ``--image <path> --prompt "<text>"`` → runs one text-grounded
  segmentation and saves the predicted mask PNG (no ground truth, no metrics).
- Dataset folder: ``--data_root <dir>`` (BiomedParse layout) → the batch path below.

What the dataset-folder mode does:
- Recursively find ``**/test.json`` under --data_root
- For each unit (dir containing test.json):
    - Run text-grounding segmentation per (image, ann) instance
    - Save predicted mask PNGs (1:1 mappable to GT mask_file paths)
    - Optional: save red-overlay visualization PNGs
    - Write a metrics JSON identical in schema to the BiomedParse reference:
        {
          "biomed_<tag>/grounding": {
            "grounding": {
              "scores": {...},
              "instance_results": [...]
            }
          }
        }

How weights are loaded:
1. ``Qwen3VLForConditionalGeneration.from_pretrained(--ckpt_path)`` — loads
   Qwen3-VL backbone + modality_router.decoders.med_seg.proj from the trained
   checkpoint. The MedSegDecoder skeleton is registered automatically via
   _register_all_modalities() because the saved config carries med_seg_config.
2. ``Sam3Model(Sam3Config.from_pretrained(--sam3_model_path))`` — builds the
   SAM3 topology from config only (no weight download), then
   ``decoder.set_sam3(sam3)``. The bundled ``model/sam3/`` dir supplies the
   config + processor.
3. Overlay ``model.modality_router.decoders.med_seg.sam3.*`` tensors from
   --ckpt_path onto the live SAM3 instance — these are the medical-domain
   fine-tuned SAM3 weights, embedded in the released checkpoint.

Distributed: torchrun launches N processes; rank 0 enumerates units and
broadcasts the list. Within each unit, samples are sharded across ranks via
DistributedSampler. Per-rank instance results are gathered to rank 0 which
writes the per-unit metrics JSON.
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Sam3Model, Sam3Processor

# Make the repo root importable when launched via torchrun from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mkb.models.modeling_bio_qwen3_vl import Qwen3VLForConditionalGeneration  # noqa: E402
from mkb.models.configuration_bio_qwen3_vl import Qwen3VLConfig  # noqa: E402
from mkb.modalities.med_seg.processor import DEFAULT_MED_SEG_SYSTEM_PROMPT  # noqa: E402
from mkb.modalities.med_seg.sam3_resize import (  # noqa: E402
    patch_sam3_for_image_size,
    patch_sam3_processor_for_image_size,
)
from mkb.utils.checkpoint_io import load_state_dict_from_ckpt_dir  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────
def dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank_env() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def get_local_rank_env() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank0() -> bool:
    return (dist.get_rank() == 0) if dist_is_ready() else (get_rank_env() == 0)


def dist_barrier():
    if dist_is_ready():
        dist.barrier()


def broadcast_object_list_py(obj_list: List[Any]) -> List[Any]:
    if not dist_is_ready():
        return obj_list
    holder = [obj_list] if is_rank0() else [None]
    dist.broadcast_object_list(holder, src=0)
    if holder[0] is None:
        raise RuntimeError("broadcast_object_list failed")
    return holder[0]


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    rank = get_rank_env()
    log_path = os.path.join(output_dir, f"infer_rank{rank}.log")

    logger = logging.getLogger("infer_med_seg_qwen3vl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)
    logger.info(f"Logging to: {log_path}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# IO utils
# ─────────────────────────────────────────────────────────────────────────────
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_bool(path: str) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.array(m)
    mx = int(arr.max()) if arr.size > 0 else 0
    if mx <= 10:
        return arr > 0
    return arr > 127


# ─────────────────────────────────────────────────────────────────────────────
# Metric utilities (BiomedParse reference schema)
# ─────────────────────────────────────────────────────────────────────────────
def calc_iou_dice(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, int, int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    I = int((pred & gt).sum())
    U = int((pred | gt).sum())
    pA = int(pred.sum())
    gA = int(gt.sum())
    iou = float(I / (U + 1e-6)) if U > 0 else (1.0 if I == 0 else 0.0)
    denom = pA + gA
    dice = float((2 * I) / (denom + 1e-6)) if denom > 0 else (1.0 if I == 0 else 0.0)
    return iou, dice, I, U, pA, gA


def summarize_scores(instance_results: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(instance_results) == 0:
        return {
            "precision@0.5": 0.0, "precision@0.6": 0.0, "precision@0.7": 0.0,
            "precision@0.8": 0.0, "precision@0.9": 0.0,
            "cIoU": 0.0, "mIoU": 0.0, "cDice": 0.0, "mDice": 0.0,
        }

    ious = np.array([float(x["IoU"][0]) for x in instance_results], dtype=np.float64)
    dices = np.array([float(x["Dice"][0]) for x in instance_results], dtype=np.float64)
    Is = np.array([float(x["I"][0]) for x in instance_results], dtype=np.float64)
    Us = np.array([float(x["U"][0]) for x in instance_results], dtype=np.float64)
    pA = np.array([float(x["pred_area"][0]) for x in instance_results], dtype=np.float64)

    # gt_area reconstructed: dice = 2I/(P+G) → G = 2I/dice − P
    gtA = (2.0 * Is / np.maximum(dices, 1e-12)) - pA
    gtA = np.maximum(gtA, 0.0)

    scores: Dict[str, float] = {}
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        scores[f"precision@{t}"] = float((ious >= t).mean() * 100.0)
    scores["cIoU"] = float(Is.sum() / (Us.sum() + 1e-12) * 100.0)
    scores["mIoU"] = float(ious.mean() * 100.0)
    scores["cDice"] = float((2.0 * Is.sum()) / (pA.sum() + gtA.sum() + 1e-12) * 100.0)
    scores["mDice"] = float(dices.mean() * 100.0)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Dataset parsing
# (BiomedParseData layout: each unit dir has test.json + test/ + test_mask/)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GroundingSample:
    image_path: str
    mask_path: Optional[str]  # None in single-image mode (no ground truth)
    prompt_text: str
    metadata: Dict[str, Any]
    save_rel: str  # relative path under dest_dir/pred_masks/


def _finite_list(xs) -> bool:
    try:
        for v in xs:
            if not math.isfinite(float(v)):
                return False
        return True
    except Exception:
        return False


def _get_text_from_ann(ann: Dict[str, Any], cat_name_by_id: Dict[int, str]) -> str:
    for k in ["text", "phrase", "caption", "sentence", "sent", "query"]:
        if k in ann and isinstance(ann[k], str) and ann[k].strip():
            return ann[k].strip()
    if "sentences" in ann and isinstance(ann["sentences"], list) and ann["sentences"]:
        v0 = ann["sentences"][0]
        if isinstance(v0, str) and v0.strip():
            return v0.strip()
        if isinstance(v0, dict):
            for kk in ["sent", "raw", "text"]:
                if kk in v0 and isinstance(v0[kk], str) and v0[kk].strip():
                    return v0[kk].strip()
    cid = ann.get("category_id", None)
    if cid is not None:
        try:
            cname = cat_name_by_id.get(int(cid), None)
            if cname:
                return str(cname)
        except Exception:
            pass
    return "visual"


def _resolve_existing_path(rel_or_abs: str, bases: List[str]) -> Optional[str]:
    if not rel_or_abs:
        return None
    if os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs):
        return rel_or_abs
    for b in bases:
        p = os.path.join(b, rel_or_abs)
        if os.path.exists(p):
            return p
    return None


def _safe_rel_from_json_path(p: str) -> str:
    import re
    p = (p or "").replace("\\", "/").strip()
    p = re.sub(r"^[A-Za-z]:", "", p)
    p = p.lstrip("/")
    parts = [x for x in p.split("/") if x not in ("", ".", "..")]
    return "/".join(parts)


def parse_test_json_to_samples(
    test_json_path: str, data_root: str, logger: logging.Logger,
) -> List[GroundingSample]:
    test_json_path = os.path.abspath(test_json_path)
    unit_dir = os.path.dirname(test_json_path)
    data_root = os.path.abspath(data_root)
    j = load_json(test_json_path)

    img_bases = [
        unit_dir,
        os.path.join(unit_dir, "test"),
        os.path.join(unit_dir, "Test"),
        os.path.join(unit_dir, "images"),
        os.path.join(unit_dir, "images", "test"),
        os.path.join(unit_dir, "imgs"),
        data_root,
    ]
    mask_bases = [
        unit_dir,
        os.path.join(unit_dir, "test_mask"),
        os.path.join(unit_dir, "test_masks"),
        os.path.join(unit_dir, "Test_mask"),
        os.path.join(unit_dir, "masks"),
        os.path.join(unit_dir, "masks", "test"),
        data_root,
    ]

    cat_name_by_id: Dict[int, str] = {}
    if isinstance(j, dict) and "categories" in j and isinstance(j["categories"], list):
        for c in j["categories"]:
            try:
                cid = int(c.get("id"))
                cat_name_by_id[cid] = str(c.get("name"))
            except Exception:
                pass

    samples: List[GroundingSample] = []

    def pick_mask_file(ann: Dict[str, Any]) -> Optional[str]:
        for k in ["mask_file", "mask_path", "mask", "segmentation_mask", "seg_mask"]:
            if k in ann and isinstance(ann[k], str) and ann[k]:
                return ann[k]
        return None

    if isinstance(j, dict) and "images" in j and "annotations" in j:
        images = {im["id"]: im for im in j["images"] if isinstance(im, dict) and "id" in im}
        for ann in j["annotations"]:
            if not isinstance(ann, dict):
                continue
            mfile = pick_mask_file(ann)
            if not mfile:
                continue
            img_id = ann.get("image_id", None)
            im = images.get(img_id, None) if img_id is not None else None
            if im is None:
                continue
            file_name = im.get("file_name") or im.get("path") or im.get("image_file")
            if not isinstance(file_name, str) or not file_name:
                continue
            img_path = _resolve_existing_path(file_name, img_bases)
            mask_path = _resolve_existing_path(mfile, mask_bases)
            if img_path is None or mask_path is None:
                continue
            prompt_text = _get_text_from_ann(ann, cat_name_by_id)

            gi = {
                "area": int(ann.get("area", 0) or 0),
                "mask_file": mfile,
                "iscrowd": int(ann.get("iscrowd", 0) or 0),
                "image_id": int(img_id) if img_id is not None else 0,
                "category_id": ann.get("category_id", None),
                "id": ann.get("id", ann.get("ann_id", 0)),
                "file_name": Path(file_name).name,
                "split": ann.get("split", "test"),
                "ann_id": ann.get("ann_id", ann.get("id", 0)),
                "ref_id": ann.get("ref_id", ann.get("id", 0)),
            }
            meta = {
                "file_name": file_name,
                "image_id": int(img_id) if img_id is not None else 0,
                "grounding_info": [gi],
            }
            mask_rel = _safe_rel_from_json_path(mfile)
            save_rel = str(Path(mask_rel).with_suffix(".png"))
            samples.append(GroundingSample(img_path, mask_path, prompt_text, meta, save_rel))

    elif isinstance(j, list):
        for item in j:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata", None)
            if not isinstance(meta, dict):
                continue
            file_name = meta.get("file_name", None)
            image_id = meta.get("image_id", 0)
            gi_list = meta.get("grounding_info", [])
            if not file_name or not isinstance(gi_list, list):
                continue
            for gi in gi_list:
                if not isinstance(gi, dict):
                    continue
                mfile = gi.get("mask_file", None)
                if not mfile:
                    continue
                img_path = _resolve_existing_path(file_name, img_bases)
                mask_path = _resolve_existing_path(mfile, mask_bases)
                if img_path is None or mask_path is None:
                    continue
                prompt_text = item.get("text") or item.get("prompt") or "visual"
                meta2 = {
                    "file_name": file_name,
                    "image_id": int(image_id),
                    "grounding_info": [gi],
                }
                mask_rel = _safe_rel_from_json_path(mfile)
                save_rel = str(Path(mask_rel).with_suffix(".png"))
                samples.append(GroundingSample(img_path, mask_path, str(prompt_text), meta2, save_rel))

    logger.info(f"[Parse] {test_json_path} -> {len(samples)} grounding instances")
    return samples


class GroundingInferDataset(Dataset):
    def __init__(self, samples: List[GroundingSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self.samples[idx]


def collate_samples(batch: List[GroundingSample]) -> List[GroundingSample]:
    return batch


def discover_all_test_jsons(data_root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(data_root):
        if "test.json" in filenames:
            out.append(os.path.join(dirpath, "test.json"))
    out.sort()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# user_text_mask — bit-identical to MedSegCollator._build_user_text_mask
#
# Covers the full Qwen3-VL native image block (<|vision_start|> ... <|vision_end|>)
# plus the trailing text query. SAM3 must see the same span at inference as it
# did at training time.
# ─────────────────────────────────────────────────────────────────────────────
def build_user_text_mask(
    input_ids_1d: torch.Tensor,
    attention_mask_1d: torch.Tensor,
    qwen_image_pad_id: Optional[int],
    qwen_vision_start_id: Optional[int],
    pad_token_id: int = 0,
) -> torch.Tensor:
    m = torch.zeros_like(input_ids_1d, dtype=torch.bool)
    if attention_mask_1d is not None:
        valid = attention_mask_1d.to(dtype=torch.bool)
    else:
        valid = input_ids_1d.ne(int(pad_token_id))
    valid_pos = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_pos.numel() == 0:
        return m
    start = int(valid_pos[0].item())
    end = int(valid_pos[-1].item())

    image_pad_pos = None
    if qwen_image_pad_id is not None:
        matches = torch.nonzero(
            input_ids_1d.eq(int(qwen_image_pad_id)) & valid, as_tuple=False
        ).flatten()
        if matches.numel() > 0:
            image_pad_pos = int(matches[0].item())

    if image_pad_pos is not None:
        start = image_pad_pos
        if qwen_vision_start_id is not None:
            prefix = input_ids_1d[: image_pad_pos + 1].eq(int(qwen_vision_start_id))
            prefix = prefix & valid[: image_pad_pos + 1]
            starts = torch.nonzero(prefix, as_tuple=False).flatten()
            if starts.numel() > 0:
                start = int(starts[-1].item())
    elif qwen_vision_start_id is not None:
        matches = torch.nonzero(
            input_ids_1d.eq(int(qwen_vision_start_id)) & valid, as_tuple=False
        ).flatten()
        if matches.numel() > 0:
            start = int(matches[0].item())

    m[start:end + 1] = True
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Single-batch inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def infer_pil_batch_probs(
    model: Qwen3VLForConditionalGeneration,
    qwen_processor: Any,
    sam3_processor: Sam3Processor,
    system_prompt: str,
    images: List[Image.Image],
    prompts: List[str],
    device: torch.device,
    model_dtype: torch.dtype,
    qwen_image_pad_id: Optional[int],
    qwen_vision_start_id: Optional[int],
    pad_token_id: int,
) -> List[np.ndarray]:
    """Run one image batch and return probability masks at each image's size."""
    orig_sizes: List[Tuple[int, int]] = [(im.size[1], im.size[0]) for im in images]  # (H, W)

    # ── Qwen3-VL inputs (chat template matches MedSegCollator) ──
    chat_texts: List[str] = []
    for ut in prompts:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "\n" + str(ut)},
            ]},
        ]
        chat_texts.append(qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        ))
    qwen_full = qwen_processor(
        text=chat_texts, images=images, return_tensors="pt", padding=True,
    )

    # user_text_mask per sample
    user_masks = []
    for i in range(len(images)):
        user_masks.append(build_user_text_mask(
            qwen_full["input_ids"][i],
            qwen_full["attention_mask"][i],
            qwen_image_pad_id=qwen_image_pad_id,
            qwen_vision_start_id=qwen_vision_start_id,
            pad_token_id=pad_token_id,
        ))
    user_text_mask = torch.stack(user_masks, dim=0).to(device)

    # ── SAM3 inputs ──
    # Sentinel box per sample (label=-10) so SAM3's geometry_encoder branch is
    # always exercised — same trick as MedSegCollator. text_grounding has no
    # real prompt boxes; SAM3 ignores -10-labeled boxes (modeling_sam3.py).
    sentinel_boxes = [[[0.0, 0.0, 1.0, 1.0]] for _ in images]
    sentinel_labels = [[-10] for _ in images]
    sam3_enc = sam3_processor(
        images=images,
        input_boxes=sentinel_boxes,
        input_boxes_labels=sentinel_labels,
        return_tensors="pt",
    )

    # Move tensors to device. Float tensors must match the model's dtype
    # (Qwen3-VL + SAM3 are loaded in bf16/fp16 here; the processors return
    # fp32 by default and will dtype-mismatch the first matmul without
    # this cast — we have no autocast wrapper at inference time).
    qwen_inputs = {
        "input_ids": qwen_full["input_ids"].to(device),  # int, no dtype cast
        "attention_mask": qwen_full["attention_mask"].to(device),
    }
    if "pixel_values" in qwen_full:
        qwen_inputs["pixel_values"] = qwen_full["pixel_values"].to(device=device, dtype=model_dtype)
    if "image_grid_thw" in qwen_full:
        qwen_inputs["image_grid_thw"] = qwen_full["image_grid_thw"].to(device)  # int

    pixel_values_sam3 = sam3_enc["pixel_values"].to(device=device, dtype=model_dtype)
    input_boxes = sam3_enc.get("input_boxes")
    input_boxes_labels = sam3_enc.get("input_boxes_labels")
    if input_boxes is not None:
        input_boxes = input_boxes.to(device=device, dtype=model_dtype)
    if input_boxes_labels is not None:
        input_boxes_labels = input_boxes_labels.to(device)  # int

    # ── Qwen3-VL forward → last_hidden_state ──
    # Use the inner Qwen3VLModel (skips lm_head matmul, mirrors training's
    # frozen-Qwen path under med_seg_freeze_qwen=True).
    inner_outputs = model.model(
        input_ids=qwen_inputs["input_ids"],
        attention_mask=qwen_inputs["attention_mask"],
        pixel_values=qwen_inputs.get("pixel_values"),
        image_grid_thw=qwen_inputs.get("image_grid_thw"),
    )
    hidden_states = inner_outputs.last_hidden_state  # [B, L, D]

    # ── MedSegDecoder.decode (same proj + slice + SAM3 forward as training) ──
    decoder = model.model.modality_router.decoders["med_seg"]
    out = decoder.decode(
        hidden_states=hidden_states,
        pixel_values_sam3=pixel_values_sam3,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        user_text_mask=user_text_mask,
        targets=None,  # only needed by compute_loss, not by decode/forward
    )

    if getattr(out, "pred_masks", None) is None:
        return [np.zeros(orig_sizes[i], dtype=np.float32) for i in range(len(images))]

    pred_masks = out.pred_masks      # [B, Q, h', w']
    pred_logits = out.pred_logits
    if pred_logits.ndim == 2:
        pred_logits = pred_logits.unsqueeze(-1)  # [B, Q, 1]

    probs: List[np.ndarray] = []
    for i in range(len(images)):
        q = int(pred_logits[i, :, 0].argmax().item())
        pm = pred_masks[i, q:q + 1]  # [1, h', w']
        H, W = orig_sizes[i]
        pm_up = F.interpolate(
            pm.unsqueeze(0).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        )[0, 0]
        probs.append(pm_up.sigmoid().detach().cpu().numpy().astype(np.float32))

    return probs


@torch.inference_mode()
def infer_batch(
    model: Qwen3VLForConditionalGeneration,
    qwen_processor: Any,
    sam3_processor: Sam3Processor,
    system_prompt: str,
    batch: List[GroundingSample],
    device: torch.device,
    model_dtype: torch.dtype,
    qwen_image_pad_id: Optional[int],
    qwen_vision_start_id: Optional[int],
    pad_token_id: int,
    mask_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[Image.Image]]:
    """Run one batch end-to-end and return (pred_bool_masks_at_orig_size, pil_images)."""
    images: List[Image.Image] = [load_image_rgb(s.image_path) for s in batch]
    prompts: List[str] = [s.prompt_text or "visual" for s in batch]
    probs = infer_pil_batch_probs(
        model=model,
        qwen_processor=qwen_processor,
        sam3_processor=sam3_processor,
        system_prompt=system_prompt,
        images=images,
        prompts=prompts,
        device=device,
        model_dtype=model_dtype,
        qwen_image_pad_id=qwen_image_pad_id,
        qwen_vision_start_id=qwen_vision_start_id,
        pad_token_id=pad_token_id,
    )
    preds = [(p > float(mask_threshold)).astype(bool) for p in probs]
    return preds, images


def make_overlapping_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
) -> List[Tuple[int, int, int, int]]:
    tile_size = max(1, int(tile_size))
    overlap = min(0.9, max(0.0, float(overlap)))
    tw = min(tile_size, int(width))
    th = min(tile_size, int(height))
    if tw >= width and th >= height:
        return [(0, 0, int(width), int(height))]

    stride_x = max(1, int(round(tw * (1.0 - overlap))))
    stride_y = max(1, int(round(th * (1.0 - overlap))))

    xs = list(range(0, max(width - tw, 0) + 1, stride_x))
    ys = list(range(0, max(height - th, 0) + 1, stride_y))
    if not xs or xs[-1] != width - tw:
        xs.append(max(width - tw, 0))
    if not ys or ys[-1] != height - th:
        ys.append(max(height - th, 0))

    seen = set()
    tiles: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            box = (int(x), int(y), int(x + tw), int(y + th))
            if box in seen:
                continue
            seen.add(box)
            tiles.append(box)
    return tiles


@torch.inference_mode()
def infer_batch_tiled(
    model: Qwen3VLForConditionalGeneration,
    qwen_processor: Any,
    sam3_processor: Sam3Processor,
    system_prompt: str,
    batch: List[GroundingSample],
    device: torch.device,
    model_dtype: torch.dtype,
    qwen_image_pad_id: Optional[int],
    qwen_vision_start_id: Optional[int],
    pad_token_id: int,
    tile_size: int,
    tile_overlap: float,
    tile_batch_size: int,
    mask_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[Image.Image]]:
    """Sliding-window inference: run prompt on every tile and max-merge probs."""
    final_preds: List[np.ndarray] = []
    pil_images: List[Image.Image] = []
    tile_batch_size = max(1, int(tile_batch_size))

    for sample in batch:
        image = load_image_rgb(sample.image_path)
        pil_images.append(image)
        W, H = image.size
        merged = np.zeros((H, W), dtype=np.float32)
        tiles = make_overlapping_tiles(W, H, int(tile_size), float(tile_overlap))
        for start in range(0, len(tiles), tile_batch_size):
            tile_boxes = tiles[start:start + tile_batch_size]
            tile_images = [image.crop(box) for box in tile_boxes]
            tile_prompts = [sample.prompt_text or "visual"] * len(tile_images)
            tile_probs = infer_pil_batch_probs(
                model=model,
                qwen_processor=qwen_processor,
                sam3_processor=sam3_processor,
                system_prompt=system_prompt,
                images=tile_images,
                prompts=tile_prompts,
                device=device,
                model_dtype=model_dtype,
                qwen_image_pad_id=qwen_image_pad_id,
                qwen_vision_start_id=qwen_vision_start_id,
                pad_token_id=pad_token_id,
            )
            for (x1, y1, x2, y2), prob in zip(tile_boxes, tile_probs):
                merged[y1:y2, x1:x2] = np.maximum(merged[y1:y2, x1:x2], prob)
        final_preds.append((merged > float(mask_threshold)).astype(bool))

    return final_preds, pil_images


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers (no scipy / cv2 dependency)
# ─────────────────────────────────────────────────────────────────────────────
def save_pred_mask_png(path: str, mask_bool: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(path)


def _mask_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    H, W = m.shape
    pad = np.pad(m, 1, constant_values=0)
    er = np.ones((H, W), dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            er &= pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    boundary = (m & (1 - er)).astype(bool)
    if width > 1:
        b = boundary.astype(np.uint8)
        for _ in range(width - 1):
            padb = np.pad(b, 1, constant_values=0)
            nb = np.zeros_like(b)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb |= padb[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            b = nb
        boundary = b.astype(bool)
    return boundary


def save_vis_overlay_png(
    path: str,
    image_rgb: Image.Image,
    mask_bool: np.ndarray,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (255, 0, 0),
    draw_contour: bool = True,
    contour_width: int = 2,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = image_rgb.convert("RGBA")
    rgba = np.array(img, dtype=np.float32)
    H, W = rgba.shape[:2]
    m = mask_bool.astype(bool)
    if m.shape != (H, W):
        m = np.array(
            Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), resample=Image.NEAREST)
        ) > 0
    if m.any():
        c = np.array([color[0], color[1], color[2], 255.0], dtype=np.float32)
        rgba[m] = rgba[m] * (1.0 - alpha) + c * alpha
        if draw_contour:
            b = _mask_boundary(m, width=int(contour_width))
            rgba[b] = c
    out = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), mode="RGBA")
    out.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Model construction (SAM3 attachment)
# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    ckpt_path: str,
    sam3_model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    logger: logging.Logger,
    sam3_image_size: Optional[int] = None,
) -> Tuple[Qwen3VLForConditionalGeneration, Any]:
    """Load Qwen3-VL+proj from ckpt, attach SAM3 (fresh topology + ckpt overlay).

    Returns (model, qwen_processor). The med_seg decoder's sam3 sub-module is
    attached and weight-overlaid; ``decoder.decode(...)`` is callable.
    """
    # Load OUR Qwen3VLConfig subclass directly (bypassing AutoConfig, which
    # may be hijacked by transformers' own qwen3_vl registration and silently
    # drop the med_seg_config field).
    config = Qwen3VLConfig.from_pretrained(ckpt_path)
    if getattr(config, "med_seg_config", None) is None:
        raise RuntimeError(
            f"[load] config.med_seg_config is None after loading {ckpt_path}/config.json. "
            f"Run scripts/segmentation/train/fix_resume_ckpt_config.py on the ckpt first."
        )
    if is_rank0():
        logger.info(
            f"[load] config.med_seg_config OK: "
            f"sam3_text_dim={config.med_seg_config.sam3_text_dim}, "
            f"mask_hw=({config.med_seg_config.mask_h},{config.med_seg_config.mask_w})"
        )

    # ── Load main model via from_pretrained ──
    # The sam3 sub-module is NOT yet attached at this point, so the ckpt's
    # ``model.modality_router.decoders.med_seg.sam3.*`` keys would be
    # reported as UNEXPECTED by transformers' loader. We silence that
    # report (it's misleading — those tensors will be loaded a few lines
    # below into the freshly-attached sam3 sub-module).
    if is_rank0():
        logger.info(f"[load] Qwen3VLForConditionalGeneration.from_pretrained({ckpt_path})")

    import transformers.utils.logging as _hf_logging
    _saved_verbosity = _hf_logging.get_verbosity()
    _hf_logging.set_verbosity_error()
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            ckpt_path, config=config, dtype=dtype,
        )
    finally:
        _hf_logging.set_verbosity(_saved_verbosity)
    model.to(device=device)
    model.eval()

    qwen_processor = AutoProcessor.from_pretrained(ckpt_path)

    # ── Ensure modalities are registered ──
    # ``from_pretrained`` may have copied/rebuilt the config internally and
    # dropped ``med_seg_config`` along the way; even if model.__init__ ran
    # ``_register_all_modalities``, the saved-state-restoration step can
    # replace ``model.config`` with a config that lost the dataclass field.
    # We restore the field on the live model.config and re-run registration.
    # The registration call is idempotent (skips already-registered modalities).
    if getattr(model.config, "med_seg_config", None) is None:
        if is_rank0():
            logger.info(
                "[load] model.config.med_seg_config is None after from_pretrained — "
                "restoring from explicitly-loaded config and re-registering modalities."
            )
        model.config.med_seg_config = config.med_seg_config
    model.model._register_all_modalities(model.config)

    # Move any newly-created modality components to correct device/dtype
    # (decoder skeleton was just instantiated on CPU/fp32 by register_modality).
    router = model.model.modality_router
    for mod_dict in (router.encoders, router.projectors, router.decoders):
        for name in mod_dict:
            mod_dict[name] = mod_dict[name].to(device=device, dtype=dtype)

    # nn.ModuleDict has no .get() — use __contains__ + __getitem__.
    if "med_seg" not in router.decoders:
        raise RuntimeError(
            "med_seg decoder still not registered after manual "
            "_register_all_modalities() call. "
            "Check mkb/modalities/med_seg/__init__.py exports "
            "MODALITY_CONFIG_KEY and register_modality."
        )
    decoder = router.decoders["med_seg"]

    # ── Attach SAM3 backbone ──
    # Build the SAM3 skeleton from its *config only* — no SAM3 weight file is
    # needed. The medical-fine-tuned SAM3 tensors are embedded in this
    # checkpoint (model.modality_router.decoders.med_seg.sam3.*) and are
    # overlaid a few lines below. sam3_model_path only supplies config.json
    # (topology); the bundled model/sam3/ dir carries exactly that.
    from transformers import Sam3Config
    if is_rank0():
        logger.info(f"[load] Sam3Model from config at {sam3_model_path} (weights come from the checkpoint)")
    sam3_cfg = Sam3Config.from_pretrained(sam3_model_path)
    # Mirror the training-time image_size override: if the checkpoint was trained
    # at a non-default image_size, pass the same --sam3_image_size, otherwise the
    # trained FPN/feature shapes won't fit.
    if sam3_image_size:
        new_mask_size = patch_sam3_for_image_size(sam3_cfg, int(sam3_image_size))
        if is_rank0():
            logger.info(
                f"[load] SAM3 image_size override: {sam3_image_size} "
                f"(mask {new_mask_size}×{new_mask_size})"
            )
    sam3 = Sam3Model(sam3_cfg).to(dtype=dtype)
    sam3.to(device=device)
    sam3.eval()

    # Heal any non-finite / blown-up cells in the freshly built SAM3 skeleton
    # (uninitialized-buffer artifacts). The trained-ckpt overlay below then
    # writes over every cell that was actually trained.
    healed_cells = 0
    healed_tensors = 0
    for n, p in sam3.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any() or p.abs().max() > 100.0:
            with torch.no_grad():
                bad_mask = torch.isnan(p) | torch.isinf(p) | (p.abs() > 100.0)
                n_bad = int(bad_mask.sum().item())
                n_total = p.numel()
                if n_bad / n_total < 0.01:
                    healthy = p[~bad_mask]
                    fill_val = healthy.float().median().to(p.dtype) if healthy.numel() > 0 \
                        else torch.zeros((), dtype=p.dtype, device=p.device)
                    p.data[bad_mask] = fill_val
                else:
                    if p.dim() >= 2:
                        fan_in = p.shape[1] * (p.shape[2:].numel() if p.dim() > 2 else 1)
                        std = (2.0 / fan_in) ** 0.5
                        p.data = torch.randn_like(p) * std
                    else:
                        p.data.zero_()
                healed_cells += n_bad
                healed_tensors += 1
    if healed_cells > 0 and is_rank0():
        logger.warning(
            f"[sam3] healed {healed_cells} corrupted cells across "
            f"{healed_tensors} tensors in fresh SAM3 weights."
        )

    decoder.set_sam3(sam3)

    # ── Overlay trained sam3 weights from the bio_qwen3vl ckpt ──
    # The sam3 sub-module was not present during from_pretrained, so its
    # tensors landed as UNEXPECTED (silenced above). We recover them here.
    src_state = load_state_dict_from_ckpt_dir(ckpt_path)
    sam3_prefix = "model.modality_router.decoders.med_seg.sam3."
    ckpt_sam3 = {
        k[len(sam3_prefix):]: v
        for k, v in src_state.items()
        if k.startswith(sam3_prefix)
    }
    if ckpt_sam3:
        ckpt_sam3 = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
            for k, v in ckpt_sam3.items()
        }
        res = sam3.load_state_dict(ckpt_sam3, strict=False)
        if is_rank0():
            logger.info(
                f"[sam3] Overlaid trained SAM3 weights from {ckpt_path}: "
                f"loaded={len(ckpt_sam3)}, missing={len(res.missing_keys)}, "
                f"unexpected={len(res.unexpected_keys)}"
            )
            if res.missing_keys:
                logger.info(f"   missing sample: {res.missing_keys[:3]}")
            if res.unexpected_keys:
                logger.info(f"   unexpected sample: {res.unexpected_keys[:3]}")
    else:
        if is_rank0():
            logger.warning(
                f"[sam3] No trained SAM3 weights in ckpt ({ckpt_path}); "
                f"using fresh facebook/sam3."
            )
    del src_state, ckpt_sam3

    return model, qwen_processor


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # Two input modes (mutually exclusive):
    #  (a) single image  : --image <path> --prompt "<text description>"
    #  (b) dataset folder : --data_root <dir> (BiomedParse layout, with GT + metrics)
    ap.add_argument("--data_root", required=False, default=None,
                    help="Dataset root in BiomedParse layout (each unit dir has "
                         "test.json + test/ + test_mask/). Folder mode: computes Dice "
                         "against ground-truth masks.")
    ap.add_argument("--image", required=False, default=None,
                    help="Single input image path. Single-image mode: pair with "
                         "--prompt; saves the predicted mask, no ground-truth/metrics.")
    ap.add_argument("--prompt", required=False, default=None,
                    help="Text description of the target region (single-image mode), "
                         "e.g. 'left heart ventricle in cardiac MRI'.")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--ckpt_path", required=True,
                    help="Trained Qwen3-VL+SAM3 checkpoint dir (e.g. .../checkpoint-23500)")
    ap.add_argument("--sam3_model_path", required=False, default=None,
                    help="SAM 3 config/processor dir (topology + processor only; no "
                         "weights needed — the fine-tuned SAM 3 weights are embedded in "
                         "the checkpoint). Optional: defaults to the bundled sam3/ subdir "
                         "under --ckpt_path (i.e. model/sam3/).")
    ap.add_argument("--system_prompt", default="")
    ap.add_argument("--system_prompt_file", default="")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_pred_masks", action="store_true")
    ap.add_argument("--skip_existing_masks", action="store_true")
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--skip_existing_vis", action="store_true")
    ap.add_argument("--tile_inference", action="store_true",
                    help="Run sliding-window tiled inference and max-merge "
                         "tile probabilities back to the original image.")
    ap.add_argument("--tile_size", type=int, default=512,
                    help="Tile edge length in original-image pixels.")
    ap.add_argument("--tile_overlap", type=float, default=0.5,
                    help="Tile overlap ratio in [0, 0.9].")
    ap.add_argument("--tile_batch_size", type=int, default=1,
                    help="How many tiles from one image to run per model batch.")
    ap.add_argument("--mask_threshold", type=float, default=0.5,
                    help="Probability threshold for final binary masks.")

    # Must match the training-time --sam3_image_size value exactly. If the
    # checkpoint was trained at 2016, eval at 1008 will silently load wrong
    # FPN tensor shapes (or fail loudly, depending on the layer).
    ap.add_argument("--sam3_image_size", type=int, default=2016,
                    help="SAM3 vision input size; must match training. This "
                         "checkpoint was trained at 2016 (the default). Only change "
                         "it if you retrain at a different size.")

    args = ap.parse_args()

    # ── Resolve input mode ──
    single_image_mode = args.image is not None
    if single_image_mode:
        if args.data_root is not None:
            ap.error("--image and --data_root are mutually exclusive; pass only one.")
        if not args.prompt:
            ap.error("--image requires --prompt (a text description of the target region).")
        if not os.path.isfile(args.image):
            ap.error(f"--image not found: {args.image}")
    elif args.data_root is None:
        ap.error("provide either --image (single-image mode) or --data_root (folder mode).")

    # ── Determinism — same ckpt + same DATA_ROOT must produce the same
    #     pred mask and the same metrics, regardless of world_size or run
    #     count. Disable cudnn benchmark/TF32 to keep matmul outputs bitwise
    #     stable across runs on the same GPU.
    import random as _random
    _random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    # Distributed init
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(get_local_rank_env())
        device = torch.device("cuda", get_local_rank_env())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(args.results_root, "_logs", args.method)
    logger = setup_logger(log_dir)

    # System prompt — use repo default if neither flag is given (training default)
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = DEFAULT_MED_SEG_SYSTEM_PROMPT

    # Resolve SAM3 source: prefer bundled <ckpt>/sam3/ subdir, fall back to
    # the explicit --sam3_model_path. This lets new ckpts be self-contained.
    bundled_sam3 = os.path.join(args.ckpt_path, "sam3")
    if os.path.isfile(os.path.join(bundled_sam3, "config.json")):
        sam3_source = bundled_sam3
    elif args.sam3_model_path:
        sam3_source = args.sam3_model_path
    else:
        raise SystemExit(
            f"SAM3 metadata not found. Either pass --sam3_model_path, or use a "
            f"checkpoint that bundles config.json under {bundled_sam3}."
        )

    if is_rank0():
        logger.info(f"data_root={args.data_root}")
        logger.info(f"results_root={args.results_root}")
        logger.info(f"method={args.method}")
        logger.info(f"ckpt={args.ckpt_path}")
        logger.info(f"sam3={sam3_source}")
        logger.info(f"device={device}")
        logger.info(f"dtype={args.dtype}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, qwen_processor = build_model(
        ckpt_path=args.ckpt_path,
        sam3_model_path=sam3_source,
        dtype=dtype,
        device=device,
        logger=logger,
        sam3_image_size=args.sam3_image_size,
    )
    sam3_processor = Sam3Processor.from_pretrained(sam3_source)
    if args.sam3_image_size:
        patch_sam3_processor_for_image_size(sam3_processor, int(args.sam3_image_size))

    # Cache native Qwen3-VL token IDs for user_text_mask construction.
    tokenizer = qwen_processor.tokenizer

    def _lookup_token(token: str) -> Optional[int]:
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
            tid = int(tid) if tid is not None else None
            return tid if tid is not None and tid >= 0 else None
        except Exception:
            return None

    qwen_image_pad_id = _lookup_token("<|image_pad|>")
    qwen_vision_start_id = _lookup_token("<|vision_start|>")
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    if is_rank0():
        logger.info(
            f"[tok] image_pad_id={qwen_image_pad_id}, "
            f"vision_start_id={qwen_vision_start_id}, "
            f"pad_token_id={pad_token_id}"
        )

    # ── Single-image mode ──────────────────────────────────────────────
    # One image + one text prompt → one predicted mask. No ground truth, no
    # metrics. Runs on rank 0 only (a single sample needs no sharding).
    if single_image_mode:
        if is_rank0():
            os.makedirs(args.results_root, exist_ok=True)
            save_name = Path(args.image).stem + "_mask.png"
            out_path = os.path.join(args.results_root, save_name)
            sample = GroundingSample(
                image_path=os.path.abspath(args.image),
                mask_path=None,
                prompt_text=args.prompt,
                metadata={"file_name": os.path.basename(args.image), "prompt": args.prompt},
                save_rel=save_name,
            )
            logger.info(f"[single-image] image={args.image}")
            logger.info(f"[single-image] prompt={args.prompt!r}")
            infer_kwargs = dict(
                model=model, qwen_processor=qwen_processor, sam3_processor=sam3_processor,
                system_prompt=system_prompt, batch=[sample], device=device, model_dtype=dtype,
                qwen_image_pad_id=qwen_image_pad_id, qwen_vision_start_id=qwen_vision_start_id,
                pad_token_id=pad_token_id, mask_threshold=float(args.mask_threshold),
            )
            if args.tile_inference:
                preds, pil_images = infer_batch_tiled(
                    **infer_kwargs, tile_size=int(args.tile_size),
                    tile_overlap=float(args.tile_overlap), tile_batch_size=int(args.tile_batch_size),
                )
            else:
                preds, pil_images = infer_batch(**infer_kwargs)
            pred_bool = preds[0]
            save_pred_mask_png(out_path, pred_bool)
            logger.info(f"[single-image] mask saved -> {out_path}  "
                        f"(foreground pixels: {int(pred_bool.sum())})")
            if args.save_vis:
                vis_path = os.path.join(args.results_root, Path(args.image).stem + "_overlay.png")
                save_vis_overlay_png(vis_path, pil_images[0], pred_bool)
                logger.info(f"[single-image] overlay saved -> {vis_path}")
        dist_barrier()
        if is_rank0():
            logger.info("Single-image inference done.")
        if dist_is_ready():
            dist.destroy_process_group()
        return

    # Discover test.json units (rank0) and broadcast
    test_jsons = discover_all_test_jsons(args.data_root) if is_rank0() else []
    test_jsons = broadcast_object_list_py(test_jsons)
    if is_rank0():
        logger.info(f"Found {len(test_jsons)} test.json files under {args.data_root}")
    dist_barrier()

    # ── Per-unit loop ──
    for test_json_path in test_jsons:
        unit_dir = os.path.dirname(os.path.abspath(test_json_path))
        rel_dir = os.path.relpath(unit_dir, os.path.abspath(args.data_root))
        rel_dir = "." if rel_dir == os.curdir else rel_dir

        dest_dir = os.path.join(args.results_root, rel_dir)
        os.makedirs(dest_dir, exist_ok=True)

        tag = rel_dir.replace(os.sep, "_") if rel_dir != "." else Path(args.data_root).name
        metrics_path = os.path.join(dest_dir, f"{args.method}_{tag}_dataset_metrics.json")

        if is_rank0():
            logger.info("=" * 80)
            logger.info(f"[Dataset] test_json={test_json_path}")
            logger.info(f"[Dataset] dest_dir={dest_dir}")
            logger.info(f"[Dataset] metrics_path={metrics_path}")
        dist_barrier()

        samples = parse_test_json_to_samples(test_json_path, args.data_root, logger)
        ds = GroundingInferDataset(samples)

        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist_is_ready() else None
        dl = DataLoader(
            ds,
            batch_size=max(1, int(args.batch_size)),
            sampler=sampler,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
            collate_fn=collate_samples,
        )

        local_results: List[Dict[str, Any]] = []
        pred_root = os.path.join(dest_dir, "pred_masks")
        vis_root = os.path.join(dest_dir, "vis")

        for batch in dl:
            # Skip-on-existing — only takes effect if every output file for the
            # batch already exists; otherwise we re-run inference (cheaper than
            # diffing per-sample status).
            if args.save_pred_masks and args.skip_existing_masks and not args.save_vis:
                if all(os.path.exists(os.path.join(pred_root, s.save_rel)) for s in batch):
                    continue
            if args.save_vis and args.skip_existing_vis and not args.save_pred_masks:
                if all(os.path.exists(os.path.join(vis_root, s.save_rel)) for s in batch):
                    continue
            if (args.save_pred_masks and args.skip_existing_masks
                    and args.save_vis and args.skip_existing_vis):
                if (all(os.path.exists(os.path.join(pred_root, s.save_rel)) for s in batch)
                        and all(os.path.exists(os.path.join(vis_root, s.save_rel)) for s in batch)):
                    continue

            try:
                infer_kwargs = dict(
                    model=model,
                    qwen_processor=qwen_processor,
                    sam3_processor=sam3_processor,
                    system_prompt=system_prompt,
                    batch=batch,
                    device=device,
                    model_dtype=dtype,
                    qwen_image_pad_id=qwen_image_pad_id,
                    qwen_vision_start_id=qwen_vision_start_id,
                    pad_token_id=pad_token_id,
                    mask_threshold=float(args.mask_threshold),
                )
                if args.tile_inference:
                    preds, pil_images = infer_batch_tiled(
                        **infer_kwargs,
                        tile_size=int(args.tile_size),
                        tile_overlap=float(args.tile_overlap),
                        tile_batch_size=int(args.tile_batch_size),
                    )
                else:
                    preds, pil_images = infer_batch(**infer_kwargs)
            except Exception as e:
                logger.warning(f"[infer] batch failed ({len(batch)} samples): {e}")
                continue

            for s, pred_bool, pil_img in zip(batch, preds, pil_images):
                try:
                    gt_bool = load_mask_bool(s.mask_path)
                except Exception:
                    continue

                if args.save_pred_masks:
                    save_pred_mask_png(os.path.join(pred_root, s.save_rel), pred_bool)
                if args.save_vis:
                    save_vis_overlay_png(os.path.join(vis_root, s.save_rel), pil_img, pred_bool)

                iou, dice, I, U, pA, _gA = calc_iou_dice(pred_bool, gt_bool)
                local_results.append({
                    "metadata": s.metadata,
                    "IoU": [float(iou)],
                    "Dice": [float(dice)],
                    "I": [int(I)],
                    "U": [int(U)],
                    "IoU_box": "",
                    "pred_area": [int(pA)],
                })

        # Gather → rank 0 writes JSON
        if dist_is_ready():
            gathered = [None for _ in range(dist.get_world_size())] if is_rank0() else None
            dist.gather_object(local_results, gathered, dst=0)
            if is_rank0():
                all_results: List[Dict[str, Any]] = []
                for part in gathered:
                    if part:
                        all_results.extend(part)
            else:
                all_results = []
        else:
            all_results = local_results

        if is_rank0():
            def _stable_key(x):
                """Stable per-instance identity for dedup + sort.

                DistributedSampler(drop_last=False) pads the tail by repeating
                samples so every rank gets the same length. Without dedup the
                metric would double-count those samples and depend on
                world_size. Use (image_id, ref_id, mask_file) — the triple is
                unique across the corpus.
                """
                meta = x.get("metadata", {})
                iid = int(meta.get("image_id", 0) or 0)
                gi = (meta.get("grounding_info") or [{}])[0]
                rid = int(gi.get("ref_id", gi.get("ann_id", gi.get("id", 0))) or 0)
                mfile = str(gi.get("mask_file", ""))
                return (iid, rid, mfile)

            seen = set()
            deduped: List[Dict[str, Any]] = []
            for r in all_results:
                k = _stable_key(r)
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(r)
            n_dropped = len(all_results) - len(deduped)
            if n_dropped > 0:
                logger.info(
                    f"[Dedup] dropped {n_dropped} duplicate instance(s) "
                    f"introduced by DistributedSampler tail-padding"
                )
            all_results = deduped
            all_results.sort(key=_stable_key)
            scores = summarize_scores(all_results)
            key = f"biomed_{tag}/grounding"
            out_obj = {
                key: {
                    "grounding": {
                        "scores": scores,
                        "instance_results": all_results,
                    }
                }
            }
            save_json(metrics_path, out_obj)
            logger.info(f"[Write] {metrics_path}  (instances={len(all_results)})")
            logger.info(f"[Scores] {scores}")

        dist_barrier()

    if is_rank0():
        logger.info("All datasets done.")
    dist_barrier()
    if dist_is_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
