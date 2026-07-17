#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified text-grounding inference for the Qwen3-VL + SAM3 (med_seg) stack.

Bundles every consistency fix and every TTA strategy into ONE script that
walks ``--data_root`` once and writes, per (unit × head), a metrics JSON in
the BiomedParse reference schema plus optional mask PNGs and overlay PNGs.

Pipeline per (image, prompt) instance
-------------------------------------
1. Build Qwen+SAM3 inputs exactly the way ``MedSegCollator`` does at training
   (chat template / user_text_mask / sentinel ``-10`` box).
2. For each enabled view (full image, optional H-flip, optional tiles, optional
   multi-scale SAM3 image_size), run ONE SAM3 forward and collect a fp32
   per-query probability map at the original H×W.
3. Fuse views (mean over views, max over tiles inside the same view).
4. Read three heads from the fused tensors:
       argmax   — highest-presence query, single mask
       semantic — pixel-wise max of (presence × mask_prob) over queries
       union    — hard OR over queries with score > τ_score
5. Threshold & metric. Save mask + overlay per head.

Training-consistency notes
--------------------------
- SAM3 ``attn_implementation`` is locked to ``BIO_SAM3_ATTN_IMPL`` (default
  ``sdpa``) so attention matches the training setting.
- Qwen image-processor pixel budget is forced to ``QWEN_MAX_PIXELS`` /
  ``QWEN_MIN_PIXELS`` so the joint hidden has the same visual-token count
  as training (training default 50176 / 784).
- All upsamples are sigmoid-first → bilinear, matching the training high-res
  dice path.

Two input modes
---------------
- Single image: ``--image <path> --prompt "<text>"`` → one text-grounded
  segmentation, saves the predicted mask PNG (no ground truth, no metrics).
- Dataset folder: ``--data_root <dir>`` (BiomedParse layout) → the full
  per-unit / per-head metrics path described above.

How SAM3 weights are loaded
---------------------------
The SAM3 topology is built from ``<ckpt>/sam3`` config only (no weight
download); the medical-domain fine-tuned SAM3 tensors are embedded in the
checkpoint (``model.modality_router.decoders.med_seg.sam3.*``) and overlaid
onto the freshly-built SAM3 instance.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
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


HEADS: Tuple[str, ...] = ("argmax", "semantic", "union")


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
# Metric utilities
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
# Dataset parsing — matches v1
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GroundingSample:
    image_path: str
    mask_path: str
    prompt_text: str
    metadata: Dict[str, Any]
    save_rel: str


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
# Per-batch SAM3 forward — returns the FULL probability tensor at the original
# image resolution: ``per_query_prob`` of shape [B, Q, H, W], plus per-query
# presence scores of shape [B, Q]. Three heads + view fusion are applied
# downstream so we can mean across views (full / flip / multi-scale) before
# any head reduction.
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _forward_one_view(
    model: Qwen3VLForConditionalGeneration,
    qwen_processor: Any,
    sam3_processor: Sam3Processor,
    system_prompt: str,
    images: List[Image.Image],
    prompts: List[str],
    out_sizes: List[Tuple[int, int]],   # target (H, W) per sample for upsample
    device: torch.device,
    model_dtype: torch.dtype,
    qwen_image_pad_id: Optional[int],
    qwen_vision_start_id: Optional[int],
    pad_token_id: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run ONE SAM3 forward and return per-sample (per_query_prob, scores).

    ``per_query_prob[i]`` is fp32 [Q, out_H, out_W] in [0, 1].
    ``scores[i]`` is fp32 [Q].
    """
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

    sentinel_boxes = [[[0.0, 0.0, 1.0, 1.0]] for _ in images]
    sentinel_labels = [[-10] for _ in images]
    sam3_enc = sam3_processor(
        images=images,
        input_boxes=sentinel_boxes,
        input_boxes_labels=sentinel_labels,
        return_tensors="pt",
    )

    qwen_inputs = {
        "input_ids": qwen_full["input_ids"].to(device),
        "attention_mask": qwen_full["attention_mask"].to(device),
    }
    if "pixel_values" in qwen_full:
        qwen_inputs["pixel_values"] = qwen_full["pixel_values"].to(
            device=device, dtype=model_dtype,
        )
    if "image_grid_thw" in qwen_full:
        qwen_inputs["image_grid_thw"] = qwen_full["image_grid_thw"].to(device)

    pixel_values_sam3 = sam3_enc["pixel_values"].to(device=device, dtype=model_dtype)
    input_boxes = sam3_enc.get("input_boxes")
    input_boxes_labels = sam3_enc.get("input_boxes_labels")
    if input_boxes is not None:
        input_boxes = input_boxes.to(device=device, dtype=model_dtype)
    if input_boxes_labels is not None:
        input_boxes_labels = input_boxes_labels.to(device)

    inner_outputs = model.model(
        input_ids=qwen_inputs["input_ids"],
        attention_mask=qwen_inputs["attention_mask"],
        pixel_values=qwen_inputs.get("pixel_values"),
        image_grid_thw=qwen_inputs.get("image_grid_thw"),
    )
    hidden_states = inner_outputs.last_hidden_state

    decoder = model.model.modality_router.decoders["med_seg"]
    out = decoder.decode(
        hidden_states=hidden_states,
        pixel_values_sam3=pixel_values_sam3,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        user_text_mask=user_text_mask,
        targets=None,
    )

    if getattr(out, "pred_masks", None) is None:
        empty_probs: List[torch.Tensor] = []
        empty_scores: List[torch.Tensor] = []
        for i in range(len(images)):
            H, W = out_sizes[i]
            empty_probs.append(torch.zeros((1, H, W), dtype=torch.float32))
            empty_scores.append(torch.zeros((1,), dtype=torch.float32))
        return empty_probs, empty_scores

    pred_masks = out.pred_masks            # [B, Q, h', w']
    pred_logits = out.pred_logits
    if pred_logits.ndim == 2:
        pred_logits = pred_logits.unsqueeze(-1)
    scores_b = pred_logits[..., 0].float().sigmoid()        # [B, Q]
    mask_prob_b = pred_masks.float().sigmoid()              # [B, Q, h', w'] — sigmoid FIRST

    per_q_probs: List[torch.Tensor] = []
    per_q_scores: List[torch.Tensor] = []
    for i in range(len(images)):
        H, W = out_sizes[i]
        mp = mask_prob_b[i].unsqueeze(0)                    # [1, Q, h', w']
        mp_up = F.interpolate(mp, size=(H, W), mode="bilinear", align_corners=False)[0]
        per_q_probs.append(mp_up.detach().cpu())            # fp32 CPU to keep VRAM low
        per_q_scores.append(scores_b[i].detach().cpu())

    return per_q_probs, per_q_scores


# ─────────────────────────────────────────────────────────────────────────────
# Tile helpers
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample full-pipeline inference
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TTAConfig:
    use_full: bool = True
    use_hflip: bool = False
    use_tile: bool = False
    tile_size: int = 512
    tile_overlap: float = 0.5
    fuse_full_tile: str = "mean"     # one of {mean, max}


@torch.inference_mode()
def infer_views_for_scale(
    *,
    image: Image.Image,
    prompt: str,
    model: Qwen3VLForConditionalGeneration,
    qwen_processor: Any,
    sam3_processor: Sam3Processor,
    system_prompt: str,
    device: torch.device,
    model_dtype: torch.dtype,
    qwen_image_pad_id: Optional[int],
    qwen_vision_start_id: Optional[int],
    pad_token_id: int,
    tta: TTAConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run every enabled view for ONE (sample, scale) and return fused tensors.

    Returns:
        per_query_prob: fp32 [Q, H, W] — fused per-query probability map at
            the original image resolution.
        per_query_score: fp32 [Q] — fused per-query presence score.
    """
    W, H = image.size

    full_probs: List[torch.Tensor] = []
    full_scores: List[torch.Tensor] = []

    def _run_full(img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        probs_list, scores_list = _forward_one_view(
            model=model,
            qwen_processor=qwen_processor,
            sam3_processor=sam3_processor,
            system_prompt=system_prompt,
            images=[img],
            prompts=[prompt],
            out_sizes=[(H, W)],
            device=device,
            model_dtype=model_dtype,
            qwen_image_pad_id=qwen_image_pad_id,
            qwen_vision_start_id=qwen_vision_start_id,
            pad_token_id=pad_token_id,
        )
        return probs_list[0], scores_list[0]

    # ── Full-image view ──
    if tta.use_full:
        p, s = _run_full(image)
        full_probs.append(p)
        full_scores.append(s)

    # ── HFlip full-image view (un-flip back before fusion) ──
    if tta.use_hflip:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        p_flip, s_flip = _run_full(flipped)
        full_probs.append(torch.flip(p_flip, dims=[-1]))
        full_scores.append(s_flip)

    # ── Tile view: max-merge per-query probs across overlapping tiles ──
    tile_prob: Optional[torch.Tensor] = None
    tile_score: Optional[torch.Tensor] = None
    if tta.use_tile:
        tiles = make_overlapping_tiles(W, H, tta.tile_size, tta.tile_overlap)
        if len(tiles) > 1:
            tile_score_cnt = 0
            for (x1, y1, x2, y2) in tiles:
                tile_img = image.crop((x1, y1, x2, y2))
                tw, th = tile_img.size
                p_tile_list, s_tile_list = _forward_one_view(
                    model=model,
                    qwen_processor=qwen_processor,
                    sam3_processor=sam3_processor,
                    system_prompt=system_prompt,
                    images=[tile_img],
                    prompts=[prompt],
                    out_sizes=[(th, tw)],
                    device=device,
                    model_dtype=model_dtype,
                    qwen_image_pad_id=qwen_image_pad_id,
                    qwen_vision_start_id=qwen_vision_start_id,
                    pad_token_id=pad_token_id,
                )
                p_tile = p_tile_list[0]
                s_tile = s_tile_list[0]
                if tile_prob is None:
                    Q = p_tile.shape[0]
                    tile_prob = torch.zeros((Q, H, W), dtype=torch.float32)
                    tile_score = torch.zeros((Q,), dtype=torch.float32)
                tile_prob[:, y1:y2, x1:x2] = torch.maximum(
                    tile_prob[:, y1:y2, x1:x2], p_tile,
                )
                tile_score = tile_score + s_tile
                tile_score_cnt += 1
            if tile_score is not None:
                tile_score = tile_score / max(1, tile_score_cnt)

    # ── Fuse: full views averaged together; full vs tile per --fuse_full_tile ──
    if full_probs and tile_prob is not None:
        full_part = torch.stack(full_probs, dim=0).mean(dim=0)
        full_scores_part = torch.stack(full_scores, dim=0).mean(dim=0)
        if tta.fuse_full_tile == "max":
            fused = torch.maximum(full_part, tile_prob)
        else:
            fused = 0.5 * (full_part + tile_prob)
        fused_scores = 0.5 * (full_scores_part + tile_score)
    elif full_probs:
        fused = torch.stack(full_probs, dim=0).mean(dim=0)
        fused_scores = torch.stack(full_scores, dim=0).mean(dim=0)
    elif tile_prob is not None:
        fused = tile_prob
        fused_scores = tile_score
    else:
        # No view enabled — degenerate; return zeros.
        fused = torch.zeros((1, H, W), dtype=torch.float32)
        fused_scores = torch.zeros((1,), dtype=torch.float32)

    return fused, fused_scores


def heads_from_fused(
    per_query_prob: torch.Tensor,    # [Q, H, W] fp32 in [0, 1]
    per_query_score: torch.Tensor,   # [Q] fp32 in [0, 1]
    score_thr: float,
    mask_thr: float,
    selected_heads: Sequence[str],
    semantic_gate: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Reduce fused per-query tensors into per-head (boolean masks, prob maps).

    Returns:
        masks: {head: bool [H, W]}    — thresholded predictions
        probs: {head: float32 [H, W]} — per-head soft probability map. Used by
            postprocessing to filter low-confidence connected components
            without needing to re-run the model.
    """
    masks_out: Dict[str, np.ndarray] = {}
    probs_out: Dict[str, np.ndarray] = {}
    s = per_query_score
    mp = per_query_prob

    if "argmax" in selected_heads:
        q = int(s.argmax().item())
        prob_a = mp[q].numpy().astype(np.float32)
        probs_out["argmax"] = prob_a
        masks_out["argmax"] = (prob_a > mask_thr).astype(bool)

    if "semantic" in selected_heads:
        # semantic_gate=True (default): pre-filter queries by score_thr before
        #   taking the score-weighted per-pixel max. Keeps low-score queries
        #   from washing out the map.
        # semantic_gate=False: keep all queries (better small-region recall on
        #   multi-target data when the queries are already well-calibrated).
        if semantic_gate:
            keep = s > score_thr
            if not keep.any():
                keep = torch.zeros_like(s, dtype=torch.bool)
                keep[s.argmax()] = True
            sel_mp = mp[keep]                                   # [k, H, W]
            sel_s = s[keep]                                     # [k]
        else:
            sel_mp = mp
            sel_s = s
        weighted = sel_mp * sel_s.view(-1, 1, 1)
        sem = weighted.max(dim=0).values.numpy().astype(np.float32)
        probs_out["semantic"] = sem
        masks_out["semantic"] = (sem > mask_thr).astype(bool)

    if "union" in selected_heads:
        keep = s > score_thr
        if not keep.any():
            keep = torch.zeros_like(s, dtype=torch.bool)
            keep[s.argmax()] = True
        # For union we still threshold per-query first then OR; the
        # representative probability of each pixel is the max prob across
        # contributing queries (used by min_cc_prob postproc).
        sel = mp[keep]                                           # [k, H, W]
        bin_per_q = (sel > mask_thr)
        masks_out["union"] = bin_per_q.any(dim=0).numpy().astype(bool)
        probs_out["union"] = sel.max(dim=0).values.numpy().astype(np.float32)

    return masks_out, probs_out


# ─────────────────────────────────────────────────────────────────────────────
# Postprocessing — keep_largest_cc / min_area / min_cc_prob / closing.
#
# The four operations are independent and applied in this fixed order:
#   1. min_area_frac    — drop CC whose pixel count is < frac × image area
#   2. min_cc_prob      — drop CC whose mean probability is < threshold
#   3. keep_largest_cc  — keep only the single largest remaining CC
#   4. close_iters      — binary closing with a 3×3 structuring element
#
# Rationale: area/probability filters first so closing doesn't merge a real
# small target with a soon-to-be-discarded noise blob; keep_largest last so
# it sees the post-filter components.
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_mask(
    mask: np.ndarray,
    prob: Optional[np.ndarray] = None,
    *,
    keep_largest_cc: bool = False,
    min_area_frac: float = 0.0,
    min_cc_prob: float = 0.0,
    close_iters: int = 0,
) -> np.ndarray:
    """Clean up a predicted mask. Returns a new boolean array (does not mutate)."""
    if not mask.any():
        return mask

    try:
        from scipy.ndimage import label as cc_label, binary_closing  # noqa: WPS433
    except Exception:                                                 # pragma: no cover
        # scipy missing — bail gracefully (postprocessing is optional).
        return mask

    out = mask.astype(bool, copy=True)

    if min_area_frac > 0.0 or min_cc_prob > 0.0:
        lbl, n = cc_label(out)
        if n > 0:
            keep_lbls = []
            total_pixels = float(out.size)
            min_area_pixels = int(min_area_frac * total_pixels) if min_area_frac > 0 else 0
            for cid in range(1, n + 1):
                cc_mask = (lbl == cid)
                area = int(cc_mask.sum())
                if min_area_pixels > 0 and area < min_area_pixels:
                    continue
                if min_cc_prob > 0.0 and prob is not None:
                    cc_mean = float(prob[cc_mask].mean()) if area > 0 else 0.0
                    if cc_mean < min_cc_prob:
                        continue
                keep_lbls.append(cid)
            if keep_lbls:
                new_mask = np.zeros_like(out)
                for cid in keep_lbls:
                    new_mask |= (lbl == cid)
                out = new_mask
            else:
                out = np.zeros_like(out)

    if keep_largest_cc and out.any():
        lbl, n = cc_label(out)
        if n > 1:
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            out = (lbl == int(sizes.argmax()))

    if close_iters > 0 and out.any():
        out = binary_closing(out, iterations=int(close_iters)).astype(bool)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
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
# Model construction — mirrors training (sdpa locked, fresh SAM3 + ckpt overlay)
# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    ckpt_path: str,
    sam3_model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    logger: logging.Logger,
    sam3_image_size: Optional[int],
    sam3_attn_impl: str,
) -> Tuple[Qwen3VLForConditionalGeneration, Any]:
    """Load Qwen3-VL+proj from ckpt, attach SAM3 (fresh topology + ckpt overlay)."""
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

    # Force training-time pixel budgets so the Qwen visual-token count
    # matches training (training default 50176 / 784).
    qwen_max_pixels = int(os.environ.get("QWEN_MAX_PIXELS", "50176"))
    qwen_min_pixels = int(os.environ.get("QWEN_MIN_PIXELS", "784"))
    img_proc = getattr(qwen_processor, "image_processor", None)
    if img_proc is not None:
        old_max = getattr(img_proc, "max_pixels", None)
        old_min = getattr(img_proc, "min_pixels", None)
        try:
            img_proc.max_pixels = qwen_max_pixels
            img_proc.min_pixels = qwen_min_pixels
        except Exception:
            pass
        if is_rank0():
            logger.info(
                f"[qwen-processor] pixel budget: "
                f"max_pixels {old_max} → {qwen_max_pixels}, "
                f"min_pixels {old_min} → {qwen_min_pixels}"
            )

    if getattr(model.config, "med_seg_config", None) is None:
        if is_rank0():
            logger.info(
                "[load] model.config.med_seg_config is None after from_pretrained — "
                "restoring from explicitly-loaded config and re-registering modalities."
            )
        model.config.med_seg_config = config.med_seg_config
    model.model._register_all_modalities(model.config)

    router = model.model.modality_router
    for mod_dict in (router.encoders, router.projectors, router.decoders):
        for name in mod_dict:
            mod_dict[name] = mod_dict[name].to(device=device, dtype=dtype)

    if "med_seg" not in router.decoders:
        raise RuntimeError("med_seg decoder still not registered after manual re-registration.")
    decoder = router.decoders["med_seg"]

    # Build the SAM3 skeleton from its *config only* — no SAM3 weight file is
    # needed. The medical-fine-tuned SAM3 tensors are embedded in this
    # checkpoint (model.modality_router.decoders.med_seg.sam3.*) and are
    # overlaid a few lines below. sam3_model_path only supplies config.json
    # (topology); the bundled model/sam3/ dir carries exactly that.
    from transformers import Sam3Config
    sam3_cfg = Sam3Config.from_pretrained(sam3_model_path)
    if sam3_image_size:
        new_mask_size = patch_sam3_for_image_size(sam3_cfg, int(sam3_image_size))
        if is_rank0():
            logger.info(
                f"[load] SAM3 image_size override: {sam3_image_size} "
                f"(mask {new_mask_size}×{new_mask_size})"
            )
    # Request the desired attention backend on the config (matches training's
    # sdpa). Set on the top config and any attention-bearing sub-configs.
    try:
        sam3_cfg._attn_implementation = sam3_attn_impl
        for _sub in ("vision_config", "text_config", "prompt_encoder_config",
                     "mask_decoder_config"):
            _c = getattr(sam3_cfg, _sub, None)
            if _c is not None:
                _c._attn_implementation = sam3_attn_impl
    except Exception:
        pass
    if is_rank0():
        logger.info(
            f"[load] Sam3Model from config at {sam3_model_path} "
            f"(weights come from the checkpoint; attn={sam3_attn_impl!r})"
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
    else:
        if is_rank0():
            logger.warning(
                f"[sam3] No trained SAM3 weights found in ckpt ({ckpt_path}); "
                f"SAM3 stays at its (untrained) config-initialized values."
            )
    del src_state, ckpt_sam3

    return model, qwen_processor


# ─────────────────────────────────────────────────────────────────────────────
# Build extra SAM3 backbones + processors for multi-scale TTA. FPN feature
# sizes are baked in per image_size, so the primary backbone cannot simply be
# resized at runtime; each extra scale gets its own SAM3 (built from config,
# then weight-overlaid from the checkpoint) and a processor patched to the
# matching ``size`` / ``mask_size``. FPN tensors whose shape changed with the
# new image_size are left at their config-initialized values (strict=False).
# ─────────────────────────────────────────────────────────────────────────────
def build_extra_sam3_models_and_processors(
    sam3_source: str,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
    extra_sizes: Sequence[int],
    sam3_attn_impl: str,
    logger: logging.Logger,
) -> Dict[int, Tuple[Sam3Model, Sam3Processor]]:
    """For each extra image_size, load a freshly-resized SAM3 backbone and
    overlay the trained sam3 weights. Returns {image_size: (sam3, processor)}.
    Memory cost: ~0.5 B params × len(extra_sizes). Skip if you can't afford it.
    """
    out: Dict[int, Tuple[Sam3Model, Sam3Processor]] = {}
    if not extra_sizes:
        return out

    src_state = load_state_dict_from_ckpt_dir(ckpt_path) or {}
    sam3_prefix = "model.modality_router.decoders.med_seg.sam3."
    ckpt_sam3 = {
        k[len(sam3_prefix):]: v
        for k, v in src_state.items()
        if k.startswith(sam3_prefix)
    }
    del src_state

    from transformers import Sam3Config

    for image_size in extra_sizes:
        if is_rank0():
            logger.info(f"[multi-scale] building extra SAM3 at image_size={image_size}")
        # Config-only build + ckpt overlay (same as build_model); model/sam3
        # ships no weights.
        sam3_cfg = Sam3Config.from_pretrained(sam3_source)
        new_mask_size = patch_sam3_for_image_size(sam3_cfg, int(image_size))
        try:
            sam3_cfg._attn_implementation = sam3_attn_impl
            for _sub in ("vision_config", "text_config", "prompt_encoder_config",
                         "mask_decoder_config"):
                _c = getattr(sam3_cfg, _sub, None)
                if _c is not None:
                    _c._attn_implementation = sam3_attn_impl
        except Exception:
            pass
        sam3_extra = Sam3Model(sam3_cfg).to(dtype=dtype)
        sam3_extra.to(device=device)
        sam3_extra.eval()

        # Heal then overlay (FPN feature sizes change with image_size; the
        # FPN tensors in the ckpt were trained at the primary size — load
        # strict=False so size-mismatched FPN tensors silently keep the
        # freshly-built skeleton's values).
        for n, p in sam3_extra.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any() or p.abs().max() > 100.0:
                with torch.no_grad():
                    bad_mask = torch.isnan(p) | torch.isinf(p) | (p.abs() > 100.0)
                    healthy = p[~bad_mask]
                    fill_val = healthy.float().median().to(p.dtype) if healthy.numel() > 0 \
                        else torch.zeros((), dtype=p.dtype, device=p.device)
                    p.data[bad_mask] = fill_val

        if ckpt_sam3:
            cks = {
                k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
                for k, v in ckpt_sam3.items()
            }
            res = sam3_extra.load_state_dict(cks, strict=False)
            if is_rank0():
                logger.info(
                    f"[multi-scale] overlay @ size={image_size}: "
                    f"loaded={len(cks)}, missing={len(res.missing_keys)}, "
                    f"unexpected={len(res.unexpected_keys)} "
                    f"(missing/unexpected at this scale typically come from "
                    f"FPN tensors whose shape changed with image_size; that is "
                    f"expected)"
                )

        proc = Sam3Processor.from_pretrained(sam3_source)
        patch_sam3_processor_for_image_size(proc, int(image_size))
        out[int(image_size)] = (sam3_extra, proc)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hot-swap helper: temporarily attach an alternate SAM3 backbone to the same
# MedSegDecoder, run a forward, then restore the primary. This avoids cloning
# the whole Qwen3VL model per scale.
# ─────────────────────────────────────────────────────────────────────────────
class _SwapSam3:
    def __init__(self, model: Qwen3VLForConditionalGeneration, alt_sam3: Sam3Model):
        self.decoder = model.model.modality_router.decoders["med_seg"]
        self.alt = alt_sam3
        self.saved = None

    def __enter__(self):
        self.saved = self.decoder.sam3
        self.decoder.sam3 = self.alt
        return self

    def __exit__(self, *exc):
        self.decoder.sam3 = self.saved


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # ── Input: either a single image, or a dataset folder (mutually exclusive) ──
    ap.add_argument("--data_root", required=False, default=None,
                    help="BiomedParse-layout folder (recursively finds test.json). "
                         "Folder mode: writes per-unit / per-head metrics + masks.")
    ap.add_argument("--image", required=False, default=None,
                    help="Single-image mode: path to one image. Requires --prompt. "
                         "Mutually exclusive with --data_root.")
    ap.add_argument("--prompt", required=False, default=None,
                    help="Single-image mode: text description of the target region "
                         "(e.g. 'liver', 'left ventricle', 'polyp').")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--method", required=True,
                    help="Output filename prefix (e.g. mkb).")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--sam3_model_path", required=False, default=None)
    ap.add_argument("--single_image_head", default="argmax",
                    choices=["argmax", "semantic", "union"],
                    help="Which head's mask to save in single-image mode.")
    ap.add_argument("--system_prompt", default="")
    ap.add_argument("--system_prompt_file", default="")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_pred_masks", action="store_true")
    ap.add_argument("--skip_existing_masks", action="store_true")
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--skip_existing_vis", action="store_true")

    # ── Heads ──
    ap.add_argument("--heads", default="argmax,semantic,union",
                    help="Comma-separated subset of {argmax,semantic,union}.")
    ap.add_argument("--score_thr", type=float, default=0.5,
                    help="Per-query presence-score threshold (union head only).")
    ap.add_argument("--mask_thr", type=float, default=0.5,
                    help="Per-pixel mask probability threshold (all heads).")
    ap.add_argument("--semantic_gate", action=argparse.BooleanOptionalAction, default=True,
                    help="Pre-filter queries by score_thr in the semantic head "
                         "(default on). Pass --no-semantic_gate to keep all "
                         "queries for better small-region recall on multi-target data.")

    # ── TTA flags ──
    ap.add_argument("--use_full", action=argparse.BooleanOptionalAction, default=True,
                    help="Run the primary full-image view.")
    ap.add_argument("--use_hflip", action="store_true",
                    help="Add a horizontally-flipped full-image view (un-flipped before fusion).")
    ap.add_argument("--use_tile", action=argparse.BooleanOptionalAction, default=True,
                    help="Add a sliding-window tile view (max-merge per-query). "
                         "On by default; pass --no-use_tile for a faster single-view run.")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--tile_overlap", type=float, default=0.5)
    ap.add_argument("--fuse_full_tile", default="mean", choices=["mean", "max"],
                    help="How to combine the (full-image, tile) view pair.")
    ap.add_argument("--multi_scale_image_sizes", default="",
                    help="Comma-separated extra SAM3 image_sizes for multi-scale TTA "
                         "(in addition to --sam3_image_size). Each must be a clean "
                         "336*k size, e.g. '1680,1344'.")

    # ── SAM3 ──
    ap.add_argument("--sam3_image_size", type=int, default=2016,
                    help="Primary SAM3 input size; must match training. This "
                         "checkpoint was trained at 2016 (the default); only "
                         "change it if you retrain at a different size.")
    ap.add_argument("--sam3_attn_impl", default=None,
                    help="SAM3 attention impl. Defaults to BIO_SAM3_ATTN_IMPL or 'sdpa' "
                         "(matches training).")

    # ── Postprocessing (per-head) ──
    # Each flag accepts a single "value" (applied to every head) or a
    # comma-separated "head:value,head:value,..." spec. Examples:
    #   --pp_keep_largest_cc argmax:1,semantic:0,union:0
    #   --pp_min_cc_prob 0.55
    #   --pp_min_area_frac 0.0001
    #   --pp_close_iters argmax:1,semantic:0,union:0
    ap.add_argument("--pp_keep_largest_cc", default="",
                    help="Per-head keep-largest-connected-component. Form "
                         "'value' or 'head:value,...'. value ∈ {0,1}.")
    ap.add_argument("--pp_min_area_frac", default="",
                    help="Per-head min-area filter as a fraction of image area "
                         "(e.g. 1e-4 ≈ 0.01%%). Drops connected components below.")
    ap.add_argument("--pp_min_cc_prob", default="0.55",
                    help="Per-head min mean-probability for a connected component "
                         "(default 0.55; set 0 to disable). Real targets stay near "
                         "0.7-0.9; noise tends to sit in the 0.5-0.55 sigmoid foothill.")
    ap.add_argument("--pp_close_iters", default="",
                    help="Per-head morphological closing iterations (3x3 SE). "
                         "Smooths 1-2 px border jitter. Avoid on multi-target "
                         "(e.g. vessels / cells) — it merges adjacent objects.")

    args = ap.parse_args()

    # ── Resolve input mode (single image vs dataset folder) ──
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

    # ── Determinism ──
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

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(get_local_rank_env())
        device = torch.device("cuda", get_local_rank_env())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(args.results_root, "_logs", args.method)
    logger = setup_logger(log_dir)

    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = DEFAULT_MED_SEG_SYSTEM_PROMPT

    # Resolve SAM3 metadata source: bundled <ckpt>/sam3 wins.
    bundled_sam3 = os.path.join(args.ckpt_path, "sam3")
    if os.path.isfile(os.path.join(bundled_sam3, "config.json")):
        sam3_source = bundled_sam3
    elif args.sam3_model_path:
        sam3_source = args.sam3_model_path
    else:
        raise SystemExit(
            f"SAM3 metadata not found. Pass --sam3_model_path or use a "
            f"checkpoint that bundles config.json under {bundled_sam3}."
        )

    sam3_attn_impl = (
        args.sam3_attn_impl
        or os.environ.get("BIO_SAM3_ATTN_IMPL")
        or "sdpa"
    )

    if single_image_mode:
        # Single-image mode outputs exactly one mask — only compute its head.
        selected_heads = (args.single_image_head,)
    else:
        selected_heads = tuple(h.strip() for h in args.heads.split(",") if h.strip())
    for h in selected_heads:
        if h not in HEADS:
            raise ValueError(f"Unknown head '{h}'. Allowed: {HEADS}")

    def _parse_per_head(spec: str, cast, default):
        """Parse 'value' or 'head:value,...' into {head: cast(value)}."""
        out = {h: default for h in HEADS}
        if not spec:
            return out
        spec = spec.strip()
        if ":" not in spec:
            v = cast(spec)
            return {h: v for h in HEADS}
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            head, val = chunk.split(":", 1)
            head = head.strip()
            if head not in HEADS:
                raise ValueError(f"Unknown head in postproc spec: {head!r}")
            out[head] = cast(val.strip())
        return out

    def _bool_cast(v) -> bool:
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "on"}

    pp_keep_largest = _parse_per_head(args.pp_keep_largest_cc, _bool_cast, False)
    pp_min_area_frac = _parse_per_head(args.pp_min_area_frac, float, 0.0)
    pp_min_cc_prob = _parse_per_head(args.pp_min_cc_prob, float, 0.0)
    pp_close_iters = _parse_per_head(args.pp_close_iters, int, 0)

    extra_sizes: Tuple[int, ...] = tuple(
        int(x.strip()) for x in args.multi_scale_image_sizes.split(",") if x.strip()
    )

    if is_rank0():
        logger.info(f"data_root={args.data_root}")
        logger.info(f"results_root={args.results_root}")
        logger.info(f"method={args.method}")
        logger.info(f"ckpt={args.ckpt_path}")
        logger.info(f"sam3={sam3_source}")
        logger.info(f"sam3_attn_impl={sam3_attn_impl}")
        logger.info(f"sam3_image_size={args.sam3_image_size}")
        logger.info(f"multi_scale_extra_sizes={extra_sizes}")
        logger.info(f"heads={selected_heads}")
        logger.info(
            f"TTA: full={args.use_full}, hflip={args.use_hflip}, "
            f"tile={args.use_tile} (size={args.tile_size}, overlap={args.tile_overlap}, "
            f"fuse={args.fuse_full_tile})"
        )
        logger.info(
            f"score_thr={args.score_thr}, mask_thr={args.mask_thr}, "
            f"semantic_gate={args.semantic_gate}"
        )
        logger.info(
            f"postproc: keep_largest_cc={pp_keep_largest}, "
            f"min_area_frac={pp_min_area_frac}, "
            f"min_cc_prob={pp_min_cc_prob}, "
            f"close_iters={pp_close_iters}"
        )
        logger.info(f"device={device}, dtype={args.dtype}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, qwen_processor = build_model(
        ckpt_path=args.ckpt_path,
        sam3_model_path=sam3_source,
        dtype=dtype,
        device=device,
        logger=logger,
        sam3_image_size=args.sam3_image_size,
        sam3_attn_impl=sam3_attn_impl,
    )

    sam3_processor_primary = Sam3Processor.from_pretrained(sam3_source)
    if args.sam3_image_size:
        patch_sam3_processor_for_image_size(sam3_processor_primary, int(args.sam3_image_size))

    # Build extra SAM3 backbones + processors for multi-scale TTA.
    extra_models_and_processors = build_extra_sam3_models_and_processors(
        sam3_source=sam3_source,
        ckpt_path=args.ckpt_path,
        dtype=dtype,
        device=device,
        extra_sizes=extra_sizes,
        sam3_attn_impl=sam3_attn_impl,
        logger=logger,
    )

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

    tta = TTAConfig(
        use_full=bool(args.use_full),
        use_hflip=bool(args.use_hflip),
        use_tile=bool(args.use_tile),
        tile_size=int(args.tile_size),
        tile_overlap=float(args.tile_overlap),
        fuse_full_tile=str(args.fuse_full_tile),
    )

    # ── Single-image mode ──────────────────────────────────────────────
    # One image + one text prompt → one predicted mask via the full
    # TTA / multi-scale / head / postproc pipeline. No ground truth, no
    # metrics. Runs on rank 0 only (a single sample needs no sharding).
    if single_image_mode:
        if is_rank0():
            os.makedirs(args.results_root, exist_ok=True)
            head = args.single_image_head
            pil_img = load_image_rgb(args.image)
            logger.info(f"[single-image] image={args.image}")
            logger.info(f"[single-image] prompt={args.prompt!r}  head={head}")

            scale_probs: List[torch.Tensor] = []
            scale_scores: List[torch.Tensor] = []
            fused_p, fused_s = infer_views_for_scale(
                image=pil_img, prompt=args.prompt, model=model,
                qwen_processor=qwen_processor, sam3_processor=sam3_processor_primary,
                system_prompt=system_prompt, device=device, model_dtype=dtype,
                qwen_image_pad_id=qwen_image_pad_id,
                qwen_vision_start_id=qwen_vision_start_id,
                pad_token_id=pad_token_id, tta=tta,
            )
            scale_probs.append(fused_p)
            scale_scores.append(fused_s)
            for image_size, (alt_sam3, alt_proc) in extra_models_and_processors.items():
                with _SwapSam3(model, alt_sam3):
                    fp, fs = infer_views_for_scale(
                        image=pil_img, prompt=args.prompt, model=model,
                        qwen_processor=qwen_processor, sam3_processor=alt_proc,
                        system_prompt=system_prompt, device=device, model_dtype=dtype,
                        qwen_image_pad_id=qwen_image_pad_id,
                        qwen_vision_start_id=qwen_vision_start_id,
                        pad_token_id=pad_token_id, tta=tta,
                    )
                scale_probs.append(fp)
                scale_scores.append(fs)

            final_prob = torch.stack(scale_probs, dim=0).mean(dim=0)
            final_score = torch.stack(scale_scores, dim=0).mean(dim=0)
            masks_per_head, probs_per_head = heads_from_fused(
                per_query_prob=final_prob, per_query_score=final_score,
                score_thr=float(args.score_thr), mask_thr=float(args.mask_thr),
                selected_heads=selected_heads, semantic_gate=bool(args.semantic_gate),
            )
            pred_bool = postprocess_mask(
                masks_per_head[head], prob=probs_per_head.get(head),
                keep_largest_cc=bool(pp_keep_largest[head]),
                min_area_frac=float(pp_min_area_frac[head]),
                min_cc_prob=float(pp_min_cc_prob[head]),
                close_iters=int(pp_close_iters[head]),
            )

            save_name = Path(args.image).stem + "_mask.png"
            out_path = os.path.join(args.results_root, save_name)
            save_pred_mask_png(out_path, pred_bool)
            logger.info(f"[single-image] mask saved -> {out_path}  "
                        f"(foreground pixels: {int(pred_bool.sum())})")
            if args.save_vis:
                vis_path = os.path.join(args.results_root, Path(args.image).stem + "_overlay.png")
                save_vis_overlay_png(vis_path, pil_img, pred_bool)
                logger.info(f"[single-image] overlay saved -> {vis_path}")
            logger.info("Single-image inference done.")
        dist_barrier()
        if dist_is_ready():
            dist.destroy_process_group()
        return

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

        if is_rank0():
            logger.info("=" * 80)
            logger.info(f"[Dataset] test_json={test_json_path}")
            logger.info(f"[Dataset] dest_dir={dest_dir}")
        dist_barrier()

        samples = parse_test_json_to_samples(test_json_path, args.data_root, logger)
        ds = GroundingInferDataset(samples)

        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist_is_ready() else None
        dl = DataLoader(
            ds,
            batch_size=1,                         # Per-sample loop simplifies multi-view fusion
            sampler=sampler,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
            collate_fn=collate_samples,
        )

        local_results: Dict[str, List[Dict[str, Any]]] = {h: [] for h in selected_heads}
        pred_root = os.path.join(dest_dir, "pred_masks")
        vis_root = os.path.join(dest_dir, "vis")

        # Per-rank progress bar. tqdm to stderr; rank0 stays unmuted, the
        # other ranks are muted to keep the terminal readable. Bar length is
        # the per-rank shard length, not the global sample count, so each
        # rank's progress is meaningful.
        _pbar_total = len(dl)
        _pbar = tqdm(
            total=_pbar_total,
            desc=f"[{tag}] rank={get_rank_env()}",
            disable=(not is_rank0()),
            dynamic_ncols=True,
            mininterval=1.0,
        )

        for batch in dl:
            sample = batch[0]
            _pbar.update(1)
            # Skip-on-existing — only when EVERY selected head's outputs exist.
            if args.save_pred_masks and args.skip_existing_masks and not args.save_vis:
                if all(
                    os.path.exists(os.path.join(pred_root, h, sample.save_rel))
                    for h in selected_heads
                ):
                    continue
            if args.save_vis and args.skip_existing_vis and not args.save_pred_masks:
                if all(
                    os.path.exists(os.path.join(vis_root, h, sample.save_rel))
                    for h in selected_heads
                ):
                    continue
            if (args.save_pred_masks and args.skip_existing_masks
                    and args.save_vis and args.skip_existing_vis):
                if (
                    all(os.path.exists(os.path.join(pred_root, h, sample.save_rel))
                        for h in selected_heads)
                    and all(os.path.exists(os.path.join(vis_root, h, sample.save_rel))
                            for h in selected_heads)
                ):
                    continue

            try:
                pil_img = load_image_rgb(sample.image_path)
                prompt = sample.prompt_text or "visual"

                # ── Per-scale: each scale uses its own SAM3 backbone, runs
                # full / hflip / tile views inside ``infer_views_for_scale``
                # and returns a fused per-query probability tensor. We then
                # average across scales — probability-level fusion gives
                # finer-grained calibration than majority voting. ──
                scale_probs: List[torch.Tensor] = []
                scale_scores: List[torch.Tensor] = []

                fused_p, fused_s = infer_views_for_scale(
                    image=pil_img,
                    prompt=prompt,
                    model=model,
                    qwen_processor=qwen_processor,
                    sam3_processor=sam3_processor_primary,
                    system_prompt=system_prompt,
                    device=device,
                    model_dtype=dtype,
                    qwen_image_pad_id=qwen_image_pad_id,
                    qwen_vision_start_id=qwen_vision_start_id,
                    pad_token_id=pad_token_id,
                    tta=tta,
                )
                scale_probs.append(fused_p)
                scale_scores.append(fused_s)

                for image_size, (alt_sam3, alt_proc) in extra_models_and_processors.items():
                    with _SwapSam3(model, alt_sam3):
                        fp, fs = infer_views_for_scale(
                            image=pil_img,
                            prompt=prompt,
                            model=model,
                            qwen_processor=qwen_processor,
                            sam3_processor=alt_proc,
                            system_prompt=system_prompt,
                            device=device,
                            model_dtype=dtype,
                            qwen_image_pad_id=qwen_image_pad_id,
                            qwen_vision_start_id=qwen_vision_start_id,
                            pad_token_id=pad_token_id,
                            tta=tta,
                        )
                    scale_probs.append(fp)
                    scale_scores.append(fs)

                final_prob = torch.stack(scale_probs, dim=0).mean(dim=0)
                final_score = torch.stack(scale_scores, dim=0).mean(dim=0)
                masks_per_head, probs_per_head = heads_from_fused(
                    per_query_prob=final_prob,
                    per_query_score=final_score,
                    score_thr=float(args.score_thr),
                    mask_thr=float(args.mask_thr),
                    selected_heads=selected_heads,
                    semantic_gate=bool(args.semantic_gate),
                )

                # Per-head postprocessing — drop low-confidence / tiny CCs,
                # optionally keep only the largest CC, optionally close.
                for h in selected_heads:
                    masks_per_head[h] = postprocess_mask(
                        masks_per_head[h],
                        prob=probs_per_head.get(h),
                        keep_largest_cc=bool(pp_keep_largest[h]),
                        min_area_frac=float(pp_min_area_frac[h]),
                        min_cc_prob=float(pp_min_cc_prob[h]),
                        close_iters=int(pp_close_iters[h]),
                    )

            except Exception as e:
                logger.warning(f"[infer] sample failed ({sample.image_path}): {e}")
                continue

            try:
                gt_bool = load_mask_bool(sample.mask_path)
            except Exception:
                continue

            for h in selected_heads:
                pred_bool = masks_per_head[h]

                if args.save_pred_masks:
                    save_pred_mask_png(
                        os.path.join(pred_root, h, sample.save_rel), pred_bool,
                    )
                if args.save_vis:
                    save_vis_overlay_png(
                        os.path.join(vis_root, h, sample.save_rel),
                        pil_img, pred_bool,
                    )

                iou, dice, I, U, pA, _gA = calc_iou_dice(pred_bool, gt_bool)
                local_results[h].append({
                    "metadata": sample.metadata,
                    "IoU": [float(iou)],
                    "Dice": [float(dice)],
                    "I": [int(I)],
                    "U": [int(U)],
                    "IoU_box": "",
                    "pred_area": [int(pA)],
                })

        _pbar.close()

        # ── Gather + write metrics per head ──
        for h in selected_heads:
            metrics_path = os.path.join(
                dest_dir, f"{args.method}_{tag}_dataset_metrics_{h}.json",
            )

            if dist_is_ready():
                gathered = [None for _ in range(dist.get_world_size())] if is_rank0() else None
                dist.gather_object(local_results[h], gathered, dst=0)
                if is_rank0():
                    all_results: List[Dict[str, Any]] = []
                    for part in gathered:
                        if part:
                            all_results.extend(part)
                else:
                    all_results = []
            else:
                all_results = local_results[h]

            if is_rank0():
                def _stable_key(x):
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
                        f"[Dedup/{h}] dropped {n_dropped} duplicate instance(s) "
                        f"introduced by DistributedSampler tail-padding"
                    )
                deduped.sort(key=_stable_key)
                scores = summarize_scores(deduped)
                key = f"biomed_{tag}/grounding"
                out_obj = {
                    key: {
                        "grounding": {
                            "scores": scores,
                            "instance_results": deduped,
                        }
                    }
                }
                save_json(metrics_path, out_obj)
                logger.info(
                    f"[Write/{h}] {metrics_path}  (instances={len(deduped)})"
                )
                logger.info(f"[Scores/{h}] {scores}")

        dist_barrier()

    if is_rank0():
        logger.info("All datasets done.")
    dist_barrier()
    if dist_is_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
