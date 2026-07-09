"""Processor for the med_seg (SAM3) modality.

Responsibilities:
1. Parse a BiomedParse-style JSON sample into ``InstanceAnn`` objects
   (box / mask / text / category) — port of sam_uni's BiomedMultiTaskDataset.
2. Produce the chat-template placeholder string for an image prompt:
       <image>\n{user_text}
   The collator recovers the figure-text span from Qwen3-VL's native visual
   token IDs, so med_seg does not need to add special tokens.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from qwenvl.registry.base import BaseProcessor
from qwenvl.registry.registry import ComponentRegistry

from .losses import clip_xyxy, xywh_to_xyxy


# ── BiomedParse meta-object ontology ───────────────────────────────
# The paper trains an auxiliary classifier over coarse biomedical
# meta-object types. Keep this list stable: the decoder's classifier output
# dimension follows it, and the dataset emits ids into this namespace.
META_OBJECT_NAMES: Tuple[str, ...] = (
    "lung",
    "kidney",
    "heart",
    "brain",
    "eye",
    "vessel",
    "other_organ",
    "tumor",
    "infection",
    "lesion",
    "fluid",
    "other_abnormality",
    "histology_structure",
    "pathological_cells",
    "other",
)
META_OBJECT_TO_ID = {name: i for i, name in enumerate(META_OBJECT_NAMES)}


def _norm_meta_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", " ").replace("_", " ")


def infer_meta_object_id(*values: Any) -> int:
    """Infer a BiomedParse-style meta-object id from loose text fields.

    Official JSON may already carry a meta-object string. Processed datasets
    often only have a coarse text/category/unit name, so we use conservative
    biomedical keyword rules and return ``-100`` when no reliable mapping is
    found. ``-100`` is the CE ignore index used by the decoder.
    """
    text = " ".join(_norm_meta_text(v) for v in values if v is not None)
    if not text:
        return -100

    direct = text.strip()
    if direct in META_OBJECT_TO_ID:
        return META_OBJECT_TO_ID[direct]

    def has_any(*needles: str) -> bool:
        return any(n in text for n in needles)

    # Histology/cell classes first: words like "tumor" can appear in
    # pathology prompts, but the meta-object in BiomedParse is often cell
    # lineage rather than a generic abnormality.
    if has_any(
        "neoplastic cell", "cancer cell", "tumor cell", "tumour cell",
        "inflammatory cell", "lymphocyte", "epithelial cell",
        "connective tissue cell", "pathological cell", "nuclei", "nucleus",
        "cell segmentation", "pannuke",
    ):
        return META_OBJECT_TO_ID["pathological_cells"]
    if has_any("gland", "histology structure", "tissue structure", "h&e", "pathology"):
        return META_OBJECT_TO_ID["histology_structure"]

    if has_any("tumor", "tumour", "cancer", "carcinoma", "melanoma", "glioma", "neoplasm", "lesion ultrasound"):
        return META_OBJECT_TO_ID["tumor"]
    if has_any("covid", "infection", "pneumonia", "opacity"):
        return META_OBJECT_TO_ID["infection"]
    if has_any("edema", "oedema", "effusion", "fluid", "pneumothorax", "embolism"):
        return META_OBJECT_TO_ID["fluid"]
    if has_any("lesion", "nodule", "polyp", "cyst", "abnormal"):
        return META_OBJECT_TO_ID["lesion"]

    if has_any("vessel", "artery", "vein", "aorta", "postcava", "vena cava", "retinal vessel"):
        return META_OBJECT_TO_ID["vessel"]
    if has_any("lung", "pulmonary", "chest x ray", "chest xray", "cxr"):
        return META_OBJECT_TO_ID["lung"]
    if has_any("kidney", "renal"):
        return META_OBJECT_TO_ID["kidney"]
    if has_any("heart", "cardiac", "myocard", "ventricle", "atrium"):
        return META_OBJECT_TO_ID["heart"]
    if has_any("brain", "cerebell", "hippocampus", "ventricles"):
        return META_OBJECT_TO_ID["brain"]
    if has_any("eye", "fundus", "retina", "optic disc", "optic cup", "oct", "macular"):
        return META_OBJECT_TO_ID["eye"]

    if has_any(
        "liver", "spleen", "pancreas", "adrenal", "esophagus", "oesophagus",
        "stomach", "bladder", "duodenum", "colon", "uterus", "prostate",
        "gallbladder", "organ", "abdomen", "abdominal", "fetal head",
    ):
        return META_OBJECT_TO_ID["other_organ"]

    return -100


# ── default system prompt (used everywhere unless overridden via CLI) ──
# This is the canonical instruction shown to the LLM on every batch. It
# is the single source of truth — Qwen3VLMedSegConfig, MedSegDataArgs and
# the CLI default in argument.py all import this constant.
DEFAULT_MED_SEG_SYSTEM_PROMPT = (
    "You are a medical imaging assistant specialized in image understanding "
    "for segmentation.\n"
    "You will be provided with (1) a medical image and (2) a text query that "
    "specifies a target region to segment.\n"
    "Your goal is to produce a detailed, clinically plausible description that "
    "helps delineate the target region precisely.\n"
    "Please describe:\n"
    "    1) The imaging modality/type if it is evident (e.g., CT, MRI, "
    "ultrasound, X-ray, endoscopy, dermoscopy, histopathology). If uncertain, "
    "explicitly say it is unknown.\n"
    "    2) The overall field of view and orientation (e.g., axial/sagittal/"
    "coronal for cross-sectional images, or a general view if not applicable).\n"
    "    3) The main anatomical structures/organs visible, and for each one "
    "provide approximate location using spatial terms. (left/right, superior/"
    "inferior, anterior/posterior, central/peripheral) and relative "
    "relationships between organs.\n"
    "    4) Most importantly, focus on the structure mentioned in the text "
    "query: describe its approximate position, boundaries, shape, size, and "
    "its appearance (intensity/color, texture, homogeneity, and contrast to "
    "surrounding tissues).\n"
    "    Be concise but detailed, use precise spatial language, and emphasize "
    "features helpful for pixel-accurate segmentation."
)


@dataclass
class InstanceAnn:
    box_xyxy: List[float]
    mask: Optional[np.ndarray]
    text: Optional[str]
    category_name: Optional[str]
    mask_path: Optional[str] = None
    meta_object_id: int = -100


def _safe_join(root: Optional[str], path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if not root:
        return path
    return path if os.path.isabs(path) else os.path.join(root, path)


def _load_mask(path: str) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.array(m)
    mx = int(arr.max()) if arr.size > 0 else 0
    if mx <= 10:
        return arr > 0
    return arr > 127


def _finite_list(xs) -> bool:
    try:
        for v in xs:
            if not math.isfinite(float(v)):
                return False
        return True
    except Exception:
        return False


def _collect_text_candidates(ann: Dict[str, Any]) -> List[str]:
    """Pull every text variant from a BiomedParse-style annotation.

    BiomedParseData ships a ``sentences`` list per annotation containing
    1-36 paraphrased descriptions (see microsoft/BiomedParseData on HF).
    Returns a deduped list preserving first-seen order so callers can
    randomly pick one per epoch.
    """
    seen: Dict[str, None] = {}

    for k in ("text", "phrase", "caption", "sentence", "sent"):
        v = ann.get(k)
        if isinstance(v, str) and v.strip():
            seen.setdefault(v.strip(), None)

    sents = ann.get("sentences")
    if isinstance(sents, list):
        for item in sents:
            if isinstance(item, str) and item.strip():
                seen.setdefault(item.strip(), None)
            elif isinstance(item, dict):
                for kk in ("sent", "raw", "text"):
                    val = item.get(kk)
                    if isinstance(val, str) and val.strip():
                        seen.setdefault(val.strip(), None)
                        break  # one per sentence dict
    return list(seen.keys())


def _get_text_from_ann(
    ann: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Return one text variant.

    When ``rng`` is provided, randomly pick from all available variants
    (BiomedParse-style description augmentation — see Zhao et al.,
    Nature Methods 2024). Otherwise return the first variant
    deterministically (eval / inference).
    """
    candidates = _collect_text_candidates(ann)
    if not candidates:
        return None
    if rng is not None:
        return rng.choice(candidates)
    return candidates[0]


def parse_instances(
    anns: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    cat_name_by_id: Optional[Dict[int, str]] = None,
    cat_meta_by_id: Optional[Dict[int, Any]] = None,
    mask_root: Optional[str] = None,
    load_mask_now: bool = False,
    rng: Optional[random.Random] = None,
    unit_name: Optional[str] = None,
) -> List[InstanceAnn]:
    """Convert a list of raw COCO/BiomedParse annotations into InstanceAnn."""
    cat_name_by_id = cat_name_by_id or {}
    cat_meta_by_id = cat_meta_by_id or {}
    out: List[InstanceAnn] = []
    for ann in anns:
        # ---- box ----
        box_xyxy = None
        if "bbox" in ann and len(ann["bbox"]) == 4 and _finite_list(ann["bbox"]):
            box_xyxy = xywh_to_xyxy([float(x) for x in ann["bbox"]])
        elif "box_xyxy" in ann and len(ann["box_xyxy"]) == 4 and _finite_list(ann["box_xyxy"]):
            box_xyxy = [float(x) for x in ann["box_xyxy"]]
        elif "box" in ann and len(ann["box"]) == 4 and _finite_list(ann["box"]):
            box_xyxy = [float(x) for x in ann["box"]]
        if box_xyxy is None:
            continue
        box_xyxy = clip_xyxy(box_xyxy, w=img_w, h=img_h)

        # ---- mask ----
        mask = None
        mask_path = None
        if ann.get("mask_file"):
            mask_path = _safe_join(mask_root, ann["mask_file"])
            if load_mask_now and mask_path and os.path.exists(mask_path):
                try:
                    m = _load_mask(mask_path)
                    if m.sum() > 0:
                        mask = m
                except Exception:
                    mask = None

        # ---- text / category ----
        # When rng is provided (training), each call picks a random variant
        # from the annotation's ``sentences`` list. This is the BiomedParse
        # description-augmentation step (avg ~8 sentences per object type).
        text = _get_text_from_ann(ann, rng=rng)
        cid = ann.get("category_id")
        cname = cat_name_by_id.get(int(cid)) if cid is not None else None
        cmeta = cat_meta_by_id.get(int(cid)) if cid is not None else None
        meta_id = -100
        for k in (
            "meta_object_id", "meta_id", "meta_label_id",
            "meta_object", "meta_object_type", "meta_type",
            "meta_class", "supercategory",
        ):
            if k not in ann:
                continue
            v = ann.get(k)
            if isinstance(v, int):
                meta_id = int(v)
                break
            meta_id = infer_meta_object_id(v)
            if meta_id >= 0:
                break
        if meta_id < 0:
            meta_id = infer_meta_object_id(cmeta, text, cname, unit_name)

        out.append(InstanceAnn(
            box_xyxy=box_xyxy,
            mask=mask,
            text=text,
            category_name=cname,
            mask_path=mask_path,
            meta_object_id=meta_id,
        ))
    return out


def sample_task(rng: random.Random, task_weights: Dict[str, float]) -> str:
    items = list(task_weights.items())
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if r <= acc:
            return k
    return items[-1][0]


@ComponentRegistry.register_processor("med_seg_processor")
class MedSegProcessor(BaseProcessor):
    """Processor for the med_seg modality.

    The chat-template placeholder format is:
        <image>\n{user_text}
    where ``<image>`` itself expands into Qwen3-VL's standard
    ``<|vision_start|><|image_pad|>×N<|vision_end|>`` block. The data
    collator constructs ``user_text_mask`` from these native token IDs.
    """

    def __init__(self, **_):
        # No state — config knobs live on Qwen3VLMedSegConfig and the
        # decoder; we keep the processor stateless to be picklable across
        # DataLoader workers without surprises.
        pass

    @property
    def modality_name(self) -> str:
        return "med_seg"

    def process_input(self, raw_input: Any, **kwargs) -> Dict[str, Any]:
        """Light wrapper — most heavy lifting (image+SAM3-processor) happens
        in the data collator, since SAM3's batching couples image resize
        with per-sample prompt boxes.
        """
        return {"raw": raw_input}

    def build_placeholder(
        self,
        raw_input: Any,
        is_output: bool = False,
        **kwargs,
    ) -> Tuple[str, Optional[Any]]:
        user_text = kwargs.get("text", "visual")
        placeholder = f"<image>\n{str(user_text)}"
        return placeholder, None
