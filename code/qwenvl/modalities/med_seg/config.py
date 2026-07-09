"""Config for the med_seg modality (SAM3 medical-image segmentation).

Held under ``main_config.med_seg_config`` and passed to ``register_modality``
when the user enables med-seg via ``--sam3_model_path``.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .processor import DEFAULT_MED_SEG_SYSTEM_PROMPT


@dataclass
class Qwen3VLMedSegConfig:
    """Configuration for the SAM3 med-seg decoder.

    Notes:
    - ``sam3_text_dim`` must match the SAM3 checkpoint's text-embedding dim
      (256 for facebook/sam3). The proj layer maps Qwen LLM hidden → this.
    - ``mask_h`` / ``mask_w`` come from ``sam3_processor.image_processor.mask_size``
      and are filled in at runtime by the data collator.
    - The collator builds ``user_text_mask`` from Qwen3-VL's native visual
      tokens. It covers the *entire* figure-text span (image block + text
      query) without adding med_seg-specific special tokens.
    """

    sam3_model_path: Optional[str] = None
    sam3_text_dim: int = 256
    freeze_sam3: bool = False
    freeze_sam3_vision: bool = False  # finer-grained: keep proj+decoder, freeze backbone

    # Loss weights (matched to sam_uni defaults)
    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    # ── Mask-overlap matching cost (default: off for backward-compat) ──
    # Adds cost_dice*dice + cost_mask*BCE on a coarse grid to the Hungarian
    # cost. Stabilises single-GT matching (box+class alone lets the winning
    # query flip every step → cls loss swings 1e-7<->0.2, mask supervision
    # keeps re-targeting). Mask2Former defaults are cost_mask=5, cost_dice=5.
    cost_mask: float = 0.0
    cost_dice: float = 0.0
    mask_cost_res: int = 64
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    loss_w_cls: float = 1.0
    loss_w_bbox_l1: float = 5.0
    loss_w_bbox_giou: float = 2.0
    loss_w_mask_focal: float = 2.0
    loss_w_mask_dice: float = 2.0
    # ── Auxiliary mask losses (defaults: off for backward-compat) ──
    # High-resolution mask loss: dice/BCE against the full-res GT
    # (``masks_fullres_padded`` from the collator). Requires the collator
    # to emit a non-empty fullres tensor; otherwise this branch is a
    # no-op. Pred is bilinear-upsampled from SAM3's native 288 to the
    # fullres GT size — gradient flows back through the interp.
    loss_w_mask_dice_high: float = 0.0
    # Semantic-segmentation loss: dice on
    # ``max_q ( sigmoid(score_q) * sigmoid(mask_logit_q) )`` vs the
    # union of all valid GT masks per image. Helps when targets span
    # multiple disconnected regions (PanNuke, DRIVE, GlaS, OCT-CME,
    # NeoPolyp), where single-query argmax under-recalls.
    loss_w_mask_semantic: float = 0.0
    # BiomedParse-style auxiliary meta-object classification. The dataset
    # emits a coarse object id (lung/kidney/.../tumor/cell/etc.) when it can
    # infer one; unknown labels use -100 and are ignored by CE.
    loss_w_meta_ce: float = 0.0
    meta_num_classes: int = 15
    # Two-layer projection: replaces the single Linear(D_qwen → 256)
    # with LayerNorm → Linear → GELU → Linear. Cheap (~5M params) but
    # gives the Qwen→SAM3 bridge enough capacity for L3 fine-grained
    # tasks. Set False to keep the legacy single-layer projection (must
    # also be False when resuming from a single-layer checkpoint).
    proj_mlp: bool = True
    proj_hidden_mult: int = 2  # hidden size = sam3_text_dim * mult

    # System / chat-template settings — single source of truth lives in
    # qwenvl/modalities/med_seg/processor.py::DEFAULT_MED_SEG_SYSTEM_PROMPT.
    system_prompt: str = field(default_factory=lambda: DEFAULT_MED_SEG_SYSTEM_PROMPT)
    use_query_special_tokens: bool = False

    # Mask output resolution (filled in at runtime from the SAM3 processor).
    mask_h: int = 288
    mask_w: int = 288

    upsample_mask_loss: bool = False
    mask_loss_target_res: Optional[int] = None

    med_query_start_id: Optional[int] = None
    med_query_end_id: Optional[int] = None
