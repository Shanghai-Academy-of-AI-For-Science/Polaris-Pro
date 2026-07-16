"""Helpers for running SAM3 at a non-default vision input size.

Why this exists
---------------
The bundled facebook/sam3 config hard-codes:
  - vision_config.image_size              = 1008
  - vision_config.backbone_feature_sizes  = [[288,288], [144,144], [72,72]]
  - image_processor.size                  = 1008 (longest_edge / shortest_edge)
  - image_processor.mask_size             = {height: 288, width: 288}

For ultra-fine targets (e.g. DRIVE 1-3 px retinal vessels) the 72×72 ViT
grid quantizes the structure away — at 14 px/token the vessel signal is
diluted to 0.5%-5% of a token's content. Bumping image_size to 2016 gives
a 144×144 grid (7 px/token) and a 576×576 mask output, restoring more usable
signal for these targets.

The four fields above must move TOGETHER. This helper does that in one
call so they stay in lock-step — a mismatch would silently degrade Dice: the
ViT's RoPE adapts to any size but the mask decoder's spatial assumptions
are baked in via FPN feature sizes.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _validate_sam3_image_size(
    image_size: Optional[int],
    *,
    patch_size: int = 14,
    pos_embed_tile_grid: int = 24,
) -> Tuple[Optional[int], Optional[int]]:
    """Validate SAM3 resize dimensions and return (image_size, grid).

    SAM3's learned position embedding is tiled from a 24x24 base grid. A
    runtime grid that is not an integer multiple of 24 creates a partial
    final tile with wrapped position codes. Keep sizes at 14 * 24 * k = 336k.
    """
    if image_size is None:
        return None, None

    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError(f"image_size must be > 0, got {image_size}")

    clean_step = patch_size * pos_embed_tile_grid
    lower = max(clean_step, (image_size // clean_step) * clean_step)
    upper = math.ceil(image_size / clean_step) * clean_step
    if lower == upper:
        hint = f"{upper}"
    else:
        hint = f"{lower} or {upper}"

    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size ({image_size}) must be a multiple of patch_size "
            f"({patch_size}) and should be a clean SAM3 tile size "
            f"({clean_step} * k). Suggested clean size: {hint}."
        )

    grid = image_size // patch_size
    if grid % pos_embed_tile_grid != 0:
        raise ValueError(
            f"image_size ({image_size}) gives ViT grid {grid}, but SAM3's "
            f"learned position embedding tiles cleanly only when grid is a "
            f"multiple of {pos_embed_tile_grid}. Use image_size={hint} "
            f"({clean_step} * k)."
        )

    return image_size, grid


def patch_sam3_for_image_size(
    sam3_config: Any,
    image_size: Optional[int],
    *,
    patch_size: int = 14,
    mask_upsample_factor: int = 4,
) -> Optional[int]:
    """Patch a Sam3Config in-place so the model accepts a new image_size.

    Must be called BEFORE ``Sam3Model.from_pretrained(..., config=sam3_config)``.

    Args:
        sam3_config: A Sam3Config (top-level), Sam3VideoConfig, or anything
            that exposes ``.detector_config.vision_config`` or ``.vision_config``.
        image_size: New input edge length. None = no-op (returns None).
            Must be a clean SAM3 tile size: 14 * 24 * k = 336k.
        patch_size: ViT patch size (14 for facebook/sam3).
        mask_upsample_factor: FPN upsample ratio at the highest level (4 for
            facebook/sam3 — 72→288 in the default, scaled to grid×4 here).

    Returns:
        The new mask output edge length (e.g. 576 for image_size=2016), or
        None when image_size is None. The collator should resize GT masks
        to (mask_size, mask_size).
    """
    if image_size is None:
        return None

    image_size, grid = _validate_sam3_image_size(
        image_size, patch_size=patch_size,
    )

    # Walk the config tree to find vision_config. Sam3Config (image-only)
    # lives at .detector_config.vision_config; Sam3VideoConfig wraps it
    # similarly. Single-level configs may expose .vision_config directly.
    vision_config = None
    for path in ("detector_config", None):
        node = sam3_config if path is None else getattr(sam3_config, path, None)
        if node is None:
            continue
        if hasattr(node, "vision_config"):
            vision_config = node.vision_config
            break
        if hasattr(node, "image_size"):
            # Already at a vision config
            vision_config = node
            break
    if vision_config is None:
        raise RuntimeError(
            "Could not locate vision_config under Sam3Config tree. "
            "Sam3 config layout may have changed — please update "
            "patch_sam3_for_image_size."
        )

    fpn_high = grid * mask_upsample_factor
    fpn_mid = grid * (mask_upsample_factor // 2)
    fpn_low = grid
    new_feature_sizes = [
        [fpn_high, fpn_high],
        [fpn_mid, fpn_mid],
        [fpn_low, fpn_low],
    ]

    old_image_size = getattr(vision_config, "image_size", None)
    old_feature_sizes = getattr(vision_config, "backbone_feature_sizes", None)

    # Sam3VisionConfig provides a setter that propagates to backbone_config.
    vision_config.image_size = image_size
    vision_config.backbone_feature_sizes = new_feature_sizes

    logger.info(
        f"[sam3_resize] image_size: {old_image_size} → {image_size} "
        f"(grid {grid}×{grid}, mask {fpn_high}×{fpn_high})"
    )
    logger.info(
        f"[sam3_resize] backbone_feature_sizes: {old_feature_sizes} → {new_feature_sizes}"
    )

    return fpn_high


def patch_sam3_processor_for_image_size(
    processor: Any,
    image_size: Optional[int],
    *,
    patch_size: int = 14,
    mask_upsample_factor: int = 4,
) -> Optional[int]:
    """Patch a Sam3Processor's image_processor in-place to match a new size.

    Must be kept in lock-step with ``patch_sam3_for_image_size``. The
    image_processor controls (a) what spatial size the input is resized to
    before SAM3 sees it, and (b) what spatial size GT masks are resized to.

    Args:
        processor: A Sam3Processor instance.
        image_size: New edge length. None = no-op. Must be 336 * k.
        patch_size: ViT patch size (14 for facebook/sam3).
        mask_upsample_factor: FPN top-level upsample ratio (4 for facebook/sam3).

    Returns:
        The new mask edge length (matches patch_sam3_for_image_size).
    """
    if image_size is None:
        return None

    image_size, grid = _validate_sam3_image_size(
        image_size, patch_size=patch_size,
    )
    mask_size = grid * mask_upsample_factor

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Sam3Processor has no .image_processor attribute")

    # ``size`` controls input resize. Sam3 fast image processor accepts
    # either a dict ({"longest_edge": ..., "shortest_edge": ...}) or a
    # SizeDict — assigning a plain dict works in both cases.
    image_processor.size = {"longest_edge": image_size, "shortest_edge": image_size}

    # ``mask_size`` controls GT mask output size (also drives the schema
    # the model is expected to predict at the highest FPN level).
    image_processor.mask_size = {"height": mask_size, "width": mask_size}

    logger.info(
        f"[sam3_resize] processor.size → {image_processor.size}, "
        f"processor.mask_size → {image_processor.mask_size}"
    )
    return mask_size
