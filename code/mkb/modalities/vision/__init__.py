"""
Vision modality (image / video).

The Qwen3-VL vision encoder is treated as a built-in; the ModalitySpec
here is mainly for config completeness and future wrapping.
"""

from .config import VISION_IMAGE_SPEC, VISION_VIDEO_SPEC

__all__ = ["VISION_IMAGE_SPEC", "VISION_VIDEO_SPEC"]
