"""
ModalitySpec for image and video modalities.

These use the built-in Qwen3-VL VisionModel and do not need a separate
encoder registration — they are handled natively by the model.
"""

from mkb.registry.modality_spec import ModalitySpec

VISION_IMAGE_SPEC = ModalitySpec(
    name="image",
    input_type="image",
    start_token="<|vision_start|>",
    end_token="<|vision_end|>",
    pad_token="<|image_pad|>",
    encoder_cls=None,
    projector_cls=None,
    decoder_cls=None,
    processor_cls=None,
    uses_vision_pathway=True,
    placeholder_tag="<image>",
    legacy_token_mode=True,
    legacy_pad_token_id=151655,
)

VISION_VIDEO_SPEC = ModalitySpec(
    name="video",
    input_type="image",
    start_token="<|vision_start|>",
    end_token="<|vision_end|>",
    pad_token="<|video_pad|>",
    encoder_cls=None,
    projector_cls=None,
    decoder_cls=None,
    processor_cls=None,
    uses_vision_pathway=True,
    placeholder_tag="<video>",
    legacy_token_mode=True,
    legacy_pad_token_id=151656,
)
