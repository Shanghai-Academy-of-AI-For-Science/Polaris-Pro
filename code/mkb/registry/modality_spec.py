"""
ModalitySpec: declarative description of a single modality.

One ModalitySpec is created per active modality (RNA, protein, etc.)
and tells the framework which encoder, projector, decoder, processor,
and special tokens to use.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModalitySpec:
    """Full specification of a modality and its components."""

    # ---- identity ----
    name: str  # "rna", "dna", "protein", "mol", "image", …

    # ---- input type hint (for data pipeline routing) ----
    input_type: str = "sequence"  # "sequence", "image", "matrix", "tensor3d", "file"

    # ---- special tokens ----
    start_token: str = ""  # e.g. "<|rna_start|>"
    end_token: str = ""    # e.g. "<|rna_end|>"
    pad_token: str = ""    # e.g. "<|rna_pad|>"

    # ---- component class names (looked up in ComponentRegistry) ----
    encoder_cls: Optional[str] = None   # e.g. "rna_convformer"
    projector_cls: Optional[str] = None # e.g. "rna_mlp_projector"
    decoder_cls: Optional[str] = None   # e.g. "rna_lm_decoder"
    processor_cls: Optional[str] = None # e.g. "rna_char_processor"

    # ---- component configs (passed to constructors) ----
    encoder_config: Dict[str, Any] = field(default_factory=dict)
    projector_config: Dict[str, Any] = field(default_factory=dict)
    decoder_config: Dict[str, Any] = field(default_factory=dict)
    processor_config: Dict[str, Any] = field(default_factory=dict)

    # ---- routing hints ----
    uses_vision_pathway: bool = False
    # When True the modality reuses the Qwen vision encoder (image / video
    # pathway).  When False the modality has its own dedicated encoder.

    placeholder_tag: str = ""
    # The tag used in conversation text, e.g. "<rna>", "<protein>", "<image>".

    # ---- legacy compatibility ----
    legacy_token_mode: bool = False
    # When True, reuse Qwen vision tokens (<|image_pad|> / <|video_pad|>)
    # instead of dedicated modality tokens.  This preserves backward
    # compatibility with existing checkpoints.

    legacy_pad_token_id: Optional[int] = None
    # The Qwen token ID to reuse in legacy mode (e.g. 151655 for image_pad).

    def __post_init__(self):
        if not self.placeholder_tag and self.name:
            self.placeholder_tag = f"<{self.name}>"
