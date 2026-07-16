"""
Registry system for pluggable multimodal components.

Provides base classes, registration decorators, and a SpecialTokenManager
for building extensible encoder/projector/decoder/processor pipelines.
"""

from .base import BaseEncoder, BaseProjector, BaseDecoder, BaseProcessor
from .registry import ComponentRegistry
from .modality_spec import ModalitySpec
from .token_manager import SpecialTokenManager, MODALITY_TOKEN_DEFS, BIO_SEQ_OUTPUT_PAD
from .modality_router import ModalityRouter

__all__ = [
    "BaseEncoder",
    "BaseProjector",
    "BaseDecoder",
    "BaseProcessor",
    "ComponentRegistry",
    "ModalitySpec",
    "SpecialTokenManager",
    "MODALITY_TOKEN_DEFS",
    "BIO_SEQ_OUTPUT_PAD",
    "ModalityRouter",
]
