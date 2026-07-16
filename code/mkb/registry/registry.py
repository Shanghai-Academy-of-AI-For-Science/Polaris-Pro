"""
ComponentRegistry: central registration point for modality components.

Components are registered by name via class decorators and can later be
instantiated from a ModalitySpec without hard-coding any imports in the
model or training code.
"""

import logging
from typing import Any, Dict, Optional, Type

import torch.nn as nn

from .base import BaseDecoder, BaseEncoder, BaseProcessor, BaseProjector
from .modality_spec import ModalitySpec

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Singleton registry mapping string names to component classes."""

    _encoders: Dict[str, Type[BaseEncoder]] = {}
    _projectors: Dict[str, Type[BaseProjector]] = {}
    _decoders: Dict[str, Type[BaseDecoder]] = {}
    _processors: Dict[str, Type[BaseProcessor]] = {}

    # ---- registration decorators ----

    @classmethod
    def register_encoder(cls, name: str):
        """Decorator: ``@ComponentRegistry.register_encoder("rna_convformer")``."""
        def wrapper(klass):
            if name in cls._encoders:
                logger.warning(f"Encoder '{name}' already registered – overwriting.")
            cls._encoders[name] = klass
            return klass
        return wrapper

    @classmethod
    def register_projector(cls, name: str):
        def wrapper(klass):
            if name in cls._projectors:
                logger.warning(f"Projector '{name}' already registered – overwriting.")
            cls._projectors[name] = klass
            return klass
        return wrapper

    @classmethod
    def register_decoder(cls, name: str):
        def wrapper(klass):
            if name in cls._decoders:
                logger.warning(f"Decoder '{name}' already registered – overwriting.")
            cls._decoders[name] = klass
            return klass
        return wrapper

    @classmethod
    def register_processor(cls, name: str):
        def wrapper(klass):
            if name in cls._processors:
                logger.warning(f"Processor '{name}' already registered – overwriting.")
            cls._processors[name] = klass
            return klass
        return wrapper

    # ---- lookup helpers ----

    @classmethod
    def get_encoder_cls(cls, name: str) -> Type[BaseEncoder]:
        if name not in cls._encoders:
            raise KeyError(
                f"Encoder '{name}' not registered. "
                f"Available: {list(cls._encoders.keys())}"
            )
        return cls._encoders[name]

    @classmethod
    def get_projector_cls(cls, name: str) -> Type[BaseProjector]:
        if name not in cls._projectors:
            raise KeyError(
                f"Projector '{name}' not registered. "
                f"Available: {list(cls._projectors.keys())}"
            )
        return cls._projectors[name]

    @classmethod
    def get_decoder_cls(cls, name: str) -> Type[BaseDecoder]:
        if name not in cls._decoders:
            raise KeyError(
                f"Decoder '{name}' not registered. "
                f"Available: {list(cls._decoders.keys())}"
            )
        return cls._decoders[name]

    @classmethod
    def get_processor_cls(cls, name: str) -> Type[BaseProcessor]:
        if name not in cls._processors:
            raise KeyError(
                f"Processor '{name}' not registered. "
                f"Available: {list(cls._processors.keys())}"
            )
        return cls._processors[name]

    # ---- high-level builder ----

    @classmethod
    def build_modality_components(
        cls,
        spec: ModalitySpec,
        llm_hidden_size: int,
        **extra_kwargs,
    ) -> Dict[str, Any]:
        """Instantiate all components described by a ModalitySpec.

        Returns a dict with keys ``encoder``, ``projector``, ``decoder``,
        ``processor`` (any of which may be *None* if the spec omits them).
        """
        result: Dict[str, Any] = {
            "encoder": None,
            "projector": None,
            "decoder": None,
            "processor": None,
        }

        if spec.encoder_cls:
            enc_cls = cls.get_encoder_cls(spec.encoder_cls)
            result["encoder"] = enc_cls(**spec.encoder_config)

        if spec.projector_cls:
            proj_cls = cls.get_projector_cls(spec.projector_cls)
            proj_kwargs = {**spec.projector_config, "llm_hidden_size": llm_hidden_size}
            result["projector"] = proj_cls(**proj_kwargs)

        if spec.decoder_cls:
            dec_cls = cls.get_decoder_cls(spec.decoder_cls)
            dec_kwargs = {**spec.decoder_config, "llm_hidden_size": llm_hidden_size}
            result["decoder"] = dec_cls(**dec_kwargs)

        if spec.processor_cls:
            proc_cls = cls.get_processor_cls(spec.processor_cls)
            result["processor"] = proc_cls(**spec.processor_config)

        return result

    # ---- introspection ----

    @classmethod
    def list_encoders(cls):
        return list(cls._encoders.keys())

    @classmethod
    def list_projectors(cls):
        return list(cls._projectors.keys())

    @classmethod
    def list_decoders(cls):
        return list(cls._decoders.keys())

    @classmethod
    def list_processors(cls):
        return list(cls._processors.keys())

    @classmethod
    def summary(cls) -> str:
        lines = [
            "ComponentRegistry:",
            f"  Encoders   : {cls.list_encoders()}",
            f"  Projectors : {cls.list_projectors()}",
            f"  Decoders   : {cls.list_decoders()}",
            f"  Processors : {cls.list_processors()}",
        ]
        return "\n".join(lines)
