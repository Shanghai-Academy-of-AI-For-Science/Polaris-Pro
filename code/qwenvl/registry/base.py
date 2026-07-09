"""
Abstract base classes for multimodal components.

Every modality (RNA, protein, small molecule, etc.) implements these
interfaces so the framework can route data through the correct
encoder -> projector -> LLM -> decoder pipeline without hard-coding.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module, ABC):
    """Encodes raw modality input into a fixed set of latent vectors.

    Subclasses must implement ``encode`` and expose ``output_dim`` so
    the projector layer can be constructed automatically.
    """

    @abstractmethod
    def encode(
        self,
        inputs: Dict[str, Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the encoder on preprocessed inputs.

        Args:
            inputs: Dict produced by the matching ``BaseProcessor``.
                    Typical keys: ``input_ids``, ``attention_mask``, etc.

        Returns:
            latent:      [B, K, D_enc]  – K latent vectors per sample.
            latent_mask: [B, K] or None – 1 = valid, 0 = padding.
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of each latent vector (D_enc)."""
        ...

    @property
    def num_latent_tokens(self) -> Optional[int]:
        """Fixed number of latent tokens K, or None if variable-length."""
        return None


class BaseProjector(nn.Module, ABC):
    """Projects encoder latent vectors into the LLM hidden space."""

    @abstractmethod
    def project(self, features: Tensor) -> Tensor:
        """Map [*, D_enc] -> [*, D_llm]."""
        ...


class BaseDecoder(nn.Module, ABC):
    """Task-specific decoder head applied to LLM hidden states.

    Examples: RNA sequence decoder, classification head, regression head,
    structure prediction head.
    """

    # ``decode`` / ``compute_loss`` are training/task-specific and optional in
    # this inference build — each decoder implements only the inference entry
    # point it needs (e.g. ``generate_smiles``, ``predict_from_hidden``,
    # ``logits_to_vocab_space``). ``output_size`` stays required.
    def decode(self, hidden_states: Tensor, labels: Optional[Tensor] = None, **kwargs) -> Tensor:
        raise NotImplementedError

    def compute_loss(self, logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Number of output classes / vocab size for this head."""
        ...


class BaseProcessor(ABC):
    """Converts raw data into model-ready tensors and chat-template text.

    Each modality has its own processor that knows how to:
    1. Read raw input (sequence string, file path, numpy array, …).
    2. Produce tokenized / tensorized inputs for the encoder.
    3. Generate the placeholder string for chat-template insertion.
    """

    @abstractmethod
    def process_input(
        self,
        raw_input: Any,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Convert raw input to encoder-ready tensors.

        Returns:
            Dict with at least ``input_ids`` and ``attention_mask`` (or
            modality-appropriate equivalents like ``pixel_values``).
        """
        ...

    @abstractmethod
    def build_placeholder(
        self,
        raw_input: Any,
        is_output: bool = False,
        **kwargs,
    ) -> Tuple[str, Optional[Any]]:
        """Return the chat-template placeholder string for this input.

        Args:
            raw_input: The raw data (sequence, path, etc.).
            is_output: True when the modality appears in the assistant
                       response (e.g. RNA generation).

        Returns:
            placeholder: String to insert into the chat template.
            metadata:    Any side-channel data (e.g. real_token_ids for
                         teacher-forcing RNA output).
        """
        ...

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Canonical name of the modality this processor handles."""
        ...
