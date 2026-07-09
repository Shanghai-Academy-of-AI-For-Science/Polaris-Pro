"""
RNA projector: 2-layer MLP that maps RNA encoder hidden dim → LLM hidden dim.

This is the canonical implementation used by ModalityRouter.
"""

import torch.nn as nn
from torch import Tensor


class Qwen3VLRNAProjector(nn.Module):
    """2-layer MLP projector: RNA encoder hidden dim → LLM hidden dim."""

    def __init__(self, config, llm_hidden_size: int):
        super().__init__()
        self.config = config

        rna_hidden_size = config.rna_encoder_hidden_size
        projector_hidden_size = config.rna_projector_hidden_size

        self.norm = nn.LayerNorm(rna_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(rna_hidden_size, projector_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(projector_hidden_size, llm_hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_fc1.weight)
        nn.init.zeros_(self.linear_fc1.bias)
        nn.init.xavier_uniform_(self.linear_fc2.weight)
        nn.init.zeros_(self.linear_fc2.bias)

    def forward(self, rna_features: Tensor) -> Tensor:
        """
        Args:
            rna_features: [*, rna_hidden_size]
        Returns:
            [*, llm_hidden_size]
        """
        x = self.norm(rna_features)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x
