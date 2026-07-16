"""
Molecular projector: 2-layer MLP that maps mol encoder hidden dim -> LLM hidden dim.

Same architecture as RNA/protein projectors.
"""

import torch.nn as nn
from torch import Tensor


class MolProjector(nn.Module):
    """2-layer MLP projector: mol encoder hidden dim -> LLM hidden dim."""

    def __init__(self, config, llm_hidden_size: int):
        super().__init__()
        self.config = config

        mol_hidden_size = config.mol_encoder_hidden_size
        projector_hidden_size = config.mol_projector_hidden_size

        self.norm = nn.LayerNorm(mol_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(mol_hidden_size, projector_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(projector_hidden_size, llm_hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_fc1.weight)
        nn.init.zeros_(self.linear_fc1.bias)
        nn.init.xavier_uniform_(self.linear_fc2.weight)
        nn.init.zeros_(self.linear_fc2.bias)

    def forward(self, mol_features: Tensor) -> Tensor:
        """
        Args:
            mol_features: [*, mol_hidden_size]
        Returns:
            [*, llm_hidden_size]
        """
        x = self.norm(mol_features)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x
