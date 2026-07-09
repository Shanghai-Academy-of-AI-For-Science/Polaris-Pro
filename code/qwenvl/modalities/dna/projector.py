"""
DNA projector: 2-layer MLP that maps DNA encoder hidden dim → LLM hidden dim.

Same structure as ``Qwen3VLRNAProjector`` but reads DNA-specific config
fields and registers as an independent module under the ``dna`` slot in
the ModalityRouter.
"""

import torch.nn as nn
from torch import Tensor


class Qwen3VLDNAProjector(nn.Module):
    """2-layer MLP projector: DNA encoder hidden dim → LLM hidden dim."""

    def __init__(self, config, llm_hidden_size: int):
        super().__init__()
        self.config = config

        dna_hidden_size = config.dna_encoder_hidden_size
        projector_hidden_size = config.dna_projector_hidden_size

        self.norm = nn.LayerNorm(dna_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(dna_hidden_size, projector_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(projector_hidden_size, llm_hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_fc1.weight)
        nn.init.zeros_(self.linear_fc1.bias)
        nn.init.xavier_uniform_(self.linear_fc2.weight)
        nn.init.zeros_(self.linear_fc2.bias)

    def forward(self, dna_features: Tensor) -> Tensor:
        """
        Args:
            dna_features: [*, dna_hidden_size]
        Returns:
            [*, llm_hidden_size]
        """
        x = self.norm(dna_features)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x
