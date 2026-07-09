"""
Molecular encoder: GNN backbone + Perceiver Resampler.

Wraps the pretrained GNN from mol/graph_NN.py and compresses variable-length
graph node embeddings into a fixed number of latent tokens via cross-attention.

Forward signature is compatible with ModalityRouter:
    (input_ids, attention_mask, **kwargs) -> (latent [B,K,D], mask [B,K])

Graph data (edge_index, edge_attr, edge_index_all, batch_idx) is passed
through **kwargs. When kwargs are empty (dummy forward), returns zero tensors
connected to trainable parameters for gradient sync.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .graph_NN import GNN


class _LatentResampler(nn.Module):
    """Perceiver-style cross-attention: variable-length nodes -> K latent tokens."""

    def __init__(self, dim: int, num_latent_tokens: int = 16, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(1, num_latent_tokens, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        ffn_dim = dim * 4
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, encoder_out: Tensor, key_padding_mask: Tensor | None = None):
        """
        Args:
            encoder_out:     [B, max_nodes, D]
            key_padding_mask: [B, max_nodes]  True = ignore (padding)
        Returns:
            latent: [B, K, D]
        """
        B = encoder_out.shape[0]
        q = self.norm_q(self.latent_queries.expand(B, -1, -1))
        kv = self.norm_kv(encoder_out)
        h, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        h = q + h
        h = h + self.ffn(h)
        return h


class MolEncoder(nn.Module):
    """GNN + Perceiver Resampler -> fixed K latent tokens.

    The GNN processes molecular graphs with variable numbers of atoms.
    The resampler compresses the per-atom embeddings into K fixed-length
    latent tokens suitable for scattering into the LLM embedding space.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.mol_encoder_hidden_size
        self.num_latent_tokens = config.num_latent_tokens
        self._gnn_frozen_cached: bool | None = None

        self.gnn = GNN(
            num_layer=config.num_gnn_layers,
            emb_dim=config.mol_encoder_hidden_size,
            drop_ratio=config.gnn_drop_ratio,
            output_type="last",
        )

        self.resampler = _LatentResampler(
            dim=config.mol_encoder_hidden_size,
            num_latent_tokens=config.num_latent_tokens,
            heads=config.num_resampler_heads,
            dropout=getattr(config, "dropout", 0.1),
        )

        # Initialize resampler weights
        self.resampler.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _is_gnn_frozen(self) -> bool:
        """Check if GNN has no trainable params (cached for performance)."""
        if self._gnn_frozen_cached is None:
            self._gnn_frozen_cached = not any(
                p.requires_grad for p in self.gnn.parameters()
            )
        return self._gnn_frozen_cached

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs,
    ):
        """
        Args:
            input_ids:      Graph node features [total_nodes, 5] (real forward)
                            or [B, dummy_len] (dummy forward from ModalityRouter)
            attention_mask: Dummy (not used for graph encoder)
            **kwargs:       Graph-specific tensors:
                edge_index:     [2, num_edges]
                edge_attr:      [num_edges, 3]
                edge_index_all: [2, num_fc_edges]
                batch_idx:      [total_nodes] — molecule assignment per node

        Returns:
            latent:      [B, K, D]  — compressed latent tokens
            latent_mask: [B, K]     — all-ones mask (fixed length)
        """
        edge_index = kwargs.get("edge_index")

        if edge_index is None:
            # Fallback when no graph is provided.
            # Must produce a tensor connected to ALL trainable params.
            B = input_ids.shape[0]
            K = self.num_latent_tokens
            device = input_ids.device
            dtype = next(self.resampler.parameters()).dtype
            # Use latent_queries as KV so cross-attention weights also get gradients
            dummy_kv = self.resampler.latent_queries[:, :1, :].expand(B, -1, -1)
            latent = self.resampler(dummy_kv)  # [B, K, D] — all resampler params in graph

            # If GNN has trainable params (freeze_policy may have unfrozen them
            # after register_modality set freeze_mol_gnn=True), connect them too.
            gnn_trainable = [p for p in self.gnn.parameters() if p.requires_grad]
            if gnn_trainable:
                gnn_dummy = sum(p.sum() for p in gnn_trainable) * 0
                latent = latent + gnn_dummy

            latent_mask = torch.ones(B, K, device=device, dtype=torch.long)
            return latent, latent_mask

        # Real graph forward
        edge_attr = kwargs["edge_attr"]
        edge_index_all = kwargs["edge_index_all"]
        batch_idx = kwargs["batch_idx"]

        # GNN forward: [total_nodes, 5] -> [total_nodes, D]
        # Memory optimization: skip autograd graph for frozen GNN
        if self._is_gnn_frozen():
            with torch.no_grad():
                node_embeds = self.gnn(
                    input_ids, edge_index, edge_index_all, edge_attr, batch_idx
                )
                if isinstance(node_embeds, list):
                    node_embeds = node_embeds[-1]
                node_embeds = node_embeds.detach()
        else:
            node_embeds = self.gnn(
                input_ids, edge_index, edge_index_all, edge_attr, batch_idx
            )
            if isinstance(node_embeds, list):
                node_embeds = node_embeds[-1]

        # Group by molecule + pad -> [B, max_nodes, D]
        # Uses PyG's vectorized to_dense_batch (avoids Python for-loop)
        from torch_geometric.utils import to_dense_batch
        padded, real_mask = to_dense_batch(node_embeds, batch_idx)
        # to_dense_batch returns real_mask where True = real node
        # MultiheadAttention key_padding_mask expects True = ignore (padding)
        pad_mask = ~real_mask
        num_mols = padded.shape[0]

        # Resampler: [B, max_nodes, D] -> [B, K, D]
        latent = self.resampler(padded, key_padding_mask=pad_mask)
        K = latent.shape[1]
        latent_mask = torch.ones(num_mols, K, device=latent.device, dtype=torch.long)
        return latent, latent_mask
