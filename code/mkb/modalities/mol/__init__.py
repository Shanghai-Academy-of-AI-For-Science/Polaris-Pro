"""
Molecular (Mol) modality package.

Provides GNN-based encoder, projector, and processor for molecular graphs.
All components are self-contained — the model discovers and registers them
via the ``register_modality()`` function below.
"""

import logging

from .encoder import MolEncoder
from .projector import MolProjector
from .processor import smiles_to_graph

logger = logging.getLogger(__name__)

MODALITY_CONFIG_KEY = "mol_config"

TOKEN_DEFS = {
    "mol": {
        "start": "<|mol_start|>",
        "end": "<|mol_end|>",
        "pad": "<|mol_pad|>",
    },
}

__all__ = [
    "MolEncoder",
    "MolProjector",
    "smiles_to_graph",
    "register_modality",
    "MODALITY_CONFIG_KEY",
    "TOKEN_DEFS",
]

def register_modality(router, config, llm_hidden_size: int):
    """Register Mol modality with the ModalityRouter.

    This is the standard entry point called by the model during __init__.
    Each modality package must provide this function with the same signature:
        register_modality(router, config, llm_hidden_size) -> None

    Args:
        router: ModalityRouter instance.
        config: Modality-specific config (Qwen3VLMolConfig).
        llm_hidden_size: Hidden size of the LLM backbone.
    """
    encoder = MolEncoder(config)
    projector = MolProjector(config, llm_hidden_size)

    # NOTE: Pretrained GNN weights are loaded by the training entry AFTER
    # from_pretrained() completes (meta tensors are materialized by then).
    # Loading here would be a no-op on meta device or break dtype consistency.

    # Freeze GNN backbone if configured
    freeze_gnn = getattr(config, "freeze_mol_gnn", False)
    if freeze_gnn:
        for p in encoder.gnn.parameters():
            p.requires_grad = False
        logger.info("GNN backbone frozen, only training resampler + projector")

    from .decoder import MolARDecoder, MOL_VOCAB_SIZE
    mol_vocab_size = getattr(config, "mol_vocab_size", MOL_VOCAB_SIZE)
    decoder = MolARDecoder(
        llm_hidden_size=llm_hidden_size,
        decoder_hidden_size=getattr(config, "mol_decoder_hidden_size", 768),
        num_layers=getattr(config, "mol_decoder_num_layers", 6),
        num_heads=getattr(config, "mol_decoder_num_heads", 12),
        mol_vocab_size=mol_vocab_size,
        max_seq_length=getattr(config, "mol_decoder_max_seq_length", 512),
        dropout=getattr(config, "dropout", 0.1),
    )
    n_decoder_params = sum(p.numel() for p in decoder.parameters())
    logger.info(
        f"Mol decoder: MolARDecoder "
        f"(hidden={getattr(config, 'mol_decoder_hidden_size', 768)}, "
        f"layers={getattr(config, 'mol_decoder_num_layers', 6)}, "
        f"heads={getattr(config, 'mol_decoder_num_heads', 12)}, "
        f"vocab={mol_vocab_size}, "
        f"params={n_decoder_params/1e6:.1f}M)"
    )

    router.register_modality(
        "mol",
        encoder=encoder,
        projector=projector,
        decoder=decoder,
        is_image_like=True,
    )
    logger.info(
        f"Mol modality registered: gnn_layers={config.num_gnn_layers}, "
        f"encoder_hidden={config.mol_encoder_hidden_size}, "
        f"latent_tokens={config.num_latent_tokens}, "
        f"freeze_gnn={freeze_gnn}, decoder_vocab={mol_vocab_size}, "
        f"llm_hidden={llm_hidden_size}"
    )
