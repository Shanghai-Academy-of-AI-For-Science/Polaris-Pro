"""
DNA modality package — encoder + projector for DNA classification / regression.

Produces a separate sub-tree under ``model.modality_router.encoders.dna``. The
DNA char tokenizer keeps T as its own char (vocab id 3) rather than normalizing
to U.
"""

import logging

from .encoder import DNAConvFormer, Qwen3VLDNAEncoder
from .projector import Qwen3VLDNAProjector
from .processor import DNACharProcessor, DNACharTokenizer

logger = logging.getLogger(__name__)

MODALITY_CONFIG_KEY = "dna_config"

# Same token strings as token_manager.MODALITY_TOKEN_DEFS["dna"]; declared
# here so per-package token discovery still finds them when scanning
# ``mkb.modalities.*``.
TOKEN_DEFS = {
    "dna": {
        "start": "<|dna_start|>",
        "end": "<|dna_end|>",
        "pad": "<|dna_pad|>",
    },
}

__all__ = [
    "DNAConvFormer",
    "Qwen3VLDNAEncoder",
    "Qwen3VLDNAProjector",
    "DNACharProcessor",
    "DNACharTokenizer",
    "register_modality",
    "MODALITY_CONFIG_KEY",
    "TOKEN_DEFS",
]


def register_modality(router, config, llm_hidden_size: int):
    """Register the DNA encoder + projector under ``router.encoders.dna`` /
    ``router.projectors.dna``."""
    encoder = DNAConvFormer(config)
    projector = Qwen3VLDNAProjector(config, llm_hidden_size)

    router.register_modality(
        "dna",
        encoder=encoder,
        projector=projector,
        is_image_like=True,
    )
    logger.info(
        f"DNA modality registered: encoder_hidden={config.dna_encoder_hidden_size}, "
        f"latent_tokens={config.num_latent_tokens}, llm_hidden={llm_hidden_size}"
    )
