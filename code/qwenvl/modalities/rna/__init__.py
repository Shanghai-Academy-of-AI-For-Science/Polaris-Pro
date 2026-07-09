"""
RNA modality package.

Provides encoder, projector, decoder, and processor for RNA sequences.
All components are self-contained — the model discovers and registers them
via the ``register_modality()`` function below.

DNA was historically aliased onto this package via
``router.register_modality("rna", aliases=["dna"])``; that alias has been
removed.  DNA is now an independent modality at ``qwenvl.modalities.dna``.

Adding a new modality? Copy this package and implement register_modality().
"""

import logging

from .encoder import RNAConvFormer, Qwen3VLRNAEncoder
from .projector import Qwen3VLRNAProjector
from .decoder import RNALMDecoder
from .processor import RNACharProcessor, RNACharTokenizer

logger = logging.getLogger(__name__)

MODALITY_CONFIG_KEY = "rna_config"

TOKEN_DEFS = {
    "rna": {
        "start": "<|rna_start|>",
        "end": "<|rna_end|>",
        "pad": "<|rna_pad|>",
    },
}

__all__ = [
    "RNAConvFormer",
    "Qwen3VLRNAEncoder",
    "Qwen3VLRNAProjector",
    "RNALMDecoder",
    "RNACharProcessor",
    "RNACharTokenizer",
    "register_modality",
    "MODALITY_CONFIG_KEY",
    "TOKEN_DEFS",
]


def register_modality(router, config, llm_hidden_size: int):
    """Register RNA with the ModalityRouter.

    DNA has its own independent registration in ``qwenvl.modalities.dna``;
    the previous ``aliases=["dna"]`` mechanism has been removed.

    Args:
        router: ModalityRouter instance.
        config: Modality-specific config (e.g. Qwen3VLRNAConfig).
        llm_hidden_size: Hidden size of the LLM backbone.
    """
    encoder = RNAConvFormer(config)
    projector = Qwen3VLRNAProjector(config, llm_hidden_size)
    decoder = RNALMDecoder(
        llm_hidden_size=llm_hidden_size,
        rna_vocab_size=config.rna_vocab_size,
    )

    # NOTE: Pretrained encoder .pt loading is performed by the training
    # script *only when* this modality is newly registered (i.e., not
    # already in the bio_qwen3vl checkpoint being resumed).  Loading here
    # would either be wasted work (state_dict load overwrites it) or worse,
    # silently overwrite a previously-trained checkpoint when the user
    # accidentally points the path at an old base .pt.  See
    # the model loader.

    router.register_modality(
        "rna",
        encoder=encoder,
        projector=projector,
        decoder=decoder,
        is_image_like=True,
    )
    logger.info(
        f"RNA modality registered: encoder_hidden={config.rna_encoder_hidden_size}, "
        f"latent_tokens={config.num_latent_tokens}, llm_hidden={llm_hidden_size}"
    )
