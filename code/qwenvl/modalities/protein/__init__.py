"""
Protein modality package.

Provides encoder (ESM2 backbone + resampler), projector, and processor for
protein sequences. The model discovers and registers them via the
``register_modality()`` function.
"""

import logging
from pathlib import Path

from .encoder import ProteinEncoder, ProteinConvFormer, ProteinESMCEncoder  # noqa: F401
from .projector import ProteinProjector
from .processor import ProteinTokenizer

logger = logging.getLogger(__name__)

MODALITY_CONFIG_KEY = "protein_config"

TOKEN_DEFS = {
    "protein": {
        "start": "<|protein_start|>",
        "end": "<|protein_end|>",
        "pad": "<|protein_pad|>",
    }
}

__all__ = [
    "ProteinEncoder",
    "ProteinConvFormer",
    "ProteinESMCEncoder",
    "ProteinProjector",
    "ProteinTokenizer",
    "make_protein_tokenizer",
    "register_modality",
    "MODALITY_CONFIG_KEY",
    "TOKEN_DEFS",
]


def make_protein_tokenizer(backbone_name: str = ""):
    """Construct the right protein tokenizer for a given backbone variant.

    ESM2 has a 33-token vocab with a different ordering from the 31-token
    ESMC/ESM3 vocab; using the wrong tokenizer would feed garbage ids into
    the backbone.  Centralising the dispatch here keeps training and
    inference aligned (both call this helper).
    """
    if (backbone_name or "").startswith("esm2"):
        from .esm2_tokenizer import Esm2Tokenizer
        return Esm2Tokenizer()
    return ProteinTokenizer()


def register_modality(router, config, llm_hidden_size: int):
    """Register protein modality with the ModalityRouter.

    This is the standard entry point called by the model during __init__.
    Each modality package must provide this function with the same signature:
        register_modality(router, config, llm_hidden_size) -> None

    Args:
        router: ModalityRouter instance.
        config: Modality-specific config (Qwen3VLProteinConfig).
        llm_hidden_size: Hidden size of the LLM backbone.
    """
    import torch

    backbone_name = getattr(config, "protein_backbone_name", "esm3_sm_open_v1")
    backbone_path = getattr(config, "protein_encoder_path", None)
    freeze = getattr(config, "freeze_protein_backbone", False)

    if backbone_name == "convformer":
        # Lightweight self-contained encoder, no ESM3 dependency
        if config.protein_encoder_hidden_size >= 1536:
            logger.warning(
                f"ProteinConvFormer with hidden={config.protein_encoder_hidden_size} "
                f"is likely over-parametrized (12-layer transformer at 1536d ≈ 500M+ params). "
                f"For the documented 50–100M range, set protein_encoder_hidden_size=768 (or 1024)."
            )
        encoder = ProteinConvFormer(config)
        if backbone_path and Path(backbone_path).is_file():
            state = torch.load(backbone_path, map_location="cpu")
            result = encoder.load_state_dict(state, strict=False)
            logger.info(f"Loaded ProteinConvFormer from {backbone_path}: {result}")
        else:
            logger.info(
                f"ProteinConvFormer initialized with random weights "
                f"(hidden={config.protein_encoder_hidden_size}, "
                f"layers={getattr(config, 'num_encoder_layers', 12)})"
            )
        # Honor freeze_protein_backbone: for ConvFormer the "backbone" is
        # everything except the latent resampler (semantically analogous to
        # ESM3 backbone vs. resampler split).
        if freeze:
            for name, p in encoder.named_parameters():
                if not name.startswith("resampler."):
                    p.requires_grad = False
            logger.info(
                "ProteinConvFormer backbone frozen "
                "(tok/pos embed + conv stem + transformer + final_norm); resampler stays trainable"
            )
    elif backbone_name.startswith("esm2"):
        encoder = ProteinEncoder(config)
        _load_esm2(encoder, backbone_name, backbone_path, freeze)
    else:
        raise ValueError(
            f"Unknown protein backbone '{backbone_name}'. "
            "Use 'convformer' or 'esm2_t6_8M_UR50D' .. 'esm2_t48_15B_UR50D'."
        )

    projector = ProteinProjector(config, llm_hidden_size)

    router.register_modality(
        "protein",
        encoder=encoder,
        projector=projector,
        is_image_like=True,
    )
    logger.info(
        f"Protein modality registered: backbone={backbone_name}, "
        f"hidden={config.protein_encoder_hidden_size}, "
        f"latent_tokens={config.num_latent_tokens}, llm_hidden={llm_hidden_size}, "
        f"freeze={freeze}"
    )


# --------------------------------------------------------------------------
# Backbone loaders
# --------------------------------------------------------------------------

def _load_esm2(encoder, backbone_name, backbone_path, freeze):
    """Load an ESM2 backbone (HuggingFace ``transformers.EsmModel``).

    The wrapper exposes the same forward signature as ESM3, so
    :class:`ProteinEncoder` doesn't need to know which backbone it has.

    Validates that the loaded backbone's hidden_size matches what the
    encoder's resampler / projector were sized for — a mismatch here means
    the user's ``protein_backbone_name`` and ``protein_encoder_path`` point
    to different ESM2 variants, and would otherwise crash later inside
    LayerNorm with a confusing shape error.
    """
    from .esm2_backbone import load_esm2_backbone

    backbone = load_esm2_backbone(backbone_name, backbone_path, freeze)

    expected = encoder.config.protein_encoder_hidden_size
    actual = backbone.d_model
    if expected != actual:
        raise ValueError(
            f"ESM2 backbone hidden_size mismatch: "
            f"weights at {backbone_path} have hidden={actual}, but "
            f"protein_encoder_hidden_size={expected} (set by "
            f"protein_backbone_name={backbone_name!r}). "
            f"Either fix protein_backbone_name to match the weights, or "
            f"point protein_encoder_path to the correct variant's directory."
        )

    encoder.set_backbone(backbone)


