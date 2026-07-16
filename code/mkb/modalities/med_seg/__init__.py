"""med_seg modality package — SAM3 medical-image segmentation.

Architectural sketch (see TRAINING.md / repo-level memory for the full story):

  Image  ─►  Qwen3-VL ViT  ─►  LLM  ─►  hidden_states   ┐
                                                         │  (figure-text joint
                                                         │   span: image_pad
                                                         │   block + text)
                                                         ▼
                                            proj(qwen_h → 256)
                                                         │
                                                         ▼
  Image  ─►  SAM3 Hiera backbone (multi-scale features) ─►  cross-attn  ─►  cls / box / mask

Two visual pathways:
- Qwen visual tower contributes "task-aware semantic" hidden states
- SAM3 vision backbone contributes "dense pixel" features
Late fusion happens inside SAM3's Transformer decoder.

What this package exports
-------------------------
- ``MODALITY_CONFIG_KEY``: ``"med_seg_config"`` (key on the main Qwen3VLConfig
  where ``Qwen3VLMedSegConfig`` is stored for auto-discovery)
- ``TOKEN_DEFS``: empty by design. med_seg uses native Qwen3-VL image/text
  tokens and does not resize the tokenizer/model embedding table.
- ``register_modality``: factory invoked by the model at construction time.
  Builds a ``MedSegDecoder`` skeleton (without SAM3 weights) and registers
  it as ``decoders["med_seg"]``. The training script later attaches the
  real Sam3Model via ``decoder.set_sam3(...)`` once it has been loaded.
"""

import logging

from .config import Qwen3VLMedSegConfig
from .decoder import MedSegDecoder
from .processor import MedSegProcessor

logger = logging.getLogger(__name__)

MODALITY_CONFIG_KEY = "med_seg_config"

# Unlike RNA/DNA/protein/mol, med_seg does not inject encoder embeddings into
# the LLM input via masked_scatter. It consumes hidden states from Qwen's
# native image/text prompt, so it must not add med_seg-specific special tokens.
TOKEN_DEFS = {}

__all__ = [
    "Qwen3VLMedSegConfig",
    "MedSegDecoder",
    "MedSegProcessor",
    "MODALITY_CONFIG_KEY",
    "TOKEN_DEFS",
    "register_modality",
]


def register_modality(router, config, llm_hidden_size: int):
    """Register the med_seg decoder skeleton with the ModalityRouter.

    Note: this only registers the **proj + matcher + loss** part. The
    Sam3Model itself is heavy (≈0.5 B params, often loaded via
    ``Sam3Model.from_pretrained``) and must be attached by the training
    script after model construction:

        decoder = router.get_decoder("med_seg")
        decoder.set_sam3(loaded_sam3_model)

    Args:
        router: ModalityRouter instance.
        config: Qwen3VLMedSegConfig.
        llm_hidden_size: Hidden size of the Qwen3-VL LLM backbone.
    """
    decoder = MedSegDecoder(
        llm_hidden_size=llm_hidden_size,
        config=config,
    )
    # No encoder / projector — image goes through SAM3 itself; nothing is
    # scatter'd into LLM embeddings for this modality. Register decoder only.
    router.register_modality(
        "med_seg",
        encoder=None,
        projector=None,
        decoder=decoder,
        is_image_like=False,
    )
    logger.info(
        f"med_seg modality registered (decoder skeleton): "
        f"sam3_text_dim={config.sam3_text_dim}, "
        f"freeze_sam3={config.freeze_sam3}, "
        f"llm_hidden={llm_hidden_size}. "
        f"Call decoder.set_sam3(...) to attach the SAM3 backbone."
    )
