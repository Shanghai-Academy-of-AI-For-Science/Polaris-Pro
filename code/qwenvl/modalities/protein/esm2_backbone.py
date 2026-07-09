"""ESM2 backbone wrapper.

Wraps ``transformers.EsmModel`` to expose the same ``(sequence_tokens,
sequence_id) -> output.embeddings`` contract that :class:`ProteinEncoder`
expects from its ``backbone`` slot (originally designed for ESM3).  This
keeps :class:`ProteinEncoder` completely backbone-agnostic — switching
between ESM2 and ESM3 only requires loading a different wrapper into
``encoder.set_backbone(...)``.

ESM2 size table is the canonical mapping between the user-facing
``backbone_name`` (e.g. ``esm2_t33_650M_UR50D``) and the per-size
architecture knobs.  ``hidden_size`` is what bubbles up into
``protein_encoder_hidden_size`` so the resampler / projector reshape
themselves automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class _ESM2Output:
    """Mirrors the ``ESM3Output`` shape that :class:`ProteinEncoder` reads.

    Only ``embeddings`` is consumed; the rest of ESM3's multi-modal heads
    (structure / SS8 / SASA / function / residue) have no analog in ESM2
    and are simply not exposed here.
    """

    embeddings: torch.Tensor


class ESM2Wrapper(nn.Module):
    """Thin nn.Module wrapping a ``transformers.EsmModel`` instance.

    Exposes ``forward(sequence_tokens=..., sequence_id=...) -> _ESM2Output``
    so :class:`ProteinEncoder` can call it without knowing it's ESM2 vs ESM3.
    """

    # Per-size architecture: (n_layers, hidden_size, n_attn_heads, approx_params)
    SIZE_TABLE = {
        "esm2_t6_8M_UR50D":    (6,  320,  20,    8_000_000),
        "esm2_t12_35M_UR50D":  (12, 480,  20,   35_000_000),
        "esm2_t30_150M_UR50D": (30, 640,  20,  150_000_000),
        "esm2_t33_650M_UR50D": (33, 1280, 20,  650_000_000),
        "esm2_t36_3B_UR50D":   (36, 2560, 40,  3_000_000_000),
        "esm2_t48_15B_UR50D":  (48, 5120, 40, 15_000_000_000),
    }

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model
        self.d_model = hf_model.config.hidden_size

    def forward(self, sequence_tokens, sequence_id=None, **kwargs):
        # ``sequence_id`` is ESM3's name for the bool attention mask; ESM2
        # accepts an int/bool ``attention_mask`` of the same shape.  Fall
        # back to ``sequence_tokens != pad_id`` if the caller didn't pass it.
        if sequence_id is None:
            attention_mask = (sequence_tokens != 1).long()
        else:
            attention_mask = sequence_id.long() if sequence_id.dtype != torch.long else sequence_id
        out = self.hf_model(input_ids=sequence_tokens, attention_mask=attention_mask)
        return _ESM2Output(embeddings=out.last_hidden_state)


def load_esm2_backbone(
    backbone_name: str,
    backbone_path: Optional[str],
    freeze: bool,
) -> ESM2Wrapper:
    """Build an ESM2 backbone from one of three sources.

    1. **Local directory** (``backbone_path`` is a directory that exists):
       ``EsmModel.from_pretrained(backbone_path)``.  Recommended for offline
       clusters — pre-download with
       ``huggingface-cli download facebook/<name> --local-dir <path>``.
    2. **Random init** (``backbone_path`` in ``{None, '', 'random', 'protein',
       'none'}``): construct ``EsmConfig`` from :data:`ESM2Wrapper.SIZE_TABLE`
       and instantiate without weights.
    3. **HuggingFace Hub** (anything else): ``from_pretrained(f"facebook/{name}")``.
       Requires network or a populated HF cache.
    """
    from transformers import EsmConfig, EsmModel

    if backbone_name not in ESM2Wrapper.SIZE_TABLE:
        raise ValueError(
            f"Unknown ESM2 variant: {backbone_name!r}. "
            f"Choices: {sorted(ESM2Wrapper.SIZE_TABLE.keys())}"
        )

    if backbone_path and Path(backbone_path).is_dir():
        # Detect a meta-device parent context: when this encoder is built
        # during ``Qwen3VLForConditionalGeneration.from_pretrained`` (eval /
        # inference), transformers 5.x initializes the outer model on the
        # ``meta`` device for low-memory loading.  A NESTED ``from_pretrained``
        # is then rejected ("from_pretrained with a meta device context").  In
        # that case we don't need the pretrained ESM2 weights at all — the
        # trained backbone tensors live in the parent checkpoint's
        # model.safetensors and overwrite whatever we build here.  So build the
        # architecture from the local config.json instead and let the parent
        # ckpt load fill in the weights.
        in_meta_ctx = False
        try:
            in_meta_ctx = (torch.empty(0).device.type == "meta")
        except Exception:
            in_meta_ctx = False
        if in_meta_ctx:
            logger.info(
                f"Building ESM2 '{backbone_name}' from config at {backbone_path} "
                f"(meta-device parent context; weights come from the parent checkpoint)"
            )
            hf_model = EsmModel(EsmConfig.from_pretrained(backbone_path))
        else:
            logger.info(f"Loading ESM2 '{backbone_name}' from local dir {backbone_path}")
            hf_model = EsmModel.from_pretrained(backbone_path)
    elif backbone_path in (None, "", "random", "protein", "none"):
        layers, hidden, heads, _ = ESM2Wrapper.SIZE_TABLE[backbone_name]
        logger.info(
            f"Initializing ESM2 '{backbone_name}' with random weights "
            f"(layers={layers}, hidden={hidden}, heads={heads})"
        )
        cfg = EsmConfig(
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=hidden * 4,
            vocab_size=33,
            max_position_embeddings=1026,
            position_embedding_type="rotary",
            # ESM2's official tokenizer maps <pad> -> id 1.  Matching ids
            # below means EsmEmbeddings.padding_idx is well-defined; the
            # default EsmConfig leaves pad_token_id=None, which makes
            # ``create_position_ids_from_input_ids`` call ``.ne(None)``
            # at the first forward and crash with a confusing "ne()
            # received NoneType" TypeError.
            pad_token_id=1,
            mask_token_id=32,
            bos_token_id=0,
            eos_token_id=2,
        )
        hf_model = EsmModel(cfg)
    else:
        # Treat as HF Hub id; transformers will use HF cache if offline.
        hub_id = backbone_path if "/" in str(backbone_path) else f"facebook/{backbone_name}"
        logger.info(f"Loading ESM2 from HuggingFace: {hub_id}")
        hf_model = EsmModel.from_pretrained(hub_id)

    # Belt-and-suspenders: ESM2 ckpts in the wild occasionally ship
    # config.json with pad_token_id=None.  Patch the live module's
    # padding_idx so create_position_ids_from_input_ids doesn't ``.ne(None)``.
    if getattr(hf_model.config, "pad_token_id", None) is None:
        logger.warning(
            "ESM2 config has pad_token_id=None; defaulting to 1 (the standard "
            "ESM2 vocab pad id) to avoid create_position_ids_from_input_ids crash."
        )
        hf_model.config.pad_token_id = 1
    embeddings = getattr(hf_model, "embeddings", None)
    if embeddings is not None and getattr(embeddings, "padding_idx", None) is None:
        embeddings.padding_idx = int(hf_model.config.pad_token_id)

    n_params = sum(p.numel() for p in hf_model.parameters())
    logger.info(f"ESM2 backbone '{backbone_name}': {n_params:,} parameters")

    # ESM2Wrapper consumes only ``last_hidden_state``. The optional HF pooler
    # is newly initialized for local ESM2 checkpoints and is not on the
    # active forward path, so leaving it trainable creates rank-dependent
    # unused pooler.
    pooler = getattr(hf_model, "pooler", None)
    if pooler is not None:
        for p in pooler.parameters():
            p.requires_grad = False
        logger.info("ESM2 pooler frozen (unused by ProteinEncoder)")

    if freeze:
        for p in hf_model.parameters():
            p.requires_grad = False
        logger.info("ESM2 backbone frozen")

    return ESM2Wrapper(hf_model)
