"""Bio_qwen3vl-style ERA5 dataset.

Produces samples ready for the bio_qwen3vl pipeline:

* ``input_ids`` / ``attention_mask`` / ``position_ids`` / ``labels``:
  the chat template with ``<|weather_start|><|weather_pad|>*N<|weather_end|>``
  expanded to ``meteo_num_tokens`` pad tokens.

* ``weather_input_ids`` / ``weather_attention_mask`` / ``weather_grid_thw``:
  dummy placeholders sized to ``swin_H * swin_W`` so the
  ``ModalityRouter.scatter_all`` finds them under the standard kwargs.

* The actual meteorological data — ``meteo_values`` / ``targets`` / ``times`` /
  ``lead_hours`` / ``polaris_task`` / optional ``channel_mask`` — is
  carried in a ``"meteo_data"`` dict so the collator can stack across
  the batch in fp32.

The collator unpacks ``meteo_data`` into the canonical
``weather_meteo_values`` / ``weather_targets`` / ``weather_times`` /
``weather_lead_hours`` / ``weather_polaris_task`` kwargs that the encoder
reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import transformers


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"  # only used for compatibility; the
                                       # weather pipeline uses <|weather_pad|>


# ---------------------------------------------------------------------------
# Chat-template builder for the meteo task
# ---------------------------------------------------------------------------

def _build_meteo_messages(
    input_text: str,
    meteo_pad_token: str,
    weather_start_token: str,
    weather_end_token: str,
    meteo_num_tokens: int,
) -> List[Dict[str, Any]]:
    """Build a 2-turn system+user message list with the meteo placeholder.

    The placeholder string is the same as what Polaris originally used,
    swapping ``<|vision_start|>``→``<|weather_start|>`` and
    ``<|image_pad|>``→``<|weather_pad|>`` so the bio_qwen3vl router knows
    where to scatter the encoder output.
    """
    pads = meteo_pad_token * meteo_num_tokens
    user_text = (
        f"{weather_start_token}{pads}\n{weather_end_token}"
        f"Describe this image."
    )
    return [
        {"role": "system",
         "content": [{"type": "text", "text": f"forecast\n{input_text}"}]},
        {"role": "user",
         "content": [{"type": "text", "text": user_text}]},
    ]


def preprocess_meteo_chat(
    input_text: str,
    processor,
    meteo_pad_token: str,
    weather_start_token: str,
    weather_end_token: str,
    meteo_num_tokens: int,
    add_assistant_prompt: bool = True,
) -> Dict[str, torch.Tensor]:
    msgs = _build_meteo_messages(
        input_text, meteo_pad_token, weather_start_token,
        weather_end_token, meteo_num_tokens,
    )
    full = processor.apply_chat_template(
        msgs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_assistant_prompt,
    )
    input_ids = full["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
    full["input_ids"] = input_ids
    return full
