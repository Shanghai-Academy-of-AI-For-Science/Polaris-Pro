"""Weather processor (placeholder).

The bio modalities use a per-modality processor to turn raw inputs into
encoder tensors.  Weather is different: the ``ERA5QwenVLDataset`` (in
``mkb/modalities/weather/data/era5_dataset.py``) directly emits all
encoder tensors plus the chat-template placeholder.  This file exists
only so that ``BaseProcessor``-style callers — if any — see a concrete
class for the modality.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class WeatherProcessor:
    """No-op processor; the dataset class does the real work."""

    modality_name = "weather"

    def process_input(self, raw_input: Any, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(
            "Weather inputs come straight from ERA5QwenVLDataset; the dataset "
            "produces encoder-ready tensors directly."
        )

    def build_placeholder(
        self, raw_input: Any, is_output: bool = False, **kwargs,
    ) -> Tuple[str, Optional[Any]]:
        raise NotImplementedError(
            "Use the chat-template helper in mkb/modalities/weather/data/era5_dataset.py."
        )
