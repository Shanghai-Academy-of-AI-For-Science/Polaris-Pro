"""Identity projector for the weather modality.

The Polaris-style ``meteo_merger`` already runs inside ``WeatherEncoder``
and projects swin hidden → ``qwenvl_dim``.  The router still expects a
projector module so we register a no-op ``IdentityWeatherProjector`` to
keep the registration logic uniform with the bio modalities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityWeatherProjector(nn.Module):
    """Pass-through projector. Encoder already outputs qwenvl_dim."""

    def __init__(self, qwenvl_dim: int):
        super().__init__()
        self.qwenvl_dim = qwenvl_dim
        # Single dummy parameter so that an "is the projector trainable"
        # check does not return False unexpectedly when tune_weather_projector
        # is left at its default True; the router treats projectors with
        # no trainable params as frozen which would short-circuit the
        # dummy_forward gradient sync path.
        self._noop = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
