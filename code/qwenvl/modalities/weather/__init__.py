"""Weather modality package.

Polaris-style global forecast support, integrated as a standard modality
under the ``ModalityRouter`` framework.

The package contributes:

* ``WeatherEncoder``    — Polaris ``CubeEmbedConv`` + Swin encoder + meteo merger.
* ``IdentityWeatherProjector`` — pass-through; merger already projects to LLM hidden.
* ``WeatherDecoder``    — Polaris meteo head + Charbonnier regression loss
  (registers ``compute_loss_from_hidden`` for the router).

Special tokens: ``<|weather_start|>``, ``<|weather_end|>``, ``<|weather_pad|>``.

Adding the modality at runtime:

    --weather_config_path qwenvl/modalities/weather/configs/config_weather_0p25_channel70.json
    --tune_weather_encoder True --tune_weather_decoder True
"""

import logging

from .encoder import WeatherEncoder
from .projector import IdentityWeatherProjector
from .decoder import WeatherDecoder
from .processor import WeatherProcessor

logger = logging.getLogger(__name__)

MODALITY_CONFIG_KEY = "weather_config"

TOKEN_DEFS = {
    "weather": {
        "start": "<|weather_start|>",
        "end": "<|weather_end|>",
        "pad": "<|weather_pad|>",
    },
}

__all__ = [
    "WeatherEncoder",
    "IdentityWeatherProjector",
    "WeatherDecoder",
    "WeatherProcessor",
    "register_modality",
    "MODALITY_CONFIG_KEY",
    "TOKEN_DEFS",
]


def register_modality(router, config, llm_hidden_size: int):
    """Register the weather modality with the ``ModalityRouter``.

    Args:
        router:           ModalityRouter instance.
        config:           Qwen3VLWeatherConfig (Polaris-style hyperparams).
        llm_hidden_size:  Hidden size of the LLM backbone — overrides
                          ``config.qwenvl_dim`` so the merger / mlp_qwen2swin
                          bridges are sized to the actual LLM in use.
    """
    # Force qwenvl_dim to match the active LLM, ignoring whatever was saved
    # in the JSON config.  (The merger is randomly initialised when
    # init_weather_from is used, so 3584→4096 transitions are seamless.)
    if config.qwenvl_dim != llm_hidden_size:
        logger.info(
            f"[weather] overriding qwenvl_dim {config.qwenvl_dim} → {llm_hidden_size} "
            f"(active LLM hidden size)"
        )
        config.qwenvl_dim = llm_hidden_size

    encoder = WeatherEncoder(config)
    projector = IdentityWeatherProjector(qwenvl_dim=llm_hidden_size)
    decoder = WeatherDecoder(config, encoder=encoder)

    router.register_modality(
        "weather",
        encoder=encoder,
        projector=projector,
        decoder=decoder,
        is_image_like=True,
    )
    logger.info(
        f"[weather] registered: in_chans={config.in_chans}, "
        f"image_size={config.image_size}, patch_size={config.patch_size}, "
        f"hidden={config.hidden_size}, qwenvl_dim={config.qwenvl_dim}"
    )
