"""Autoregressive forecast rollout aligned with our training engine.

Mirrors the per-step state advancement of
``qwenvl.modalities.weather.rollout.WeatherRolloutEngine``:

  * ``step_idx`` recurs as the actual rollout step (0, 1, 2, ...). This
    drives both ``hour``/``doy`` (via ``cur_times = times + lead·step_idx``)
    AND ``step_scalar = log1p(step_idx) / log1p(Tmax)`` inside
    ``WeatherEncoder.get_condition`` — exactly what the model has been
    trained against.
  * ``times`` (the init time) stays fixed; ``cur_times`` is derived per
    step from ``step_idx``.
  * ``lead_hours`` stays fixed at the training-time per-step interval
    (typically 6 h).  ``effective_hour = lead_hours * (step_idx + 1)``.
  * ``cur_values`` slides: drop the oldest historical frame, append the
    model's previous prediction.

Important: Polaris's bundled ``infer_forecast.py`` uses ``step_idx=0``
forever and advances ``times`` instead.  That convention is *not* what
this checkpoint was trained on — using it would feed the model a
``step_scalar=0`` it never saw at training time.  Use this script for
the weather checkpoint.

Inputs
------
``--input``: a netCDF or zarr path with a single ``[T_in, C, H, W]`` (or
``[C, H, W]``) tensor named after channels in the same order the model
was trained on.  ``T_in`` must equal ``ERA5_HIST_FRAMES`` (default 1).
``--init_time``: ISO timestamp of the input frame; falls back to the
``time`` attribute on the netCDF.

Outputs
-------
For every rollout step, saves ``<save_dir>/<effective_hour>03dh.nc`` with
the unnormalized prediction over all channels.  ``--max_lead_hour`` and
``--lead_step_hours`` together determine how many rollout steps to run
(``ceil(max_lead_hour / lead_step_hours)``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Ensure repo root is on PYTHONPATH so this script is runnable directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoProcessor

from qwenvl.models.modeling_bio_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.modalities.weather.data import load_meteorological_buffers
from qwenvl.modalities.weather.data.era5_dataset import preprocess_meteo_chat


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--checkpoint", required=True,
                        help="Path to a trained bio_qwen3vl_weather checkpoint (the directory "
                             "with config.json + safetensors / deepspeed shards).")
    parser.add_argument("--era5_data_path", required=True,
                        help="ERA5 zarr path used during training. Provides mean/std/weight "
                             "buffers via load_meteorological_buffers; must match training.")
    parser.add_argument("--input", required=True,
                        help="netCDF (.nc) input frame: [T_in, C, H, W] or [C, H, W].")
    parser.add_argument("--save_dir", default="outputs/forecast",
                        help="Output directory for <effective_hour>h.nc files.")

    parser.add_argument("--init_time", default=None,
                        help="ISO timestamp of the input frame, e.g. '2023-07-01T00:00'. "
                             "If omitted, falls back to input_da.attrs['time'] or the last "
                             "time coord on input_da.")
    parser.add_argument("--max_lead_hour", type=int, default=240,
                        help="Maximum forecast lead time (hours).")
    parser.add_argument("--lead_step_hours", type=int, default=6,
                        help="Hours per rollout step. Must match training (default 6).")

    parser.add_argument("--era5_image_size", type=int, nargs=2, default=[721, 1440])
    parser.add_argument("--era5_latlon_range", type=float, nargs=4, default=None,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--remove_channels", type=str, nargs="*", default=None,
                        help="Channels removed at training time. MUST match training: "
                             "the encoder mean/std buffers depend on this.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--save_channels", type=str, nargs="*", default=None,
                        help="If set, only save these channels in each output .nc. "
                             "Default: save all channels.")

    parser.add_argument("--load_ema", action="store_true",
                        help="If set, load weather_ema_weights.pt from the checkpoint dir "
                             "(written by WeatherEMACallback) and overlay onto the model "
                             "BEFORE rollout. Improves stability/quality of the prediction; "
                             "no effect if the file is missing.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_input(input_path: str, init_time_arg: Optional[str]):
    """Read a netCDF and return (init_field np.float32, init_time, channel_names, lats, lons).

    init_field shape: ``[T_in, C, H, W]`` (T_in inferred from the file).
    """
    if input_path.endswith(".zarr") or os.path.isdir(input_path):
        da = xr.open_zarr(input_path)
    else:
        da = xr.open_dataarray(input_path)

    # When opened as a Dataset (zarr), pull the first variable.
    if hasattr(da, "data_vars"):
        var = list(da.data_vars)[0]
        da = da[var]

    if "time" in da.dims:
        if init_time_arg is not None:
            init_time = pd.to_datetime(init_time_arg)
            da = da.sel(time=init_time, method="nearest")
            init_field = da.values
        else:
            init_time = pd.to_datetime(da.time.values[-1])
            init_field = da.isel(time=-1).values
    else:
        init_time = pd.to_datetime(init_time_arg or da.attrs.get("time", "2023-07-01"))
        init_field = da.values

    init_field = init_field.astype(np.float32)
    if init_field.ndim == 3:
        init_field = init_field[None]  # → [T_in=1, C, H, W]

    channels = [str(c) for c in da.channel.values] if "channel" in da.coords else None
    lats = da.lat.values if "lat" in da.coords else None
    lons = da.lon.values if "lon" in da.coords else None

    return init_field, init_time, channels, lats, lons


def _compute_meteo_num_tokens(weather_config) -> int:
    in_h, in_w = weather_config.image_size if isinstance(weather_config.image_size, (list, tuple)) \
        else (weather_config.image_size, weather_config.image_size)
    ps = weather_config.patch_size
    swin_h = in_h // 2 * 2 if ps == 1 else in_h // ps
    swin_w = in_w if ps == 1 else in_w // ps
    return swin_h * swin_w


def _build_chat_inputs(processor, lead_hours: int, num_tokens: int, weather_token_ids, device):
    """Construct ``input_ids`` / ``attention_mask`` matching the dataset's
    chat template so the LLM sees the same prompt structure used at training."""
    pad_str = processor.tokenizer.convert_ids_to_tokens(int(weather_token_ids["pad"]))
    start_str = processor.tokenizer.convert_ids_to_tokens(int(weather_token_ids["start"]))
    end_str = processor.tokenizer.convert_ids_to_tokens(int(weather_token_ids["end"]))

    input_text = f"Predict global weather state {int(lead_hours)} hours ahead at a 0.25° resolution"
    full = preprocess_meteo_chat(
        input_text=input_text,
        processor=processor,
        meteo_pad_token=pad_str,
        weather_start_token=start_str,
        weather_end_token=end_str,
        meteo_num_tokens=num_tokens,
        add_assistant_prompt=True,
    )
    input_ids = full["input_ids"].to(device)
    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id).to(device)
    return input_ids, attention_mask


def _save_step(output_unnorm: np.ndarray,
               init_time, effective_hour: int,
               channels, lats, lons,
               save_dir: str,
               keep_channels: Optional[List[str]]):
    if channels is not None and keep_channels is not None:
        idx = [channels.index(c) for c in keep_channels if c in channels]
        if idx:
            output_unnorm = output_unnorm[idx]
            channels = [channels[i] for i in idx]

    fcst_time = pd.to_datetime(init_time) + pd.Timedelta(hours=int(effective_hour))
    da = xr.DataArray(
        output_unnorm,
        dims=["channel", "lat", "lon"],
        coords=dict(
            channel=channels if channels is not None else np.arange(output_unnorm.shape[0]),
            lat=lats if lats is not None else np.arange(output_unnorm.shape[1]),
            lon=lons if lons is not None else np.arange(output_unnorm.shape[2]),
        ),
        attrs=dict(
            init_time=str(init_time),
            fcst_time=str(fcst_time),
            lead_hour=int(effective_hour),
        ),
    )
    fname = os.path.join(save_dir, f"{int(effective_hour):03d}h.nc")
    da.to_netcdf(fname)


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    # ── Load model + processor ────────────────────────────────────────
    print(f"[load] checkpoint = {args.checkpoint}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint, use_fast=False)

    inner = model.model
    router = inner.modality_router
    if "weather" not in router.encoders:
        raise RuntimeError("checkpoint has no weather modality registered")
    encoder = router.encoders["weather"]
    decoder = router.decoders["weather"]

    # Optional EMA overlay: WeatherEMACallback drops a sidecar
    # ``weather_ema_weights.pt`` next to each checkpoint with fp32 EMA
    # tensors keyed by the live model namespace.  Apply *after*
    # from_pretrained finished so the EMA weights win.
    if args.load_ema:
        ema_path = os.path.join(args.checkpoint, "weather_ema_weights.pt")
        if not os.path.isfile(ema_path):
            print(f"[load] --load_ema set but {ema_path} not found, "
                  f"falling back to main weights", flush=True)
        else:
            ema_state = torch.load(ema_path, map_location="cpu")
            ema_overlay = {}
            for k, v in ema_state.items():
                if not (
                    k.startswith("model.modality_router.encoders.weather.")
                    or k.startswith("model.modality_router.decoders.weather.")
                ):
                    continue
                ema_overlay[k] = (
                    v.to(device=device, dtype=dtype)
                    if v.is_floating_point()
                    else v.to(device=device)
                )
            result = model.load_state_dict(ema_overlay, strict=False)
            # Note: ``missing_keys`` here is huge — it lists every model
            # parameter that wasn't in the EMA overlay (i.e. Qwen3 LLM
            # weights, vision tower, etc.).  That's expected; the EMA file
            # only stores weather params.  We only flag *unexpected* keys
            # (EMA tensors that didn't match a real model parameter) since
            # those signal a real namespace drift.
            print(
                f"[load] EMA overlay: {len(ema_overlay)} weather tensors applied; "
                f"unexpected_keys={len(result.unexpected_keys)}",
                flush=True,
            )
            if result.unexpected_keys:
                print(
                    f"[load] WARNING — EMA had keys the model didn't accept: "
                    f"{result.unexpected_keys[:5]}...",
                    flush=True,
                )

    weather_token_ids = (model.config.bio_token_ids or {}).get("weather")
    if weather_token_ids is None or "pad" not in weather_token_ids:
        raise RuntimeError("config.bio_token_ids['weather'] missing — check checkpoint config")

    # ── Inject ERA5 statistics (must match training) ──────────────────
    print(f"[load] ERA5 buffers from {args.era5_data_path}", flush=True)
    ch, idx, coords, buffers = load_meteorological_buffers(
        data_path=args.era5_data_path,
        image_size=tuple(args.era5_image_size),
        latlon_range=tuple(args.era5_latlon_range) if args.era5_latlon_range else None,
        remove_channels=args.remove_channels,
    )
    encoder.inject_meteorological_context(ch, idx, coords, buffers)

    # ── Read initial field ────────────────────────────────────────────
    init_field, init_time, in_channels, in_lats, in_lons = _load_input(args.input, args.init_time)
    print(f"[load] input field shape={init_field.shape}, init_time={init_time}", flush=True)

    # Use coords from the input file when present; otherwise fall back to
    # the buffer coords (which always cover the global 0.25° grid).
    out_channels = in_channels if in_channels is not None else ch
    out_lats = in_lats if in_lats is not None else np.array(coords["lat"])
    out_lons = in_lons if in_lons is not None else np.array(coords["lon"])

    # Sanity check: input channel ordering must match the ERA5 buffers,
    # otherwise mean/std normalisation pairs the wrong stats with the
    # wrong variable.  The pipeline trusts positional ordering — there
    # is no automatic reorder.
    if in_channels is not None:
        if list(in_channels) != list(ch):
            print(
                f"[load] WARNING — input channel ordering differs from "
                f"ERA5 buffer ordering. Predictions will be garbage.\n"
                f"  input first 5: {list(in_channels)[:5]}\n"
                f"  buffer first 5: {list(ch)[:5]}\n"
                f"  Reorder your input nc to match, or strip channel coord "
                f"from the input.",
                flush=True,
            )

    # ── Set up cur_values: same normalisation path as encoder.forward ──
    cur_values = torch.from_numpy(init_field).to(device)
    cur_values = torch.nan_to_num(cur_values).unsqueeze(0)  # add batch dim → [1, T_in, C, H, W]
    cur_values = cur_values.to(torch.float32)
    cur_values = encoder._reset_input(cur_values)

    # ── Chat tokens (built once: structure doesn't change per step) ───
    weather_config = model.config.weather_config
    num_tokens = _compute_meteo_num_tokens(weather_config)
    input_ids, attention_mask = _build_chat_inputs(
        processor=processor,
        lead_hours=int(args.lead_step_hours),
        num_tokens=num_tokens,
        weather_token_ids=weather_token_ids,
        device=device,
    )
    weather_input_ids = torch.ones(1, num_tokens, dtype=torch.long, device=device)
    weather_attention_mask = torch.ones(1, num_tokens, dtype=torch.long, device=device)
    weather_grid_thw = torch.tensor([[1, 1, num_tokens]], dtype=torch.long, device=device)

    # ── Rollout ───────────────────────────────────────────────────────
    n_steps = (args.max_lead_hour + args.lead_step_hours - 1) // args.lead_step_hours
    print(f"[rollout] steps={n_steps}, lead_step={args.lead_step_hours}h, "
          f"max_lead={args.max_lead_hour}h", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    times_idx = pd.DatetimeIndex([init_time])
    lead_hours_t = torch.tensor([float(args.lead_step_hours)], device=device, dtype=torch.float32)

    t_start = time.perf_counter()
    for t in range(n_steps):
        effective_hour = int((t + 1) * args.lead_step_hours)

        # Each forward call refreshes the encoder's internal cache.
        outputs = inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            weather_input_ids=weather_input_ids,
            weather_attention_mask=weather_attention_mask,
            weather_grid_thw=weather_grid_thw,
            weather_meteo_values=cur_values,
            weather_lead_hours=lead_hours_t,
            weather_times=times_idx,
            weather_step_idx=t,            # ← critical: aligns with training
        )
        hidden = outputs[0]

        next_frame = decoder.predict_from_hidden(
            hidden,
            input_ids=input_ids,
            __weather_pad_id__=int(weather_token_ids["pad"]),
        )
        if next_frame is None:
            raise RuntimeError(f"step {t}: predict_from_hidden returned None — check pad tokens.")

        # next_frame is [B=1, C, H, W] in normalised space.
        pred_unnorm = encoder.unnormalize(next_frame.unsqueeze(1))  # → [1, 1, C, H, W]
        pred_np = pred_unnorm[0, 0].float().cpu().numpy()

        _save_step(
            pred_np, init_time, effective_hour,
            channels=out_channels, lats=out_lats, lons=out_lons,
            save_dir=args.save_dir, keep_channels=args.save_channels,
        )

        print(f"  step {t + 1:>2}/{n_steps}  effective {effective_hour:>3d}h  "
              f"range [{pred_np.min():.2f}, {pred_np.max():.2f}]", flush=True)

        # Slide the input window: drop oldest historical frame, append
        # this step's prediction (in normalised space, fp32).  Mirrors
        # WeatherRolloutEngine._slide_window.
        next_norm = next_frame.to(torch.float32).unsqueeze(1)  # [1, 1, C, H, W]
        cur_values = torch.cat([cur_values[:, 1:], next_norm], dim=1)

    print(f"[done] {time.perf_counter() - t_start:.1f}s, saved to {args.save_dir}", flush=True)


if __name__ == "__main__":
    run_inference(_parse_args())
