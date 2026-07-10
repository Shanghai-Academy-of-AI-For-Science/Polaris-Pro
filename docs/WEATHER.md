# Weather forecasting

Polaris-Pro produces a global **ERA5 0.25° forecast** (721 × 1440 grid, 70
channels): given an initial atmospheric state, it autoregressively rolls the
forecast forward in fixed lead-hour steps. This modality has its own script.

## Requirements
- `netCDF4`, `xarray`, `zarr` (in `requirements.txt`).
- **ERA5 statistics** (`--era5_data_path`): the per-channel mean/std, latitude
  weights, and static fields used by the pipeline. Bundled at
  `assets/era5_stats/` (~24 MB; derived statistics only, no ERA5 reanalysis data).
- An **input frame**: a netCDF `[C, H, W]` (or `[T_in, C, H, W]`) tensor whose
  channels are in the model's exact training order, **already normalized**
  per channel — see the note below.

> ⚠️ **Input must be pre-normalized.** The model operates in normalized space:
> the encoder does **not** apply `(x − mean) / std` to the input, and the output
> is converted back to physical units with `unnormalize` (`x · std + mean`). So
> the `--input` frame must already be standardized with the same per-channel
> `mean`/`std` in `assets/era5_stats/` (i.e. `(physical − mean) / std`). Feeding
> raw physical values yields wrong-scale forecasts that drift over the rollout.

## Run
```bash
export PYTHONPATH=$PWD/code

python code/scripts/weather/infer_forecast.py \
    --checkpoint model \
    --era5_data_path assets/era5_stats \
    --input <single_frame.nc> \
    --init_time 2024-12-31T00:00:00 \
    --era5_image_size 721 1440 \
    --max_lead_hour 24 --lead_step_hours 6 \
    --save_dir out/forecast
```
Each rollout step writes `out/forecast/<lead_hour>h.nc`.

## Notes
- `--remove_channels` and `--lead_step_hours` **must match training** — the
  encoder's mean/std buffers and step conditioning depend on them.
- Input channel order must match the ERA5 buffer order exactly (no auto-reorder).
- `--dtype bf16` (default) is stable; RoPE buffers self-heal at load.

Run `--help` for all options.
