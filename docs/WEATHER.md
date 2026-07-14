# Weather forecasting

神珍 produces a global **ERA5 0.25° forecast** (721 × 1440 grid, 70
channels): given an initial atmospheric state, it autoregressively rolls the
forecast forward in fixed lead-hour steps. This modality has its own script.

## Requirements
- `netCDF4`, `xarray`, `zarr` (in `requirements.txt`).
- **ERA5 statistics** (`--era5_data_path`): the per-channel mean/std, latitude
  weights, and static fields used by the pipeline. Bundled at
  `assets/era5_stats/` (~24 MB; derived statistics only, no ERA5 reanalysis data).
- An **input frame**: a netCDF `[C, H, W]` (or `[T_in, C, H, W]`) tensor holding
  the model's **70 evaluated channels in training order** (z/t/u/v/q on 13
  pressure levels, then `msl, t2m, ws10m, u10m, v10m`), each **normalized** as
  described below. Channels are consumed positionally — the tensor must contain
  exactly these 70 in this order (no auto-drop or reorder). `--remove_channels`
  selects the matching statistics buffers and must equal the training list.

> ⚠️ **Input is expected in normalized space.** The encoder consumes normalized
> values and the output is returned in physical units via `x · std + mean`. The
> `--input` frame must therefore be standardized per channel with the `mean`/`std`
> in `assets/era5_stats/`, i.e. `(physical − mean) / std`. Physical-unit input
> produces wrong-scale forecasts that compound over the rollout.

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
    --remove_channels q2m d2m sst ws100m u100m v100m lcc mcc hcc tcc ssr ssrd fdir ttr tcw tp \
    --save_dir out/forecast
```
Each rollout step writes `out/forecast/<lead_hour>h.nc` (in physical units).

## Notes
- `--remove_channels` and `--lead_step_hours` **must match training** — the
  encoder's mean/std buffers and step conditioning depend on them. The list above
  is the set this checkpoint was trained with (85 → 70 evaluated channels).
- Input channel order must match the ERA5 buffer order exactly (no auto-reorder).
- `--dtype bf16` (default) is stable; RoPE buffers self-heal at load.

Run `--help` for all options.
