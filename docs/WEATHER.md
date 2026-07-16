# Weather forecasting

神珍 produces a global **ERA5 0.25° forecast** (721 × 1440 grid, 70 channels).
Given a single initial atmospheric state, it rolls the forecast forward
autoregressively: each step advances `--lead_step_hours` (6h) and feeds its own
output back in, until `--max_lead_hour` is reached. This modality has its own
script, `code/scripts/weather/infer_forecast.py`.

## Requirements
- `netCDF4`, `xarray`, `zarr` (in `requirements.txt`).
- **ERA5 statistics** (`--era5_data_path`): the per-channel mean/std, latitude
  weights, and static fields the pipeline needs. Bundled at `assets/era5_stats/`
  (~24 MB; derived statistics only, no ERA5 reanalysis data).
- **One input frame** (`--input`): see the exact format below.

## The `--input` file

A netCDF `.nc` file holding **one** atmospheric state. Concretely it must be a
DataArray with:

| aspect | requirement |
|--------|-------------|
| dims   | `(channel, lat, lon)` — or `(time, channel, lat, lon)`, in which case the last time step (or `--init_time`) is used |
| shape  | `(70, 721, 1440)` — the 70 evaluated channels on the global 0.25° grid |
| `channel` coord | the 70 channel **names in training order** (see below) — channels are matched positionally, the file must contain exactly these 70 in this order (no auto-drop or reorder) |
| `lat` / `lon` coords | 721 latitudes, 1440 longitudes of the ERA5 grid |
| dtype  | float; values **normalized per channel** (see the warning below) |

**The 70 channels, in order:** `z/t/u/v/q` each on 13 pressure levels
(`50,100,150,200,250,300,400,500,600,700,850,925,1000`), then the surface
fields `msl, t2m, ws10m, u10m, v10m`.

> ⚠️ **The input must be in normalized space, not physical units.** The encoder
> consumes normalized values and only its *output* is converted back to physical
> units (via `x · std + mean`). So the `--input` frame must first be standardized
> per channel using the `mean`/`std` in `assets/era5_stats/`, i.e.
> `(physical − mean) / std`. Feeding raw physical values (e.g. temperature in
> kelvin, geopotential in m²/s²) produces wrong-scale forecasts that compound
> over the rollout. If you export a frame straight from the ERA5 zarr the model
> was trained on, it is already normalized.

Minimal example of building a valid input frame from a normalized `[70,721,1440]`
NumPy array:

```python
import xarray as xr
CHANNELS = (["z%d"%l for l in (50,100,150,200,250,300,400,500,600,700,850,925,1000)]
          + ["t%d"%l for l in (50,100,150,200,250,300,400,500,600,700,850,925,1000)]
          + ["u%d"%l for l in (50,100,150,200,250,300,400,500,600,700,850,925,1000)]
          + ["v%d"%l for l in (50,100,150,200,250,300,400,500,600,700,850,925,1000)]
          + ["q%d"%l for l in (50,100,150,200,250,300,400,500,600,700,850,925,1000)]
          + ["msl","t2m","ws10m","u10m","v10m"])
xr.DataArray(arr, dims=("channel","lat","lon"),
             coords={"channel": CHANNELS, "lat": lats, "lon": lons}
             ).to_netcdf("init_frame.nc")
```

## Run
```bash
export PYTHONPATH=$PWD/code

python code/scripts/weather/infer_forecast.py \
    --checkpoint model \
    --era5_data_path assets/era5_stats \
    --input init_frame.nc \
    --init_time 2024-12-31T00:00:00 \
    --era5_image_size 721 1440 \
    --max_lead_hour 24 --lead_step_hours 6 \
    --remove_channels q2m d2m sst ws100m u100m v100m lcc mcc hcc tcc ssr ssrd fdir ttr tcw tp \
    --save_dir out/forecast
```

## Parameters

| Argument | Meaning |
|----------|---------|
| `--checkpoint` | The model directory (config + weights), e.g. `model`. |
| `--era5_data_path` | ERA5 statistics directory. Use the bundled `assets/era5_stats`. |
| `--input` | The initial-state `.nc` file described above. |
| `--init_time` | ISO timestamp of the input frame, e.g. `2024-12-31T00:00:00`. Sets the forecast's start time (used for the time-of-day / day-of-year conditioning). If omitted, the file's `time` coord or `time` attribute is used. |
| `--max_lead_hour` | How far ahead to forecast, in hours. `24` → forecast out to +24h; `240` → out to 10 days. |
| `--lead_step_hours` | Hours advanced per rollout step. **Must be 6** (the training interval). With `--max_lead_hour 24` this yields 4 steps (6/12/18/24h). |
| `--remove_channels` | The channels dropped at training time (85 → 70). **Must match training** — use the list shown above verbatim; the mean/std buffers depend on it. |
| `--era5_image_size` | Grid size `H W`; default `721 1440`. |
| `--save_channels` | Optional: save only these channels in each output file (e.g. `z500 t2m`). Default saves all 70. |
| `--dtype` | `bf16` (default) / `fp16` / `fp32`. bf16 is stable. |
| `--attn_implementation` | `flash_attention_2` (default) / `sdpa` / `eager`. Use `sdpa` or `eager` if flash-attn is not installed. |

Run `--help` for the full list.

## Output

Each rollout step writes one file `out/forecast/<lead_hour>h.nc`, **in physical
units** — e.g. `006h.nc`, `012h.nc`, … Each is a `(channel, lat, lon)` DataArray
with the same channel/lat/lon coords as the input (or only `--save_channels` if
set). `006h.nc` is the +6h forecast, `012h.nc` the +12h forecast, and so on.

## Notes
- On load you may see `[weather] broken decoder RoPE buffers detected … recomputing`.
  This is **normal**: the rotary-position tables are recomputable constants that
  aren't stored in the checkpoint, so the script rebuilds them at startup. It is
  a one-time info message, not an error.
