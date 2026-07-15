# assets

- `era5_stats/` — ERA5 normalization statistics (per-channel mean/std, latitude
  weights, static fields) required by weather forecasting via
  `--era5_data_path assets/era5_stats`. ~24 MB; derived statistics only, no ERA5
  reanalysis data. See [docs/WEATHER.md](../docs/WEATHER.md).
