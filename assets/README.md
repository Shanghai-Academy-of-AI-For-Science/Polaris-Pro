# assets

- `era5_stats/` — ERA5 normalization statistics (per-channel mean/std, latitude
  weights, static fields) required by weather forecasting via
  `--era5_data_path assets/era5_stats`. ~24 MB; derived statistics only, no ERA5
  reanalysis data. See [docs/WEATHER.md](../docs/WEATHER.md).
- `benchmarks.png` — the benchmark chart referenced by the top-level `README.md`
  (add it here; also place a copy next to the model card in the weights repo).
