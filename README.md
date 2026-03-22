# PyTorchProject_TimeSeries

Small sandbox repository for time-series experiments with PyTorch and Chronos.

## What's inside

- `main.py`: quick sine/cosine Plotly example that exports `sin_cos_plot.html`.
- `synthetic_experiment/data_sources.py`: synthetic sine-wave dataset and PyTorch `DataLoader`.
- `synthetic_experiment/draft_chronos_2.py`: draft Chronos-2 forecasting flow on generated data.
- `synthetic_experiment/utils_plots.py`: helper for forecast visualization.

## Quick start

```bash
uv sync
uv run main.py
uv run synthetic_experiment/data_sources.py
```

This repo is currently exploratory, with some experiment files intentionally left as drafts/placeholders.
