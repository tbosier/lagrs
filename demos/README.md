# Lagrs Demos

This directory contains demonstration scripts for the lagrs time series engine.

## Scripts

### `demo.py`
Basic demonstration of lagrs functionality:
- Rolling mean computation
- ARIMA model fitting and forecasting
- Simple usage examples

**Run:**
```bash
source ../.venv/bin/activate
python demo.py
```

### `benchmark_demo.py`
Comprehensive benchmark comparing lagrs vs statsmodels:
- Rolling mean performance at different scales
- Single SKU ARIMA comparison
- Multi-SKU parallel batch processing
- Performance metrics and speedup calculations

**Run:**
```bash
source ../.venv/bin/activate
python benchmark_demo.py
```

## Requirements

- Python 3.10+
- numpy
- statsmodels (for benchmarks)
- lagrs (built and installed)

