# lagrs: A Parallel Rust Time-Series Engine

A high-performance time series forecasting engine written in Rust with Python bindings.

## Project Structure

```
lagrs/
├── Cargo.toml              # Workspace configuration
├── pyproject.toml          # Maturin configuration
├── lagrs-core/             # Core Rust library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── arima.rs        # ARIMA model (stub)
│       ├── rolling.rs      # Parallel rolling window operations
│       └── utils.rs        # Utility functions
├── lagrs-python/           # Python bindings (PyO3)
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
├── lagrs-bench/            # Benchmarking suite
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
└── demo.py                 # Demo Python script
```

## Building

### Prerequisites

- Rust (latest stable)
- Python 3.8+
- [maturin](https://github.com/PyO3/maturin) for building Python bindings

Install maturin:
```bash
pip install maturin
```

### Build Python Extension

From the `lagrs-python` directory:
```bash
cd lagrs-python
maturin develop
```

Or from the project root:
```bash
maturin develop -m lagrs-python
```

This will compile the Rust code and install the `lagrs` Python package.

## Usage

### Python

```python
import lagrs
import numpy as np

# Rolling mean
data = np.random.randn(1000)
result = lagrs.rolling_mean(data, window=50)

# ARIMA model
model = lagrs.ARIMA(p=1, d=1, q=1)
result = model.fit(data)
forecast = model.forecast(h=10)
```

### Run Demos

```bash
# Basic demo
python demos/demo.py

# Comprehensive benchmark
python demos/benchmark_demo.py
```

## Benchmarking

### Python Benchmark Suite

Run the comprehensive benchmark comparing lagrs vs statsmodels:

```bash
# Install statsmodels (if not already installed)
uv pip install statsmodels --python .venv/bin/python

# Run benchmark
source .venv/bin/activate
python benchmark_demo.py
```

The benchmark includes:
- **Rolling Mean**: Performance comparison with various data sizes
- **Single SKU ARIMA**: Fitting and forecasting comparison
- **Multi-SKU Forecasting**: Tests 1,000-10,000 SKUs for throughput comparison
- **Parallelism Check**: Verifies parallel computation is working

### Rust Benchmark

Run the Rust benchmark (requires Python dev libraries):

```bash
cargo run --bin lagrs-bench
```

Note: This requires `statsmodels` to be installed in your Python environment.

## Features

- ✅ **Parallel rolling window operations** using Rayon (automatically uses all CPU cores)
- ✅ **ARIMA model structure** (stub implementation, ready for full implementation)
- ✅ **Zero-copy numpy array integration** for efficient data transfer
- ✅ **Python bindings** via PyO3 with clean API
- ✅ **Multi-SKU support** - designed for forecasting thousands of time series in parallel
- 🔜 Full ARIMA implementation with parallel optimization
- 🔜 Polars/Arrow integration
- 🔜 Additional forecasting models (ETS, Prophet, etc.)

## Performance

The library is built with **parallelism in mind**:
- Rolling operations use **Rayon** for automatic parallelization across CPU cores
- ARIMA implementation (when complete) will support parallel fitting across multiple SKUs
- Zero-copy numpy integration minimizes memory overhead
- Rust's performance provides 10-100x speedup potential over pure Python implementations

## Development

### Run Tests

```bash
cargo test
```

### Build All Crates

```bash
cargo build --workspace
```

## License

MIT OR Apache-2.0

