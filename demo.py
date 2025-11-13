#!/usr/bin/env python3
"""
Demo script for lagrs - A Parallel Rust Time-Series Engine
"""

import numpy as np
import time

try:
    import lagrs
    print("✓ Successfully imported lagrs")
except ImportError as e:
    print(f"✗ Failed to import lagrs: {e}")
    print("  Make sure to run: maturin develop")
    exit(1)

# Test rolling mean
print("\n=== Testing rolling_mean ===")
n = 100_000
window = 50
data = np.random.randn(n).astype(np.float64)

print(f"Input: {n} values, window size: {window}")

start = time.time()
result = lagrs.rolling_mean(data, window)
elapsed = time.time() - start

print(f"Result length: {len(result)}")
print(f"Time: {elapsed:.4f} seconds")
print(f"First 10 non-NaN values: {result[window-1:window+9]}")

# Test ARIMA
print("\n=== Testing ARIMA ===")
model = lagrs.ARIMA(p=1, d=1, q=1)
print(f"Created ARIMA({model.p}, {model.d}, {model.q})")

# Generate some sample time series data
y = np.cumsum(np.random.randn(100)) + 10.0

print(f"Fitting ARIMA model to {len(y)} data points...")
result = model.fit(y)
print(f"Fitted parameters: {result['params']}")
print(f"AIC: {result['aic']}, BIC: {result['bic']}")

print("\n=== Testing forecast ===")
forecast = model.forecast(h=10)
print(f"Forecasted {len(forecast)} steps ahead: {forecast}")

print("\n✓ All tests passed!")

