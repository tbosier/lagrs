"""
Unit tests for lagrs Python bindings
"""
import pytest
import numpy as np


def test_import():
    """Test that lagrs can be imported"""
    import lagrs
    assert lagrs is not None


def test_rolling_mean_basic():
    """Test basic rolling mean functionality"""
    import lagrs
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = lagrs.rolling_mean(data, window=3)
    
    assert len(result) == 5
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == 2.0  # (1+2+3)/3
    assert result[3] == 3.0  # (2+3+4)/3
    assert result[4] == 4.0  # (3+4+5)/3


def test_rolling_mean_large():
    """Test rolling mean with larger dataset"""
    import lagrs
    data = np.random.randn(1000).astype(np.float64)
    result = lagrs.rolling_mean(data, window=50)
    
    assert len(result) == 1000
    assert not np.all(np.isnan(result))  # Should have some valid values


def test_arima_creation():
    """Test ARIMA model creation"""
    import lagrs
    model = lagrs.ARIMA(p=1, d=1, q=1)
    
    assert model.p == 1
    assert model.d == 1
    assert model.q == 1


def test_arima_fit():
    """Test ARIMA model fitting"""
    import lagrs
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 10.0
    
    model = lagrs.ARIMA(p=1, d=1, q=1)
    result = model.fit(y)
    
    assert 'params' in result
    assert 'aic' in result
    assert 'bic' in result
    assert 'log_likelihood' in result
    assert len(result['params']) == 2  # p + q
    assert isinstance(result['aic'], (int, float))
    assert not np.isnan(result['aic'])


def test_arima_forecast():
    """Test ARIMA forecasting"""
    import lagrs
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 10.0
    
    model = lagrs.ARIMA(p=1, d=1, q=1)
    _ = model.fit(y)
    forecast = model.forecast(h=10)
    
    assert len(forecast) == 10
    assert all(isinstance(x, (int, float)) for x in forecast)


def test_batch_fit_arima():
    """Test parallel batch ARIMA fitting"""
    import lagrs
    np.random.seed(42)
    sku_data = [np.cumsum(np.random.randn(50)) + 10.0 for _ in range(10)]
    
    results = lagrs.batch_fit_arima(sku_data, p=1, d=1, q=1)
    
    assert len(results) == 10
    for result in results:
        assert 'params' in result
        assert 'aic' in result
        assert len(result['params']) == 2


def test_rolling_mean_edge_cases():
    """Test rolling mean edge cases"""
    import lagrs
    
    # Empty array
    data = np.array([], dtype=np.float64)
    result = lagrs.rolling_mean(data, window=5)
    assert len(result) == 0
    
    # Window larger than data
    data = np.array([1.0, 2.0], dtype=np.float64)
    result = lagrs.rolling_mean(data, window=5)
    assert len(result) == 2
    assert np.all(np.isnan(result))


def test_arima_edge_cases():
    """Test ARIMA edge cases"""
    import lagrs
    
    # Too little data
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    model = lagrs.ARIMA(p=1, d=1, q=1)
    result = model.fit(y)
    
    # Should handle gracefully (return inf AIC or similar)
    assert 'aic' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

