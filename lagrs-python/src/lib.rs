use lagrs_core::{arima::{ARIMA, ARIMAResult, batch_fit_arima}, rolling::rolling_mean};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compute a rolling mean over a time series using parallel computation.
///
/// # Arguments
/// * `series` - Input time series as a numpy array or Python list
/// * `window` - Window size for rolling mean
///
/// # Returns
/// A numpy array containing the rolling mean values
#[pyfunction]
#[pyo3(name = "rolling_mean")]
fn rolling_mean_py(
    series: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<PyArray1<f64>>> {
    let array = series.as_array();
    let data = array.as_slice().unwrap();
    
    let result = rolling_mean(data, window);
    
    Ok(PyArray1::from_vec_bound(series.py(), result))
}

/// ARIMA model for time series forecasting
#[pyclass(name = "ARIMA")]
struct PyARIMA {
    inner: ARIMA,
}

#[pymethods]
impl PyARIMA {
    /// Create a new ARIMA model
    ///
    /// # Arguments
    /// * `p` - AR order
    /// * `d` - Differencing order
    /// * `q` - MA order
    #[new]
    fn new(p: usize, d: usize, q: usize) -> Self {
        PyARIMA {
            inner: ARIMA::new(p, d, q),
        }
    }
    
    /// Get the AR order (p)
    #[getter]
    fn p(&self) -> usize {
        self.inner.p
    }
    
    /// Get the differencing order (d)
    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }
    
    /// Get the MA order (q)
    #[getter]
    fn q(&self) -> usize {
        self.inner.q
    }
    
    /// Fit the ARIMA model to the provided time series
    ///
    /// # Arguments
    /// * `y` - Time series data as a numpy array or Python list
    ///
    /// # Returns
    /// A dictionary containing fitted parameters and diagnostics
    fn fit(&self, y: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
        let array = y.as_array();
        let data = array.as_slice().unwrap();
        
        let result = self.inner.fit(data);
        
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("params", result.params)?;
            dict.set_item("aic", result.aic)?;
            dict.set_item("bic", result.bic)?;
            dict.set_item("log_likelihood", result.log_likelihood)?;
            Ok(dict.into())
        })
    }
    
    /// Forecast future values
    ///
    /// # Arguments
    /// * `h` - Number of steps ahead to forecast
    ///
    /// # Returns
    /// A numpy array of forecasted values
    fn forecast(&self, h: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let result = self.inner.forecast(h);
            let array = PyArray1::from_vec_bound(py, result);
            Ok(array.into())
        })
    }
}

/// Batch fit multiple ARIMA models in parallel
///
/// # Arguments
/// * `sku_data` - List of time series (each as numpy array)
/// * `p` - AR order
/// * `d` - Differencing order
/// * `q` - MA order
///
/// # Returns
/// List of ARIMAResult dictionaries
#[pyfunction]
#[pyo3(name = "batch_fit_arima")]
fn batch_fit_arima_py(
    sku_data: Vec<PyReadonlyArray1<f64>>,
    p: usize,
    d: usize,
    q: usize,
) -> PyResult<Vec<PyObject>> {
    // Convert Python arrays to Rust Vec<Vec<f64>>
    let rust_data: Vec<Vec<f64>> = sku_data
        .iter()
        .map(|arr| {
            let array = arr.as_array();
            array.as_slice().unwrap().to_vec()
        })
        .collect();
    
    // Fit in parallel
    let results = batch_fit_arima(&rust_data, p, d, q);
    
    // Convert back to Python objects
    Python::with_gil(|py| {
        let mut py_results = Vec::new();
        for result in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("params", result.params)?;
            dict.set_item("aic", result.aic)?;
            dict.set_item("bic", result.bic)?;
            dict.set_item("log_likelihood", result.log_likelihood)?;
            py_results.push(dict.into());
        }
        Ok(py_results)
    })
}

/// lagrs: A Parallel Rust Time-Series Engine
#[pymodule]
fn lagrs(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(batch_fit_arima_py, m)?)?;
    m.add_class::<PyARIMA>()?;
    Ok(())
}

