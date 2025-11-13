use lagrs_core::rolling::rolling_mean;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::time::Instant;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Generate test data
        let n = 2_000_000;
        let window = 50;
        
        println!("Generating {} random values...", n);
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64).sin() + (i as f64 * 0.1).cos())
            .collect();
        
        // Benchmark Rust implementation
        println!("\n=== Rust lagrs rolling_mean ===");
        let start = Instant::now();
        let rust_result = rolling_mean(&data, window);
        let rust_duration = start.elapsed();
        println!("Time: {:?}", rust_duration);
        println!("Result length: {}", rust_result.len());
        println!("First few non-NaN values: {:?}", 
                 rust_result.iter()
                     .skip(window - 1)
                     .take(5)
                     .collect::<Vec<_>>());
        
        // Benchmark Python statsmodels
        println!("\n=== Python statsmodels rolling_mean ===");
        let start = Instant::now();
        
        // Import statsmodels
        let filtertools = PyModule::import_bound(py, "statsmodels.tsa.filters.filtertools")?;
        
        // Create numpy array from Rust data
        let py_array = PyArray1::from_vec_bound(py, data.clone());
        
        // Call roll_mean function
        let roll_mean_func = filtertools.getattr("roll_mean")?;
        let py_result = roll_mean_func.call1((py_array, window))?;
        let python_duration = start.elapsed();
        
        println!("Time: {:?}", python_duration);
        
        // Extract result length if it's a numpy array
        if let Ok(len) = py_result.getattr("__len__")?.call0()?.extract::<usize>() {
            println!("Result length: {}", len);
        }
        
        // Performance comparison
        println!("\n=== Performance Comparison ===");
        println!("Rust lagrs:    {:?}", rust_duration);
        println!("Python statsmodels: {:?}", python_duration);
        
        if rust_duration.as_secs_f64() > 0.0 && python_duration.as_secs_f64() > 0.0 {
            let speedup = python_duration.as_secs_f64() / rust_duration.as_secs_f64();
            println!("Speedup: {:.2}x", speedup);
        }
        
        Ok(())
    })
}

