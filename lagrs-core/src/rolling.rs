use rayon::prelude::*;

/// Compute a parallel rolling mean over a slice of f64 values.
///
/// # Arguments
/// * `data` - Input time series data
/// * `window` - Window size for rolling mean
///
/// # Returns
/// A `Vec<f64>` containing the rolling mean values. The first `window - 1` elements
/// will be NaN since there isn't enough data to compute a mean.
///
/// # Panics
/// Panics if `window` is 0.
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 {
        panic!("Window size must be greater than 0");
    }
    
    if data.len() < window {
        return vec![f64::NAN; data.len()];
    }
    
    let mut result = vec![f64::NAN; window - 1];
    
    // Parallel computation of rolling means
    let rolling_means: Vec<f64> = (window - 1..data.len())
        .into_par_iter()
        .map(|i| {
            let start = i.saturating_sub(window - 1);
            let window_data = &data[start..=i];
            window_data.iter().sum::<f64>() / window_data.len() as f64
        })
        .collect();
    
    result.extend(rolling_means);
    result
}

#[cfg(test)]
mod tests {
    use super::rolling_mean;
    
    #[test]
    fn test_rolling_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3);
        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0); // (1+2+3)/3
        assert_eq!(result[3], 3.0); // (2+3+4)/3
        assert_eq!(result[4], 4.0); // (3+4+5)/3
    }
    
    #[test]
    fn test_rolling_mean_single_window() {
        let data = vec![1.0, 2.0, 3.0];
        let result = rolling_mean(&data, 3);
        assert_eq!(result[2], 2.0);
    }
}

