/// ARIMA model result structure
#[derive(Debug, Clone)]
pub struct ARIMAResult {
    /// Fitted parameters: [AR params (p), MA params (q)]
    pub params: Vec<f64>,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Residuals from fitting
    pub residuals: Vec<f64>,
    /// Fitted values
    pub fitted_values: Vec<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
}

use std::cell::RefCell;
use rayon::prelude::*;

/// ARIMA (AutoRegressive Integrated Moving Average) model
#[derive(Debug, Clone)]
pub struct ARIMA {
    /// AR order (p)
    pub p: usize,
    /// Differencing order (d)
    pub d: usize,
    /// MA order (q)
    pub q: usize,
    /// Fitted AR parameters (using RefCell for interior mutability)
    ar_params: RefCell<Option<Vec<f64>>>,
    /// Fitted MA parameters
    ma_params: RefCell<Option<Vec<f64>>>,
    /// Original series (for undifferencing)
    original_series: RefCell<Option<Vec<f64>>>,
    /// Differenced series
    differenced_series: RefCell<Option<Vec<f64>>>,
    /// Residuals for forecasting
    residuals: RefCell<Option<Vec<f64>>>,
    /// Variance of residuals
    variance: RefCell<Option<f64>>,
}

impl ARIMA {
    /// Create a new ARIMA model with specified orders
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        ARIMA {
            p,
            d,
            q,
            ar_params: RefCell::new(None),
            ma_params: RefCell::new(None),
            original_series: RefCell::new(None),
            differenced_series: RefCell::new(None),
            residuals: RefCell::new(None),
            variance: RefCell::new(None),
        }
    }
    
    /// Apply differencing to make series stationary
    fn diff(&self, y: &[f64]) -> Vec<f64> {
        let mut result = y.to_vec();
        for _ in 0..self.d {
            let mut diffed = Vec::with_capacity(result.len().saturating_sub(1));
            for i in 1..result.len() {
                diffed.push(result[i] - result[i - 1]);
            }
            result = diffed;
        }
        result
    }
    
    /// Undifference to get back to original scale
    fn undiff(&self, diffed: &[f64], original: &[f64]) -> Vec<f64> {
        if diffed.is_empty() {
            return vec![];
        }
        
        let mut result = diffed.to_vec();
        for _ in 0..self.d {
            let mut undiffed = Vec::with_capacity(result.len());
            let start_idx = original.len().saturating_sub(result.len() + 1);
            let mut last_val = if start_idx < original.len() {
                original[start_idx]
            } else {
                0.0
            };
            
            for i in 0..result.len() {
                last_val = last_val + result[i];
                undiffed.push(last_val);
            }
            result = undiffed;
        }
        result
    }
    
    /// Calculate log-likelihood for given parameters
    fn log_likelihood(&self, z: &[f64], ar_params: &[f64], ma_params: &[f64]) -> f64 {
        let n = z.len();
        let max_lag = self.p.max(self.q);
        
        if n <= max_lag {
            return f64::NEG_INFINITY;
        }
        
        // Calculate residuals
        let mut residuals = vec![0.0; n];
        let mut z_pred = vec![0.0; n];
        
        // Initialize early values
        for i in 0..max_lag {
            z_pred[i] = z[i];
            residuals[i] = 0.0;
        }
        
        // Calculate residuals iteratively
        for i in max_lag..n {
            let mut pred = 0.0;
            
            // AR component
            for j in 0..self.p {
                if i > j {
                    pred += ar_params[j] * z[i - j - 1];
                }
            }
            
            // MA component
            for j in 0..self.q {
                if i > j {
                    pred += ma_params[j] * residuals[i - j - 1];
                }
            }
            
            z_pred[i] = pred;
            residuals[i] = z[i] - pred;
        }
        
        // Calculate variance and log-likelihood
        let effective_n = n - max_lag;
        let sse: f64 = residuals[max_lag..].iter().map(|r| r * r).sum();
        let variance = sse / effective_n as f64;
        
        if variance <= 0.0 {
            return f64::NEG_INFINITY;
        }
        
        // Log-likelihood: -n/2 * ln(2πσ²) - 1/(2σ²) * Σε²
        let log_likelihood = -0.5 * effective_n as f64 * 
            (2.0 * std::f64::consts::PI * variance).ln() - 
            sse / (2.0 * variance);
        
        log_likelihood
    }
    
    /// Fit the ARIMA model using Maximum Likelihood Estimation (MLE)
    pub fn fit(&self, y: &[f64]) -> ARIMAResult {
        if y.len() < self.p + self.q + self.d + 10 {
            return ARIMAResult {
                params: vec![0.0; self.p + self.q],
                aic: f64::INFINITY,
                bic: f64::INFINITY,
                residuals: vec![],
                fitted_values: vec![],
                log_likelihood: f64::NEG_INFINITY,
            };
        }
        
        // Store original series
        *self.original_series.borrow_mut() = Some(y.to_vec());
        
        // Apply differencing
        let z = self.diff(y);
        *self.differenced_series.borrow_mut() = Some(z.clone());
        
        if z.len() < self.p + self.q + 5 {
            return ARIMAResult {
                params: vec![0.0; self.p + self.q],
                aic: f64::INFINITY,
                bic: f64::INFINITY,
                residuals: vec![],
                fitted_values: vec![],
                log_likelihood: f64::NEG_INFINITY,
            };
        }
        
        // Initial parameter estimates using least squares
        let initial_params = self.initial_estimate(&z);
        
        // Optimize using gradient descent (simplified MLE)
        let optimized_params = self.optimize_mle(&z, &initial_params);
        
        // Split parameters
        let ar_params: Vec<f64> = optimized_params[..self.p].to_vec();
        let ma_params: Vec<f64> = optimized_params[self.p..].to_vec();
        
        *self.ar_params.borrow_mut() = Some(ar_params.clone());
        *self.ma_params.borrow_mut() = Some(ma_params.clone());
        
        // Calculate final residuals and fitted values
        let (fitted, residuals) = self.calculate_fitted_and_residuals(&z, &ar_params, &ma_params);
        *self.residuals.borrow_mut() = Some(residuals.clone());
        
        // Calculate variance
        let effective_n = residuals.len() - self.p.max(self.q);
        let sse: f64 = residuals[self.p.max(self.q)..].iter().map(|r| r * r).sum();
        let variance = sse / effective_n as f64;
        *self.variance.borrow_mut() = Some(variance);
        
        // Calculate log-likelihood
        let log_likelihood = self.log_likelihood(&z, &ar_params, &ma_params);
        
        // Calculate AIC and BIC
        let n_obs = effective_n as f64;
        let n_params = (self.p + self.q) as f64;
        let aic = -2.0 * log_likelihood + 2.0 * n_params;
        let bic = -2.0 * log_likelihood + n_params * n_obs.ln();
        
        // Combine params
        let mut all_params = ar_params;
        all_params.extend(ma_params);
        
        ARIMAResult {
            params: all_params,
            aic,
            bic,
            residuals,
            fitted_values: fitted,
            log_likelihood,
        }
    }
    
    /// Initial parameter estimate using least squares
    fn initial_estimate(&self, z: &[f64]) -> Vec<f64> {
        let n = z.len();
        let max_lag = self.p.max(self.q);
        let start_idx = max_lag;
        
        let mut x_matrix = Vec::new();
        let mut y_vec = Vec::new();
        
        for i in start_idx..n {
            let mut row = Vec::new();
            
            // AR terms
            for j in 1..=self.p {
                row.push(z[i - j]);
            }
            
            // MA terms (using past values as proxy initially)
            for j in 1..=self.q {
                if i >= j {
                    row.push(z[i - j]);
                } else {
                    row.push(0.0);
                }
            }
            
            x_matrix.push(row);
            y_vec.push(z[i]);
        }
        
        self.least_squares(&x_matrix, &y_vec)
    }
    
    /// Optimize parameters using gradient descent for MLE
    fn optimize_mle(&self, z: &[f64], initial: &[f64]) -> Vec<f64> {
        let mut params = initial.to_vec();
        let learning_rate = 0.01;
        let max_iterations = 100;
        let tolerance = 1e-6;
        
        let mut prev_ll = self.log_likelihood(z, 
            &params[..self.p], 
            &params[self.p..]);
        
        for iteration in 0..max_iterations {
            // Calculate gradient (numerical approximation)
            let mut gradient = vec![0.0; params.len()];
            let epsilon = 1e-5;
            
            for i in 0..params.len() {
                let mut params_plus = params.clone();
                params_plus[i] += epsilon;
                
                let ll_plus = self.log_likelihood(z,
                    &params_plus[..self.p],
                    &params_plus[self.p..]);
                
                gradient[i] = (ll_plus - prev_ll) / epsilon;
            }
            
            // Update parameters
            for i in 0..params.len() {
                params[i] += learning_rate * gradient[i];
                
                // Apply constraints (AR/MA parameters should be in reasonable range)
                if i < self.p {
                    // AR: typically between -1 and 1 for stationarity
                    params[i] = params[i].max(-0.99).min(0.99);
                } else {
                    // MA: typically between -1 and 1 for invertibility
                    params[i] = params[i].max(-0.99).min(0.99);
                }
            }
            
            let current_ll = self.log_likelihood(z,
                &params[..self.p],
                &params[self.p..]);
            
            // Check convergence
            if (current_ll - prev_ll).abs() < tolerance {
                break;
            }
            
            prev_ll = current_ll;
            
            // Reduce learning rate if not improving
            if iteration > 10 && current_ll < prev_ll {
                // Learning rate already applied, continue
            }
        }
        
        params
    }
    
    /// Simple least squares solver
    fn least_squares(&self, x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        let m = if n > 0 { x[0].len() } else { 0 };
        
        if n < m {
            return vec![0.0; m];
        }
        
        let mut xtx = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    xtx[i][j] += x[k][i] * x[k][j];
                }
            }
        }
        
        let mut xty = vec![0.0; m];
        for i in 0..m {
            for k in 0..n {
                xty[i] += x[k][i] * y[k];
            }
        }
        
        self.solve_linear_system(&xtx, &xty)
    }
    
    /// Solve linear system Ax = b using Gaussian elimination
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        
        for i in 0..n {
            let mut max_row = i;
            let mut max_val = a[i][i].abs();
            for k in (i + 1)..n {
                if a[k][i].abs() > max_val {
                    max_val = a[k][i].abs();
                    max_row = k;
                }
            }
            
            if max_val < 1e-10 {
                return vec![0.0; n];
            }
            
            if max_row != i {
                a.swap(i, max_row);
                b.swap(i, max_row);
            }
            
            for k in (i + 1)..n {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
        
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = b[i];
            for j in (i + 1)..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }
        
        x
    }
    
    /// Calculate fitted values and residuals
    fn calculate_fitted_and_residuals(
        &self,
        z: &[f64],
        ar_params: &[f64],
        ma_params: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = z.len();
        let max_lag = self.p.max(self.q);
        let mut fitted = vec![0.0; n];
        let mut residuals = vec![0.0; n];
        
        for i in 0..max_lag {
            fitted[i] = z[i];
            residuals[i] = 0.0;
        }
        
        for i in max_lag..n {
            let mut pred = 0.0;
            
            for j in 0..self.p {
                if i > j {
                    pred += ar_params[j] * z[i - j - 1];
                }
            }
            
            for j in 0..self.q {
                if i > j {
                    pred += ma_params[j] * residuals[i - j - 1];
                }
            }
            
            fitted[i] = pred;
            residuals[i] = z[i] - pred;
        }
        
        (fitted, residuals)
    }
    
    /// Forecast future values
    pub fn forecast(&self, steps: usize) -> Vec<f64> {
        let ar_params_opt = self.ar_params.borrow();
        let ma_params_opt = self.ma_params.borrow();
        let diff_series_opt = self.differenced_series.borrow();
        let residuals_opt = self.residuals.borrow();
        
        if ar_params_opt.is_none() || ma_params_opt.is_none() {
            return vec![0.0; steps];
        }
        
        let ar_params = ar_params_opt.as_ref().unwrap().clone();
        let ma_params = ma_params_opt.as_ref().unwrap().clone();
        let z = diff_series_opt.as_ref().unwrap().clone();
        let residuals = residuals_opt.as_ref().unwrap().clone();
        
        let n = z.len();
        let mut forecast_diff = Vec::with_capacity(steps);
        let mut z_extended = z.clone();
        let mut res_extended = residuals.clone();
        
        for h in 0..steps {
            let mut pred = 0.0;
            
            for j in 0..self.p {
                let idx = n + h - j - 1;
                if idx < z_extended.len() {
                    pred += ar_params[j] * z_extended[idx];
                }
            }
            
            for j in 0..self.q {
                let idx = n + h - j - 1;
                if idx < res_extended.len() {
                    pred += ma_params[j] * res_extended[idx];
                }
            }
            
            forecast_diff.push(pred);
            z_extended.push(pred);
            res_extended.push(0.0);
        }
        
        let original_opt = self.original_series.borrow();
        if let Some(original) = original_opt.as_ref() {
            self.undiff(&forecast_diff, original)
        } else {
            forecast_diff
        }
    }
}

/// Batch fit multiple ARIMA models in parallel
pub fn batch_fit_arima(
    sku_data: &[Vec<f64>],
    p: usize,
    d: usize,
    q: usize,
) -> Vec<ARIMAResult> {
    sku_data
        .par_iter()
        .map(|y| {
            let model = ARIMA::new(p, d, q);
            model.fit(y)
        })
        .collect()
}

/// Batch forecast multiple ARIMA models in parallel
/// Note: Models must be fitted before calling this
pub fn batch_forecast_arima(
    models: Vec<ARIMA>,
    steps: usize,
) -> Vec<Vec<f64>> {
    models
        .into_par_iter()
        .map(|model| model.forecast(steps))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::ARIMA;
    
    #[test]
    fn test_arima_creation() {
        let model = ARIMA::new(1, 1, 1);
        assert_eq!(model.p, 1);
        assert_eq!(model.d, 1);
        assert_eq!(model.q, 1);
    }
    
    #[test]
    fn test_differencing() {
        let model = ARIMA::new(0, 1, 0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let diffed = model.diff(&data);
        assert_eq!(diffed, vec![1.0, 1.0, 1.0, 1.0]);
    }
    
    #[test]
    fn test_arima_fit() {
        let model = ARIMA::new(1, 1, 1);
        let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1 + (i as f64).sin()).collect();
        let result = model.fit(&data);
        assert_eq!(result.params.len(), 2);
        assert!(!result.aic.is_nan());
        assert!(!result.log_likelihood.is_infinite() || result.log_likelihood.is_finite());
    }
    
    #[test]
    fn test_arima_forecast() {
        let model = ARIMA::new(1, 1, 1);
        let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1 + (i as f64).sin()).collect();
        let _result = model.fit(&data);
        let forecast = model.forecast(10);
        assert_eq!(forecast.len(), 10);
    }
}
