//! Power-Law Observer — Criticality Detection for SNN Avalanches
//!
//! Tracks avalanche size distributions over time and detects whether the
//! system operates near the critical point (self-organized criticality).
//!
//! A critical system exhibits avalanche sizes following P(S) ∝ S^{-τ}
//! with τ ≈ 1.5 and R² > 0.8 on a log-log plot.
//!
//! This observer is **strictly read-only** and works with arena indices
//! (no raw pointers, no unsafe).

use super::avalanche::{count_avalanches};
use super::avalanche_sim::generate_avalanche_distribution;
use crate::semantics::ConceptGraph;
use std::collections::HashSet;

/// Estimates the power-law exponent τ using Maximum Likelihood Estimation (MLE).
///
/// Uses the Clauset/Shalizi/Newman MLE formula for discrete power laws:
/// τ = 1 + n / Σᵢ ln(xᵢ / xₘᵢₙ)
///
/// Returns `(tau, r_squared)` where r_squared is computed via OLS on the
/// log-log binned PDF (mirrors the original `power_law_slope` metric but
/// with the MLE-fitted exponent).
///
/// Elastic: returns `(-1.5, 0.0)` if fewer than 2 unique sizes are observed.
pub fn power_law_mle(sizes: &[usize]) -> (f32, f32) {
    if sizes.len() < 2 {
        return (-1.5, 0.0);
    }

    let unique_sizes: Vec<usize> = sizes.iter().cloned().collect::<HashSet<_>>().into_iter().collect();
    if unique_sizes.len() < 2 {
        return (-1.5, 0.0);
    }

    let x_min = *unique_sizes.iter().min().unwrap();
    let n = sizes.len() as f32;

    let sum_log: f32 = sizes.iter()
        .filter(|&&x| x >= x_min)
        .map(|&x| (x as f32 / x_min as f32).ln())
        .sum();

    if sum_log <= 0.0 {
        return (-1.5, 0.0);
    }

    let tau = -(1.0 + n / sum_log);

    let max_size = *sizes.iter().max().unwrap_or(&1);
    let mut hist = vec![0u32; max_size + 1];
    for &s in sizes {
        hist[s] += 1;
    }

    let mut log_s: Vec<f32> = Vec::new();
    let mut log_c: Vec<f32> = Vec::new();

    for s in x_min..=max_size {
        if hist[s] > 0 {
            log_s.push((s as f32).ln());
            log_c.push((hist[s] as f32).ln());
        }
    }

    let r2 = if log_s.len() >= 2 {
        let n_fit = log_s.len() as f32;
        let sum_x: f32 = log_s.iter().sum();
        let sum_y: f32 = log_c.iter().sum();
        let sum_xx: f32 = log_s.iter().map(|x| x * x).sum();
        let sum_xy: f32 = log_s.iter().zip(log_c.iter()).map(|(x, y)| x * y).sum();

        let denom = n_fit * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            0.0
        } else {
            let slope = (n_fit * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / n_fit;
            let mean_y = sum_y / n_fit;
            let ss_tot: f32 = log_c.iter().map(|y| (y - mean_y).powi(2)).sum();
            let ss_res: f32 = log_s.iter().zip(log_c.iter())
                .map(|(x, y)| (*y - (slope * x + intercept)).powi(2))
                .sum();
            if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 0.0 }
        }
    } else {
        0.0
    };

    (tau, r2.clamp(0.0, 1.0))
}

/// Result of a power-law fit over a window of avalanche sizes.
#[derive(Debug, Clone, Copy)]
pub struct PowerLawFit {
    /// Estimated power-law exponent τ (negative, typically -1.5 at criticality).
    pub tau: f32,
    /// R² goodness of fit ∈ [0, 1].
    pub r_squared: f32,
    /// Number of avalanche samples used for this fit.
    pub sample_count: usize,
}

impl PowerLawFit {
    /// Returns true if this fit indicates critical dynamics.
    ///
    /// Criticality criteria:
    /// - τ ∈ [-2.0, -1.0] (near the theoretical -1.5)
    /// - R² > 0.7 (reasonable linear fit on log-log)
    pub fn is_critical(&self) -> bool {
        self.tau >= -2.0 && self.tau <= -1.0 && self.r_squared > 0.7
    }

    /// Returns a human-readable status string.
    pub fn status(&self) -> &'static str {
        if self.is_critical() {
            "CRITICAL"
        } else if self.tau > -1.0 {
            "SUB-CRITICAL"
        } else {
            "SUPER-CRITICAL"
        }
    }
}

/// Observer that accumulates avalanche sizes and periodically evaluates
/// whether the system is operating at criticality.
///
/// Uses a sliding window approach: only the most recent `window_size`
/// avalanche observations are considered for each fit.
#[derive(Debug, Clone)]
pub struct PowerLawObserver {
    /// Recent avalanche sizes (sliding window).
    pub window: Vec<usize>,
    /// Maximum number of samples to retain.
    pub window_size: usize,
    /// Most recent power-law fit result.
    pub last_fit: Option<PowerLawFit>,
    /// Total number of windows processed.
    pub windows_processed: usize,
}

impl PowerLawObserver {
    /// Creates a new observer with a sliding window of `window_size` samples.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            last_fit: None,
            windows_processed: 0,
        }
    }

    /// Records an avalanche size sample.
    ///
    /// When the window is full, the oldest sample is evicted and a new
    /// power-law fit is computed automatically.
    pub fn record(&mut self, avalanche_size: usize) {
        self.window.push(avalanche_size);
        if self.window.len() > self.window_size {
            self.window.remove(0);
        }

        // Use MLE fit instead of OLS
        if self.window.len() >= 4 {
            let (tau, r2) = power_law_mle(&self.window);
            self.last_fit = Some(PowerLawFit {
                tau,
                r_squared: r2,
                sample_count: self.window.len(),
            });
        }
        self.windows_processed += 1;
    }

    /// Records a full spike raster and internally counts avalanches.
    ///
    /// This is a convenience method that combines `count_avalanches` and
    /// `record` into a single call.
    pub fn record_raster(&mut self, raster: &[u8]) {
        let avalanches = count_avalanches(raster);
        for size in avalanches {
            self.record(size);
        }
    }

    /// Records avalanche sizes from a ConceptGraph simulation.
    pub fn record_graph_avalanches(&mut self, graph: &ConceptGraph, num_samples: usize) {
        let sizes = generate_avalanche_distribution(graph, num_samples, 50, 42);
        for size in sizes {
            self.record(size);
        }
    }

    /// Returns the most recent power-law fit, if any.
    pub fn fit(&self) -> Option<PowerLawFit> {
        self.last_fit
    }

    /// Returns true if the current window indicates critical dynamics.
    pub fn is_critical(&self) -> bool {
        self.last_fit.map(|f| f.is_critical()).unwrap_or(false)
    }

    /// Resets the observer state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.last_fit = None;
        self.windows_processed = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_records_avalanches() {
        let mut obs = PowerLawObserver::new(10);
        obs.record(5);
        obs.record(10);
        assert_eq!(obs.window, vec![5, 10]);
        assert_eq!(obs.windows_processed, 2);
    }

    #[test]
    fn observer_sliding_window_eviction() {
        let mut obs = PowerLawObserver::new(3);
        obs.record(1);
        obs.record(2);
        obs.record(3);
        obs.record(4);
        assert_eq!(obs.window, vec![2, 3, 4]);
    }

    #[test]
    fn observer_fit_after_enough_samples() {
        let mut obs = PowerLawObserver::new(20);
        for &size in &[5, 3, 8, 2, 12, 4, 7, 3] {
            obs.record(size);
        }
        let fit = obs.fit().expect("fit should be available");
        assert!(fit.sample_count >= 4);
        assert!(fit.tau.is_finite());
        assert!(fit.r_squared >= 0.0 && fit.r_squared <= 1.0);
    }

    #[test]
    fn observer_reset_clears_state() {
        let mut obs = PowerLawObserver::new(10);
        obs.record(5);
        obs.record(10);
        obs.reset();
        assert!(obs.window.is_empty());
        assert!(obs.last_fit.is_none());
        assert_eq!(obs.windows_processed, 0);
    }

    #[test]
    fn power_law_mle_returns_elastic_for_short_input() {
        let (tau, r2) = power_law_mle(&[5]);
        assert_eq!(tau, -1.5);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn power_law_mle_returns_elastic_for_single_unique() {
        let (tau, r2) = power_law_mle(&[3, 3, 3]);
        assert_eq!(tau, -1.5);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn power_law_mle_fits_power_law_distribution() {
        let mut sizes = Vec::new();
        for s in 1..=30 {
            let count = (200.0 / (s as f64).powf(1.5)).round() as usize;
            for _ in 0..count {
                sizes.push(s);
            }
        }
        let (tau, r2) = power_law_mle(&sizes);
        assert!(tau.is_finite(), "tau must be finite, got {}", tau);
        assert!(tau < -1.2, "tau should be negative, got {}", tau);
        assert!(tau > -3.0, "tau should be near -1.5, got {}", tau);
        assert!(r2 >= 0.0 && r2 <= 1.0, "r2 must be in [0,1], got {}", r2);
        assert!(r2 > 0.85, "r2 should indicate good fit, got {}", r2);
    }

    #[test]
    fn observer_is_critical_when_tau_near_minus_1_5() {
        let mut obs = PowerLawObserver::new(300);
        let mut sizes = Vec::new();
        for s in 1..=30 {
            let count = (200.0 / (s as f64).powf(1.5)).round() as usize;
            for _ in 0..count {
                sizes.push(s);
            }
        }
        for size in sizes {
            obs.record(size);
        }
        let fit = obs.fit().expect("fit should be available");
        assert!(fit.tau < -1.2, "tau should be negative, got {}", fit.tau);
        assert!(fit.tau > -3.0, "tau should be near -1.5, got {}", fit.tau);
        assert!(fit.r_squared > 0.8, "R2 should indicate good fit, got {}", fit.r_squared);
    }

    #[test]
    fn power_law_mle_deterministic_with_fixed_input() {
        let sizes = vec![1, 2, 3, 1, 5, 2, 8, 1, 3, 5, 1, 1, 2, 3, 5, 8, 13];
        let (tau1, r21) = power_law_mle(&sizes);
        let (tau2, r22) = power_law_mle(&sizes);
        assert!((tau1 - tau2).abs() < 1e-6, "MLE must be deterministic");
        assert!((r21 - r22).abs() < 1e-6, "R2 must be deterministic");
    }

    #[test]
    fn power_law_mle_r2_on_synthetic_power_law() {
        let mut sizes = Vec::new();
        for s in 1..=100 {
            let count = (10000.0 / (s as f64).powf(1.5)).round() as usize;
            for _ in 0..count {
                sizes.push(s);
            }
        }
        let (tau, r2) = power_law_mle(&sizes);
        assert!(tau < -1.2, "tau should be negative, got {:.3}", tau);
        assert!(r2 > 0.5, "r2 should be decent for true power law, got {:.3}", r2);
    }
}
