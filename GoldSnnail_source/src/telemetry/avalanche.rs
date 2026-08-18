//! Avalanche Metrics
//!
//! Observes spike rasters (flat `u8` slices) and computes neuronal avalanche
//! statistics without mutating any simulation state. This module is **strictly
//! read-only** and can run in parallel with the integrator.
//!
//! # Criticality Indicators
//!
//! A system near the critical point (SOC) exhibits avalanche size distributions
//! that follow a power law: P(S) ∝ S^{-τ}, with τ ≈ 1.5 (branching process).
//!
//! This module computes:
//! - Avalanche sizes and durations.
//! - The branching ratio σ (mean number of secondary spikes per spike).
//! - A simple log-log slope estimate for P(S) as a criticality proxy.

/// Counts contiguous active regions ("avalanches") in a 1-D raster.
///
/// An avalanche is a maximal run of non-zero bins.
/// Returns the size (length) of each avalanche.
pub fn count_avalanches(raster: &[u8]) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut current = 0usize;
    for &active in raster {
        if active != 0 {
            current += 1;
        } else if current > 0 {
            sizes.push(current);
            current = 0;
        }
    }
    if current > 0 {
        sizes.push(current);
    }
    sizes
}

// ============================================================================
// Branching ratio
// ============================================================================

/// Estimates the branching ratio σ = Σ descendants / Σ ancestors.
///
/// Computed from a flat raster `[u8]` where each value represents the number
/// of spikes in one time bin. The branching ratio is the ratio of spikes in
/// bin `t+1` to bin `t`, averaged over all active bins.
///
/// A ratio near 1.0 indicates criticality. < 1 = sub-critical, > 1 = super-critical.
///
/// Returns `1.0` if the raster is empty or has only one bin (safe default).
pub fn branching_ratio(spike_counts: &[u32]) -> f32 {
    if spike_counts.len() < 2 {
        return 1.0;
    }

    let mut numerator = 0.0_f32;
    let mut denominator = 0.0_f32;

    for window in spike_counts.windows(2) {
        let ancestor = window[0] as f32;
        let descendant = window[1] as f32;
        if ancestor > 0.0 {
            numerator += descendant;
            denominator += ancestor;
        }
    }

    if denominator < 1.0 {
        1.0 // elastic default for sparse or silent networks
    } else {
        numerator / denominator
    }
}

// ============================================================================
// Log-log slope (power-law proxy)
// ============================================================================

/// Estimates the power-law exponent τ of the size distribution via OLS on
/// log-log transformed counts.
///
/// `sizes` is a slice of avalanche sizes (from `count_avalanches`).
///
/// Returns `(tau, r_squared)` where `tau` is the estimated exponent (negative)
/// and `r_squared` is the fit quality ∈ [0, 1].
///
/// Elastic: returns `(-1.5, 0.0)` if fewer than 2 unique sizes are observed.
pub fn power_law_slope(sizes: &[usize]) -> (f32, f32) {
    if sizes.len() < 2 {
        return (-1.5, 0.0);
    }

    // Build a frequency histogram.
    let max_size = *sizes.iter().max().unwrap_or(&1);
    let mut hist = vec![0u32; max_size + 1];
    for &s in sizes {
        hist[s] += 1;
    }

    // Collect (log(size), log(count)) pairs for non-zero bins.
    let mut log_s: Vec<f32> = Vec::new();
    let mut log_c: Vec<f32> = Vec::new();
    for (s, &c) in hist.iter().enumerate().skip(1) {
        if c > 0 {
            log_s.push((s as f32).ln());
            log_c.push((c as f32).ln());
        }
    }

    let n = log_s.len() as f32;
    if n < 2.0 {
        return (-1.5, 0.0);
    }

    // OLS: slope = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
    let sum_x: f32 = log_s.iter().sum();
    let sum_y: f32 = log_c.iter().sum();
    let sum_xx: f32 = log_s.iter().map(|x| x * x).sum();
    let sum_xy: f32 = log_s.iter().zip(log_c.iter()).map(|(x, y)| x * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return (-1.5, 0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // R²
    let mean_y = sum_y / n;
    let ss_tot: f32 = log_c.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f32 = log_s
        .iter()
        .zip(log_c.iter())
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();

    let r2 = if ss_tot > 1e-10 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (slope, r2.clamp(0.0, 1.0))
}

// ============================================================================
// Lifetime histogram
// ============================================================================

/// Computes the lifetime (duration) distribution of avalanches.
///
/// Returns a `Vec<usize>` where each element is the duration of one avalanche
/// measured in time bins. This is the duration analogue of the size distribution.
pub fn avalanche_durations(raster: &[u8]) -> Vec<usize> {
    // For a flat raster, duration = length of the active run = size.
    // A 2-D raster (time × neurons) would yield different measures.
    // For now this is equivalent to `count_avalanches` but kept separate for
    // semantic clarity and future extension.
    count_avalanches(raster)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_avalanches_basic() {
        let raster = [0u8, 1, 1, 0, 1, 0, 1, 1, 1, 0];
        let av = count_avalanches(&raster);
        assert_eq!(av, vec![2, 1, 3]);
    }

    #[test]
    fn count_avalanches_all_active() {
        let raster = [1u8; 8];
        let av = count_avalanches(&raster);
        assert_eq!(av, vec![8]);
    }

    #[test]
    fn count_avalanches_all_silent() {
        let av = count_avalanches(&[0u8; 10]);
        assert!(av.is_empty());
    }

    #[test]
    fn branching_ratio_critical_approximate() {
        // Equal counts in every bin → σ = 1.0.
        let counts = vec![10u32; 100];
        let sigma = branching_ratio(&counts);
        assert!((sigma - 1.0).abs() < 0.01, "σ = {sigma}");
    }

    #[test]
    fn power_law_slope_returns_finite() {
        let sizes = vec![1, 1, 2, 3, 1, 5, 2, 8];
        let (tau, r2) = power_law_slope(&sizes);
        assert!(tau.is_finite(), "tau must be finite");
        assert!(r2 >= 0.0 && r2 <= 1.0, "r2 must be in [0,1]");
    }
}
