//! Entropy & Thermodynamic Observables
//!
//! Read-only calculations over flat histograms or probability vectors.
//! All functions are pure (no mutation, no side-effects) and can safely
//! run concurrently with the integrator on a background thread or CUDA stream.
//!
//! # Elastic Contract
//!
//! - Zero or negative probability bins are skipped silently (no `log(0)` NaN).
//! - Distributions that do not sum to 1.0 are handled gracefully (re-normalised
//!   or clipped) without panicking.

// ============================================================================
// Shannon entropy
// ============================================================================

/// Shannon entropy H(P) in bits of a discrete distribution `p`.
///
/// `p` need not sum to 1.0 — the function re-normalises internally.
/// Negative or zero bins are skipped (elastic zero-contribution).
///
/// Returns `0.0` for a deterministic distribution, `log2(N)` for a uniform one.
pub fn shannon_entropy(p: &[f32]) -> f32 {
    let total: f32 = p.iter().filter(|&&x| x > 0.0).sum();
    if total < 1e-12 {
        return 0.0; // undefined / silent — elastic zero
    }
    let inv_total = 1.0 / total;
    let mut h = 0.0_f32;
    for &pk in p {
        if pk > 1e-12 {
            let p_norm = pk * inv_total;
            h -= p_norm * p_norm.log2();
        }
    }
    h
}

// ============================================================================
// Rényi entropy (generalisation of Shannon)
// ============================================================================

/// Rényi entropy of order `alpha` in bits.
///
/// `H_α(P) = 1/(1-α) · log2(Σ p_i^α)`
///
/// Special cases:
/// - α → 1: converges to Shannon entropy (approximated below).
/// - α = 0: Hartley entropy (log2 of support size).
/// - α = 2: collision entropy.
///
/// Elastic: α ≤ 0 returns Shannon entropy; negative bins skipped.
pub fn renyi_entropy(p: &[f32], alpha: f32) -> f32 {
    if alpha <= 0.0 || (alpha - 1.0).abs() < 1e-4 {
        // Degenerate or limit case — fall back to Shannon.
        return shannon_entropy(p);
    }

    let total: f32 = p.iter().filter(|&&x| x > 0.0).sum();
    if total < 1e-12 {
        return 0.0;
    }
    let inv_total = 1.0 / total;

    let sum_pow: f32 = p
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| (x * inv_total).powf(alpha))
        .sum();

    if sum_pow < 1e-12 {
        return 0.0;
    }

    sum_pow.log2() / (1.0 - alpha)
}

// ============================================================================
// Joint / conditional entropy proxy
// ============================================================================

/// Estimates the joint entropy H(X, Y) from two marginal distributions `p_x`
/// and `p_y` under the **independence assumption** H(X,Y) = H(X) + H(Y).
///
/// This is a fast proxy. When X and Y are not independent the result is an
/// upper bound. Use for criticality monitoring, not for rigorous information
/// theory calculations.
pub fn joint_entropy_proxy(p_x: &[f32], p_y: &[f32]) -> f32 {
    shannon_entropy(p_x) + shannon_entropy(p_y)
}

// ============================================================================
// Effective temperature
// ============================================================================

/// Effective (thermodynamic) temperature T_eff of a spike-rate histogram.
///
/// Computed via the Boltzmann analogy: `T_eff ∝ 1 / H(P)`.
/// A critical system maximises H(P) → low T_eff.
/// A highly ordered (or silent) system has low H → high T_eff (elastic ∞ avoided).
///
/// Returns a dimensionless `f32` in `[0, 1]` by mapping through `1/(1+H)`.
pub fn effective_temperature(p: &[f32]) -> f32 {
    let h = shannon_entropy(p);
    // Elastic: map to [0, 1] without hard inversion.
    1.0 / (1.0 + h)
}

// ============================================================================
// Lyapunov divergence proxy (sensitivity to initial conditions)
// ============================================================================

/// Estimates a Lyapunov divergence proxy from two membrane potential trajectories.
///
/// Given two trajectories `traj_a` and `traj_b` (same length), computes:
/// `λ_proxy = (1/T) · Σ log(|a_t - b_t| / δ₀)` where `δ₀` is the initial separation.
///
/// Elastic: bins where `|a - b| < ε` or `δ₀ < ε` contribute 0.0.
///
/// A positive result suggests sensitive dependence on initial conditions (chaos).
pub fn lyapunov_proxy(traj_a: &[f32], traj_b: &[f32]) -> f32 {
    if traj_a.len() < 2 || traj_a.len() != traj_b.len() {
        return 0.0;
    }

    let delta_0 = (traj_a[0] - traj_b[0]).abs().max(1e-10);
    let mut sum = 0.0_f32;
    let mut count = 0usize;

    for (&a, &b) in traj_a.iter().zip(traj_b.iter()).skip(1) {
        let delta = (a - b).abs();
        if delta > 1e-10 {
            sum += (delta / delta_0).ln();
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shannon_entropy_uniform() {
        // Uniform over 4 outcomes → H = log2(4) = 2.0 bits.
        let p = [0.25f32; 4];
        let h = shannon_entropy(&p);
        assert!((h - 2.0).abs() < 1e-5, "H = {h}");
    }

    #[test]
    fn shannon_entropy_deterministic() {
        // Point mass → H = 0.
        let p = [0.0f32, 1.0, 0.0, 0.0];
        let h = shannon_entropy(&p);
        assert!(h.abs() < 1e-6, "H = {h}");
    }

    #[test]
    fn shannon_entropy_empty_does_not_panic() {
        let h = shannon_entropy(&[0.0_f32; 10]);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn renyi_entropy_approaches_shannon_at_one() {
        let p = [0.1f32, 0.2, 0.4, 0.3];
        let h_sh = shannon_entropy(&p);
        let h_re = renyi_entropy(&p, 1.0001);
        assert!((h_sh - h_re).abs() < 0.05, "Rényi(1) ≈ Shannon");
    }

    #[test]
    fn effective_temperature_in_unit_interval() {
        let p = [0.25f32; 8];
        let t = effective_temperature(&p);
        assert!(t >= 0.0 && t <= 1.0, "T_eff must be in [0,1]");
    }
}
