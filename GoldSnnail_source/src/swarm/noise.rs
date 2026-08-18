//! Stochastic Noise Injection
//!
//! All noise generators operate on flat slices; all seeds are explicit (no
//! global RNG state). Each function is `#[inline]` so the compiler can fuse it
//! into the integrator sweep when called in a tight loop.
//!
//! # Elastic Contract
//!
//! Invalid distribution parameters (e.g., σ ≤ 0, λ ≤ 0) are silently replaced
//! with safe defaults — **no panic, no error**. This keeps the noise injection
//! path from becoming an interruption in the hot loop.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};

// ============================================================================
// Gaussian white noise
// ============================================================================

/// Adds i.i.d. Gaussian white noise N(μ, σ²) in-place to every element of `buf`.
///
/// Elastic: if `sigma <= 0`, the distribution silently collapses to zero noise
/// (μ additive, no stochasticity). The seed is consumed locally; no global state.
#[inline]
pub fn add_gaussian_white(buf: &mut [f32], mu: f32, sigma: f32, seed: u64) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    // Elastic: fall back to zero-sigma if sigma is invalid.
    let dist = Normal::new(mu, sigma.max(f32::EPSILON))
        .unwrap_or_else(|_| Normal::new(0.0, f32::EPSILON).unwrap());
    for v in buf.iter_mut() {
        *v += dist.sample(&mut rng);
    }
}

// ============================================================================
// Ornstein-Uhlenbeck process (coloured noise)
// ============================================================================

/// Advances a flat array of Ornstein-Uhlenbeck processes by one time step.
///
/// The OU SDE is: `dx = θ(μ − x) dt + σ dW`
///
/// Approximated via Euler-Maruyama:
/// `x_new = x + θ(μ − x) dt + σ√dt · N(0,1)`
///
/// Arguments:
/// - `state` — current OU process values (mut). Updated in-place.
/// - `mu` — long-run mean.
/// - `theta` — mean-reversion speed (> 0). Elastic: clamped to 1e-6 if ≤ 0.
/// - `sigma` — diffusion coefficient (> 0). Elastic: clamped to ε if ≤ 0.
/// - `dt` — time step.
/// - `seed` — RNG seed for this step. Increment each tick for independent draws.
#[inline]
pub fn step_ornstein_uhlenbeck(
    state: &mut [f32],
    mu: f32,
    theta: f32,
    sigma: f32,
    dt: f32,
    seed: u64,
) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let theta_safe = theta.max(1e-6_f32);
    let sigma_safe = sigma.max(f32::EPSILON);
    let sqrt_dt = dt.max(0.0).sqrt();

    let dist = Normal::new(0.0_f32, 1.0_f32).unwrap();
    for x in state.iter_mut() {
        let dw = dist.sample(&mut rng);
        *x += theta_safe * (mu - *x) * dt + sigma_safe * sqrt_dt * dw;
    }
}

// ============================================================================
// Poisson spike-rate noise (stochastic external input)
// ============================================================================

/// Injects Poisson-distributed spike counts as additive current into `i_ext`.
///
/// Each neuron receives a current proportional to a draw from Poisson(λ · dt),
/// scaled by `weight`. This models a Poisson-rate background input.
///
/// Elastic: if `rate_hz <= 0` no noise is injected (rate clamped to 0).
#[inline]
pub fn inject_poisson_current(
    i_ext: &mut [f32],
    rate_hz: f32,
    dt_s: f32,
    weight: f32,
    seed: u64,
) {
    let lambda = (rate_hz * dt_s).max(0.0);
    if lambda < 1e-9 {
        return; // nothing to inject
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    // Poisson::new requires lambda > 0.
    let dist = Poisson::new(lambda as f64)
        .unwrap_or_else(|_| Poisson::new(1e-9).unwrap());
    for i in i_ext.iter_mut() {
        let k = dist.sample(&mut rng) as f32;
        *i += weight * k;
    }
}

// ============================================================================
// Uniform jitter (for delay randomisation)
// ============================================================================

/// Fills `jitter` with uniform random values in `[lo, hi]`.
///
/// Used to randomise synaptic delays slightly at construction time.
/// Elastic: if `hi <= lo`, fills with `lo`.
#[inline]
pub fn uniform_jitter(jitter: &mut [f32], lo: f32, hi: f32, seed: u64) {
    if hi <= lo {
        jitter.fill(lo);
        return;
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let dist = Uniform::new(lo, hi).unwrap_or(Uniform::new_inclusive(lo, lo).unwrap());
    for v in jitter.iter_mut() {
        *v = dist.sample(&mut rng);
    }
}
