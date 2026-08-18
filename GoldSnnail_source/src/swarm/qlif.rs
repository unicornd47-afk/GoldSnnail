//! QLIF (Quaternionic Leaky Integrate-and-Fire) Kernel
//!
//! Operates directly on flat `ndarray` views from `StateArena`. One call
//! integrates the entire population for a single time step.
//!
//! # Integration Model
//!
//! The QLIF model extends the classic LIF with:
//! - A **quaternionic phase** that rotates each tick (coupling membrane dynamics
//!   to orientation in the Poincaré-ball embedding).
//! - **Spike-frequency adaptation** (slow K⁺ current): raised on each spike,
//!   decays exponentially between spikes.
//! - **Elastic clamping** of `v_m` via `geometry::poincare::elastic_clamp`
//!   so that extreme input currents never produce NaN/Inf.
//!
//! # Elastic Contract
//!
//! This module never panics during integration. Degenerate inputs (NaN
//! currents, zero tau) are absorbed by the elastic clamp and saturating
//! arithmetic.

use ndarray::{ArrayViewMut1, ArrayView1};
use crate::geometry::poincare::elastic_clamp;
use crate::geometry::quaternion::{hamilton_product, normalize_soft};

// ============================================================================
// Configuration
// ============================================================================

/// Parameters for the QLIF population integrator.
#[derive(Debug, Clone, Copy)]
pub struct QlifParams {
    /// Spike threshold (same units as `v_m`). Default: `1.0`.
    pub v_thresh: f32,
    /// Reset potential after a spike. Default: `0.0`.
    pub v_reset: f32,
    /// Refractory period in ticks. Default: `5`.
    pub refract_ticks: u16,
    /// Adaptation increment: amount added to `adapt` on each spike. Default: `0.1`.
    pub adapt_increment: f32,
    /// Adaptation time-constant (ticks). Decay: `adapt *= exp(-dt / tau_adapt)`. Default: `100.0`.
    pub tau_adapt: f32,
    /// Angular velocity of the quaternion rotation per tick (radians). Default: `0.01`.
    pub omega: f32,
}

impl Default for QlifParams {
    fn default() -> Self {
        Self {
            v_thresh: 1.0,
            v_reset: 0.0,
            refract_ticks: 5,
            adapt_increment: 0.1,
            tau_adapt: 100.0,
            omega: 0.01,
        }
    }
}

// ============================================================================
// Euler integration step
// ============================================================================

/// Single-step Euler integration for the QLIF model.
///
/// Mutates `v_m`, `phase`, `adapt`, `refract`, and `quat` in-place.
/// Writes `1` into `spike_out[i]` for every neuron that fired this tick.
///
/// # Arguments
///
/// * `v_m` — membrane potential (mut, length N)
/// * `phase` — oscillation phase (mut, length N)
/// * `adapt` — adaptation variable (mut, length N)
/// * `refract` — refractory countdown (mut, length N)
/// * `tau` — per-neuron time constants (immutable, length N)
/// * `i_ext` — external input current (immutable, length N)
/// * `quat` — quaternion state as a row-major N×4 flat slice (mut)
/// * `spike_out` — output raster: `1` if fired, `0` otherwise (mut, length N)
/// * `dt` — global time step (ms)
/// * `params` — QLIF parameter set
pub fn step_euler(
    v_m: &mut ArrayViewMut1<'_, f32>,
    phase: &mut ArrayViewMut1<'_, f32>,
    adapt: &mut ArrayViewMut1<'_, f32>,
    refract: &mut ndarray::ArrayViewMut1<'_, u16>,
    tau: &ArrayView1<'_, f32>,
    i_ext: &ArrayView1<'_, f32>,
    quat_flat: &mut [f32],   // length N * 4, row-major
    spike_out: &mut [u8],    // length N
    dt: f32,
    params: &QlifParams,
) {
    let n = v_m.len();
    // Adaptation decay factor per tick.
    let adapt_decay = (-dt / params.tau_adapt).exp();
    // Small angle quaternion representing one tick of rotation about Z-axis.
    let half_omega = params.omega * dt * 0.5;
    let delta_q = [half_omega.cos(), 0.0_f32, 0.0_f32, half_omega.sin()];
    let mut tmp_q = [0.0_f32; 4];

    for i in 0..n {
        // ── Refractory guard ─────────────────────────────────────────────────
        if refract[i] > 0 {
            refract[i] = refract[i].saturating_sub(1);
            spike_out[i] = 0;
            continue;
        }

        // ── Adaptation decay ─────────────────────────────────────────────────
        adapt[i] *= adapt_decay;

        // ── Membrane dynamics (Euler) ─────────────────────────────────────────
        // τ·dv/dt = -v_m + i_ext - adapt   →   Δv = dt/τ·(i_ext - v_m - adapt)
        let tau_safe = if tau[i].abs() < 1e-6 { 1e-6 } else { tau[i] };
        let dv = (i_ext[i] - v_m[i] - adapt[i]) / tau_safe * dt;
        // Elastic clamp: map through tanh to prevent unbounded growth.
        v_m[i] = elastic_clamp(v_m[i] + dv);

        // ── Phase advance ─────────────────────────────────────────────────────
        phase[i] = (phase[i] + dt) % std::f32::consts::TAU;

        // ── Quaternion rotation ───────────────────────────────────────────────
        let q_row = &mut quat_flat[i * 4..(i + 1) * 4];
        hamilton_product(q_row, &delta_q, &mut tmp_q);
        q_row.copy_from_slice(&tmp_q);
        normalize_soft(q_row);

        // ── Spike detection ───────────────────────────────────────────────────
        if v_m[i] >= params.v_thresh {
            v_m[i] = params.v_reset;
            adapt[i] += params.adapt_increment;
            refract[i] = params.refract_ticks;
            spike_out[i] = 1;
        } else {
            spike_out[i] = 0;
        }
    }
}

// ============================================================================
// RK4 integration step (higher-accuracy alternative)
// ============================================================================

/// Fourth-order Runge-Kutta integration for a single neuron.
///
/// Returns the updated `(v_m, phase)` without mutating the input. The caller
/// should write the results back and handle spiking / refractory as in `step_euler`.
///
/// Provided as a drop-in replacement for numerically stiff populations.
#[inline]
pub fn rk4_single(
    v: f32,
    phi: f32,
    tau: f32,
    i_ext: f32,
    adapt: f32,
    dt: f32,
) -> (f32, f32) {
    let tau_safe = if tau.abs() < 1e-6 { 1e-6 } else { tau };
    let dv = |v: f32| (i_ext - v - adapt) / tau_safe;
    let dphi = |_: f32| 1.0_f32; // constant phase velocity

    let k1v = dv(v);        let k1p = dphi(phi);
    let k2v = dv(v + dt * k1v * 0.5); let k2p = dphi(phi + dt * k1p * 0.5);
    let k3v = dv(v + dt * k2v * 0.5); let k3p = dphi(phi + dt * k2p * 0.5);
    let k4v = dv(v + dt * k3v);       let k4p = dphi(phi + dt * k3p);

    let v_new = elastic_clamp(v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v));
    let phi_new = (phi + dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p))
        % std::f32::consts::TAU;
    (v_new, phi_new)
}
