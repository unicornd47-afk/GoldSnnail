//! Differentiable LIF cell (surrogate gradient) — Block 0.
//!
//! The forward pass fires a **hard** spike (binary `Θ(v − θ)`); the backward
//! pass flows gradient through the spike using a smooth **surrogate**
//! derivative. This is the standard straight-through / surrogate-gradient
//! trick that makes a spiking neuron trainable with BPTT.
//!
//! Surrogate: arctan — `S(v−θ) = atan(π·β·(v−θ))/π + 1/2`, with derivative
//! `S'(v−θ) = β / (1 + (π·β·(v−θ))²)`. `β` controls slope (β=10 default).
//!
//! Deps: none (std only). Autodiff (`candle`) arrives in Block 2/3 for the full
//! recurrent BPTT loop; this cell proves the surrogate math first, verified
//! against finite differences.

/// Smooth arctan firing surrogate `S(x) ∈ (0,1)` (for the soft/test forward).
#[inline]
pub fn surrogate_fire(x: f32, beta: f32) -> f32 {
    (std::f32::consts::PI * beta * x).atan() / std::f32::consts::PI + 0.5
}

/// Derivative of the arctan surrogate: `S'(x) = β / (1 + (π·β·x)²)`.
#[inline]
pub fn surrogate_grad(x: f32, beta: f32) -> f32 {
    let px = std::f32::consts::PI * beta * x;
    beta / (1.0 + px * px)
}

/// Hard spike: `Θ(x) = 1 if x ≥ 0 else 0`.
#[inline]
pub fn hard_spike(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

/// A single recurrent LIF layer, one timestep at a time.
///
/// Dynamics per neuron (Euler, no adaptation for v1):
///   `v ← v + dt/τ · (i_syn − v)`
///   `s ← Θ(v − θ)`
///   `v ← v·(1−s) + v_reset·s`   (hard reset)
pub struct LifCell {
    /// Membrane time constant (learnable, > 0).
    pub tau: f32,
    /// Spike threshold.
    pub v_thresh: f32,
    /// Reset potential after a spike.
    pub v_reset: f32,
    /// Surrogate slope.
    pub beta: f32,
    /// Integration timestep.
    pub dt: f32,
    /// Membrane potential state (len = n_out).
    pub v: Vec<f32>,
}

impl LifCell {
    pub fn new(n_out: usize, tau: f32, v_thresh: f32) -> Self {
        Self {
            tau,
            v_thresh,
            v_reset: 0.0,
            beta: 10.0,
            dt: 1.0,
            v: vec![0.0; n_out],
        }
    }

    /// Leak coefficient `k = dt/τ`.
    #[inline]
    pub fn leak_k(&self) -> f32 {
        self.dt / self.tau.max(1e-6)
    }

    /// Forward one step (hard spike + hard reset).
    ///
    /// Returns `(spikes, v_pre)` where `v_pre` is the membrane *before* the
    /// spike/reset (needed by [`Self::backward`]).
    pub fn forward(&mut self, i_syn: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let k = self.leak_k();
        let n = self.v.len();
        assert_eq!(i_syn.len(), n, "i_syn must match cell width");
        let mut spikes = vec![0.0f32; n];
        let mut v_pre = vec![0.0f32; n];
        for j in 0..n {
            self.v[j] += k * (i_syn[j] - self.v[j]);
            v_pre[j] = self.v[j];
            let s = hard_spike(self.v[j] - self.v_thresh);
            spikes[j] = s;
            self.v[j] = self.v[j] * (1.0 - s) + self.v_reset * s;
        }
        (spikes, v_pre)
    }

    /// Backward one step, using the surrogate derivative through the spike.
    ///
    /// Given the upstream gradient wrt spikes `g_spikes`, returns
    /// `(g_i_syn, g_v_prev)` — the gradients wrt the synaptic current and the
    /// previous membrane (for recurrence). The hard reset is a detached gate,
    /// so no gradient flows through the reset path.
    pub fn backward(&self, v_pre: &[f32], g_spikes: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let k = self.leak_k();
        let n = self.v.len();
        assert_eq!(v_pre.len(), n);
        assert_eq!(g_spikes.len(), n);
        let mut g_i = vec![0.0f32; n];
        let mut g_v = vec![0.0f32; n];
        for j in 0..n {
            let g_pre = g_spikes[j] * surrogate_grad(v_pre[j] - self.v_thresh, self.beta);
            g_i[j] = g_pre * k;
            g_v[j] = g_pre * (1.0 - k);
        }
        (g_i, g_v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_grad_matches_finite_difference() {
        let beta = 10.0f32;
        for x in [-1.0f32, -0.1, 0.0, 0.05, 0.3, 1.0] {
            let eps = 1e-4f32;
            let num =
                (surrogate_fire(x + eps, beta) - surrogate_fire(x - eps, beta)) / (2.0 * eps);
            let ana = surrogate_grad(x, beta);
            assert!((num - ana).abs() < 2e-3, "x={x}: num={num} ana={ana}");
        }
    }

    #[test]
    fn gradient_flows_through_spike() {
        let mut cell = LifCell::new(3, 2.0, 1.0);
        let i_syn = vec![1.5f32, 0.2, 2.0];
        let (spikes, v_pre) = cell.forward(&i_syn);
        assert!(spikes.iter().any(|&s| s > 0.0), "need at least one spike");

        let g_spikes = vec![1.0f32; 3];
        let (g_i, _g_v) = cell.backward(&v_pre, &g_spikes);

        let k = cell.leak_k();
        for j in 0..3 {
            let expected = surrogate_grad(v_pre[j] - cell.v_thresh, cell.beta) * k;
            assert!((g_i[j] - expected).abs() < 1e-6, "neuron {j} g_i mismatch");
            if spikes[j] > 0.0 {
                assert!(g_i[j].abs() > 1e-6, "spiked neuron {j} must carry gradient");
            }
        }
    }

    #[test]
    fn smooth_forward_gradient_matches_finite_difference() {
        // Single step, no reset, smooth firing — verifies the surrogate
        // derivative AND the chain rule through the membrane integration.
        let beta = 10.0f32;
        let theta = 1.0f32;
        let tau = 2.0f32;
        let dt = 1.0f32;
        let k = dt / tau;
        let w = [[0.5f32, -0.3, 1.2], [0.8, 0.4, -0.6]]; // 2 inputs × 3 neurons
        let x = [1.0f32, -0.5];
        let c = [1.0f32, 2.0, -1.5]; // loss weights

        let i_syn = |w: &[[f32; 3]; 2]| -> [f32; 3] {
            [
                x[0] * w[0][0] + x[1] * w[1][0],
                x[0] * w[0][1] + x[1] * w[1][1],
                x[0] * w[0][2] + x[1] * w[1][2],
            ]
        };
        let loss = |w: &[[f32; 3]; 2]| -> f32 {
            let i = i_syn(w);
            let mut l = 0.0;
            for j in 0..3 {
                let v_pre = k * i[j]; // v_prev = 0
                l += c[j] * surrogate_fire(v_pre - theta, beta);
            }
            l
        };

        for i in 0..2 {
            for j in 0..3 {
                let v_pre = k * i_syn(&w)[j];
                let ana = c[j] * surrogate_grad(v_pre - theta, beta) * k * x[i];
                let eps = 1e-4f32;
                let (mut wp, mut wm) = (w, w);
                wp[i][j] += eps;
                wm[i][j] -= eps;
                let num = (loss(&wp) - loss(&wm)) / (2.0 * eps);
                assert!((num - ana).abs() < 2e-3, "i={i} j={j}: num={num} ana={ana}");
            }
        }
    }
}
