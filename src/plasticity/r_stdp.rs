//! R-STDP (Reward-Modulated Spike-Timing-Dependent Plasticity)
//!
//! Δw = R · η · exp(-d_H(pre, post)) · f(Δt)
//!
//! Uses the 1-D Poincaré disc distance from `crate::geometry::poincare`.
//! All operations are elastic: no panics on degenerate inputs.

use crate::geometry::poincare::hyperbolic_distance;

/// Reward-modulated STDP learner.
///
/// * `lr` — base learning rate η.
/// * `tau` — time constant τ (ms) for the STDP window.
/// * `curvature` — Poincaré disc curvature (typically -1.0).
#[derive(Debug, Clone, Copy)]
pub struct RSTDP {
    pub lr: f64,
    pub tau: f64,
    pub curvature: f64,
}

impl RSTDP {
    pub fn new(lr: f64, tau: f64, curvature: f64) -> Self {
        Self { lr, tau, curvature }
    }

    /// Computes the weight delta for a single pre/post pair.
    ///
    /// * `reward` — scalar reward signal R.
    /// * `pre_time` — pre-synaptic spike time (ms).
    /// * `post_time` — post-synaptic spike time (ms).
    /// * `pre_embed` — pre-synaptic embedding in the 1-D Poincaré disc.
    /// * `post_embed` — post-synaptic embedding in the 1-D Poincaré disc.
    pub fn compute(
        &self,
        reward: f64,
        pre_time: f64,
        post_time: f64,
        pre_embed: f32,
        post_embed: f32,
    ) -> f64 {
        let dt = post_time - pre_time;
        let dist = hyperbolic_distance(pre_embed, post_embed) as f64;

        // STDP time window: exponential causal/anti-causal.
        let time_factor = if dt > 0.0 {
            (-dt / self.tau).exp()
        } else {
            -(-dt / self.tau).exp()
        };

        // Geometric proximity factor.
        let geo_factor = (-dist).exp();

        reward * self.lr * time_factor * geo_factor
    }

    /// Batch-update a flat weight slice for a set of pre-synaptic spikes.
    ///
    /// * `weights` — mutable flat weight slice (in-place).
    /// * `pre_spikes` — indices of pre-synaptic neurons that fired.
    /// * `pre_times` — spike times per neuron (flat slice, indexed by neuron id).
    /// * `post_time` — post-synaptic spike time.
    /// * `pre_embeds` — Poincaré embeddings per pre-synaptic neuron.
    /// * `post_embed` — post-synaptic embedding.
    /// * `reward` — scalar reward.
    pub fn update_weights(
        &self,
        weights: &mut [f64],
        pre_spikes: &[usize],
        pre_times: &[f64],
        post_time: f64,
        pre_embeds: &[f32],
        post_embed: f32,
        reward: f64,
    ) {
        for &pre_idx in pre_spikes {
            if pre_idx >= weights.len() {
                continue;
            }
            let _dt = post_time - pre_times[pre_idx];
            let dw = self.compute(reward, pre_times[pre_idx], post_time, pre_embeds[pre_idx], post_embed);
            weights[pre_idx] += dw;
            // Elastic hard bounds to prevent runaway.
            weights[pre_idx] = weights[pre_idx].clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_reward_potentiates() {
        let stdp = RSTDP::new(0.01, 20.0, -1.0);
        let dw = stdp.compute(1.0, 0.0, 5.0, 0.1, 0.11);
        assert!(dw > 0.0, "Pre-before-Post + positive reward = potentiation, got {}", dw);
    }

    #[test]
    fn negative_reward_depresses() {
        let stdp = RSTDP::new(0.01, 20.0, -1.0);
        let dw = stdp.compute(-1.0, 0.0, 5.0, 0.1, 0.11);
        assert!(dw < 0.0, "Negative reward = depression, got {}", dw);
    }

    #[test]
    fn far_embeddings_weak_update() {
        let stdp = RSTDP::new(0.01, 20.0, -1.0);
        let dw = stdp.compute(1.0, 0.0, 5.0, 0.0, 0.99);
        assert!(dw.abs() < 1e-4, "Far apart embeddings should give negligible update, got {}", dw);
    }
}
