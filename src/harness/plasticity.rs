//! Plasticity Engine — Homeostatic Scaling + Batch R-STDP
//!
//! Extends the existing R-STDP rule with:
//! - Homeostatic scaling (target firing rate regulation)
//! - Structural pruning (remove weak synapses)
//! - Batch update over replay buffer samples

use crate::plasticity::RSTDP;
use crate::swarm::{Swarm, SwarmConfig};
use crate::harness::replay::Transition;

/// Plasticity configuration.
#[derive(Debug, Clone, Copy)]
pub struct PlasticityConfig {
    /// Base learning rate for R-STDP.
    pub stdp_lr: f64,
    /// Time constant for STDP window (ms).
    pub stdp_tau: f64,
    /// Poincaré disc curvature.
    pub curvature: f64,
    /// Target firing rate per neuron per episode (Hz-equivalent).
    pub target_firing_rate: f32,
    /// Enable homeostatic scaling.
    pub homeostasis: bool,
    /// Enable structural pruning (weights below threshold are zeroed).
    pub pruning: bool,
    /// Pruning threshold (absolute weight).
    pub prune_threshold: f32,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            stdp_lr: 0.01,
            stdp_tau: 20.0,
            curvature: -1.0,
            target_firing_rate: 5.0,
            homeostasis: true,
            pruning: true,
            prune_threshold: 0.02,
        }
    }
}

/// Plasticity engine combining R-STDP, homeostasis, and structural plasticity.
#[derive(Debug, Clone)]
pub struct PlasticityEngine {
    pub stdp: RSTDP,
    pub config: PlasticityConfig,
}

impl PlasticityEngine {
    /// Creates a new plasticity engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(PlasticityConfig::default())
    }

    /// Creates a plasticity engine with explicit configuration.
    pub fn with_config(config: PlasticityConfig) -> Self {
        let stdp = RSTDP::new(config.stdp_lr, config.stdp_tau, config.curvature);
        Self { stdp, config }
    }

    /// Applies homeostatic scaling to the swarm weight matrix.
    ///
    /// For each neuron, scales incoming weights to bring firing rate closer to target.
    pub fn apply_homeostasis(&self, swarm: &mut Swarm) {
        if !self.config.homeostasis {
            return;
        }

        let n = swarm.arena.membrane.len();
        let target = self.config.target_firing_rate;

        for post in 0..n {
            // Approximate actual firing rate from recent spike count
            // In a full implementation, we'd track per-neuron rates over a window.
            // Here we use a simplified proxy: number of active incoming synapses.
            let mut active_in = 0.0f32;
            for pre in 0..n {
                if swarm.weights.get(pre, post).abs() > 1e-6 {
                    active_in += 1.0;
                }
            }
            let actual_rate = active_in / n as f32 * 20.0; // rough scaling

            if actual_rate > 1e-6 {
                let scale = (target / actual_rate).clamp(0.5, 2.0);
                for pre in 0..n {
                    let idx = swarm.weights.index(pre, post);
                    swarm.weights.data[idx] *= scale;
                    swarm.weights.data[idx] = swarm.weights.data[idx].clamp(-1.0, 1.0);
                }
            }
        }
    }

    /// Applies structural pruning: zeros weights below threshold.
    pub fn apply_pruning(&self, swarm: &mut Swarm) {
        if !self.config.pruning {
            return;
        }
        let threshold = self.config.prune_threshold;
        let n = swarm.arena.membrane.len();
        for i in 0..n {
            for j in 0..n {
                let w = swarm.weights.get(i, j);
                if w.abs() < threshold {
                    swarm.weights.set(i, j, 0.0);
                }
            }
        }
    }

    /// Performs a single R-STDP update for a set of pre/post spike pairs.
    ///
    /// `pre_embed` and `post_embed` are 1-D Poincaré embeddings (f32).
    pub fn apply_stdp(
        &self,
        swarm: &mut Swarm,
        pre_spikes: &[usize],
        post_spikes: &[usize],
        reward: f64,
        pre_time: f64,
        post_time: f64,
        pre_embed: f32,
        post_embed: f32,
        lr: f64,
    ) {
        let n = swarm.arena.membrane.len();
        for &pre_idx in pre_spikes {
            if pre_idx >= n {
                continue;
            }
            for &post_idx in post_spikes {
                if post_idx >= n {
                    continue;
                }
                let dw = self.stdp.compute(reward, pre_time, post_time, pre_embed, post_embed);
                let idx = swarm.weights.index(pre_idx, post_idx);
                swarm.weights.data[idx] += (dw as f32) * lr as f32;
                swarm.weights.data[idx] = swarm.weights.data[idx].clamp(-1.0, 1.0);
            }
        }
    }

    /// Performs a full plasticity update step on a batch of transitions.
    ///
    /// This is the main training primitive called by the harness.
    pub fn update_batch(
        &self,
        swarm: &mut Swarm,
        transitions: &[&Transition],
        lr: f64,
    ) {
        if transitions.is_empty() {
            return;
        }

        // Simplified: use mean reward as global TD-proxy for STDP
        let mean_reward: f64 = transitions.iter().map(|t| t.reward).sum::<f64>() / transitions.len() as f64;
        let reward = mean_reward.clamp(-1.0, 1.0);

        // Collect all pre/post spikes from the batch
        let mut all_pre = Vec::new();
        let mut all_post = Vec::new();
        for t in transitions {
            all_pre.extend(t.output_spikes.iter().cloned());
            all_post.extend(t.output_spikes.iter().cloned());
        }

        // Apply STDP with batch-aggregated spikes
        if !all_pre.is_empty() && !all_post.is_empty() {
            let pre_time = 0.0;
            let post_time = 1.0;
            let pre_embed = 0.0f32;
            let post_embed = 0.0f32;
            self.apply_stdp(swarm, &all_pre, &all_post, reward, pre_time, post_time, pre_embed, post_embed, lr);
        }

        // Homeostasis and pruning
        self.apply_homeostasis(swarm);
        self.apply_pruning(swarm);
    }

    /// Returns the current number of active synapses (weight > threshold).
    pub fn count_active_synapses(&self, swarm: &Swarm) -> usize {
        let n = swarm.arena.membrane.len();
        let mut count = 0;
        for i in 0..n {
            for j in 0..n {
                if swarm.weights.get(i, j).abs() > self.config.prune_threshold {
                    count += 1;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plasticity_engine_creation() {
        let engine = PlasticityEngine::new();
        assert_eq!(engine.config.target_firing_rate, 5.0);
    }

    #[test]
    fn homeostasis_runs_without_panic() {
        let engine = PlasticityEngine::new();
        let mut swarm = Swarm::new(10, 2, 2);
        engine.apply_homeostasis(&mut swarm);
    }

    #[test]
    fn pruning_runs_without_panic() {
        let engine = PlasticityEngine::new();
        let mut swarm = Swarm::new(10, 2, 2);
        swarm.weights.set(0, 1, 0.5);
        engine.apply_pruning(&mut swarm);
        assert_eq!(swarm.weights.get(0, 1), 0.5); // above threshold
    }

    #[test]
    fn stdp_update_changes_weights() {
        let engine = PlasticityEngine::new();
        let mut swarm = Swarm::new(10, 2, 2);
        swarm.weights.set(0, 1, 0.1);
        let before = swarm.weights.get(0, 1);
        engine.apply_stdp(&mut swarm, &[0], &[1], 1.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        let after = swarm.weights.get(0, 1);
        assert!((after - before).abs() > 1e-10, "STDP should change weights");
    }
}
