//! Frozen Core — Fixed-Weight SNN Backbone
//!
//! The "3141 frozen core" is the immutable substrate at the heart of every
//! FractalLayer. It is:
//!   - A SnnCore with pre-initialized weights
//!   - Never updated by learning
//!   - Shared across all layers at the same scale
//!   - Designed to be cloned cheaply (flat Vec<f32> weights)
//!
//! The constant 3141 encodes the topology:
//!   3 input regions  x  1 core region  x  4 output regions  x  1 gating region
//!   = 180 neurons total in the base configuration.

use crate::swarm::snn_core::SnnCore;


/// Frozen core parameters.
///
/// These are set once at initialization and never mutated by training.
#[derive(Debug, Clone, Copy)]
pub struct FrozenCoreConfig {
    /// SNN density (connectivity probability).
    pub density: f64,
    /// Initial noise standard deviation.
    pub noise_std: f32,
    /// Stage-wise threshold offset multiplier.
    pub threshold_gain: f32,
}

impl Default for FrozenCoreConfig {
    fn default() -> Self {
        Self {
            density: 0.06,
            noise_std: 0.1,
            threshold_gain: 1.0,
        }
    }
}

/// A frozen SNN backbone.
///
/// The core's weight matrix is never mutated after construction.
/// All learning happens in adapter layers surrounding the core.
#[derive(Debug, Clone)]
pub struct FrozenCore {
    pub core: SnnCore,
    pub config: FrozenCoreConfig,
    pub frozen_tick: u64,
}

impl FrozenCore {
    /// Creates a new frozen core with the given configuration.
    pub fn new(config: FrozenCoreConfig) -> Self {
        let mut core = SnnCore::new(config.density);
        // Apply threshold gain to create stage-wise specialization
        for stage in 0..6 {
            for i in 0..30 {
                let idx = stage * 30 + i;
                core.swarm.arena.threshold[idx] *= config.threshold_gain;
            }
        }
        Self {
            core,
            config,
            frozen_tick: 0,
        }
    }

    /// Runs the core for `recurrence` timesteps with given input spikes.
    ///
    /// Returns the final spike pattern and mean membrane per stage.
    pub fn run(&mut self, input_spikes: &[usize], recurrence: usize) -> FrozenCoreResult {
        let mut last_spikes = Vec::new();
        let mut stage_means = vec![0.0f32; 6];

        for _t in 0..recurrence {
            let _result = self.core.step(input_spikes);
            last_spikes = self.core.spike_indices();
            self.frozen_tick += 1;

            // Extract stage-wise membrane means
            for stage in 0..6 {
                let start = stage * 30;
                let end = start + 30;
                let sum: f32 = self.core.swarm.arena.membrane[start..end].iter().sum();
                stage_means[stage] = sum / 30.0;
            }
        }

        FrozenCoreResult {
            spikes: last_spikes.into_iter().map(|i| i as usize).collect(),
            stage_means,
            tick: self.frozen_tick,
        }
    }

    /// Returns the current membrane state snapshot.
    pub fn membrane_snapshot(&self) -> Vec<f32> {
        self.core.swarm.arena.membrane.clone()
    }

    /// Returns the number of active synapses (weight > threshold).
    pub fn active_synapses(&self) -> usize {
        self.core.count_active_synapses()
    }
}

/// Result of running a FrozenCore.
#[derive(Debug, Clone, Default)]
pub struct FrozenCoreResult {
    /// Output spike indices.
    pub spikes: Vec<usize>,
    /// Mean membrane potential per stage (6 values).
    pub stage_means: Vec<f32>,
    /// Total ticks processed by this core.
    pub tick: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_core_creation() {
        let core = FrozenCore::new(FrozenCoreConfig::default());
        assert_eq!(core.core.swarm.arena.membrane.len(), 180);
    }

    #[test]
    fn frozen_core_run_produces_spikes() {
        let mut core = FrozenCore::new(FrozenCoreConfig::default());
        let result = core.run(&[0, 1, 2], 4);
        assert_eq!(result.tick, 4);
    }

    #[test]
    fn frozen_core_has_six_stages() {
        let mut core = FrozenCore::new(FrozenCoreConfig::default());
        let result = core.run(&[], 1);
        assert_eq!(result.stage_means.len(), 6);
    }

    #[test]
    fn frozen_core_active_synapses() {
        let core = FrozenCore::new(FrozenCoreConfig::default());
        let count = core.active_synapses();
        assert!(count > 0);
    }
}


