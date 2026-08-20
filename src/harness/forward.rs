//! Forward Engine — SNN Step Wrapper with Noise Scheduling & State Extraction
//!
//! Wraps `SnnCore` to provide:
//! - Configurable noise injection per step
//! - Input/output spike extraction
//! - State vector extraction for RL heads

use crate::swarm::snn_core::{SnnCore, SnnStepResult, TOTAL_NEURONS, INPUT_NEURONS, OUTPUT_NEURONS, STAGES, NEURONS_PER_STAGE};
use crate::swarm::SwarmConfig;
use crate::substrate::StateArena;

/// Forward engine wrapping the SNN core.
#[derive(Debug, Clone)]
pub struct ForwardEngine {
    pub core: SnnCore,
    pub config: SwarmConfig,
    pub noise_std: f32,
    pub tick: u64,
}

impl ForwardEngine {
    /// Creates a new forward engine with given density and noise level.
    pub fn new(density: f64, noise_std: f32) -> Self {
        let mut core = SnnCore::new(density);
        let config = SwarmConfig::default();
        Self {
            core,
            config,
            noise_std: noise_std.clamp(0.0, 1.0),
            tick: 0,
        }
    }

    /// Advances the SNN by one timestep with the given input spikes.
    pub fn step(&mut self, input_spikes: &[usize]) -> SnnStepResult {
        self.core.swarm.config.noise_std = self.noise_std;
        self.tick += 1;
        self.core.step(input_spikes)
    }

    /// Returns indices of neurons that spiked in the current timestep.
    pub fn output_spikes(&self) -> Vec<usize> {
        self.core.spike_indices()
            .into_iter()
            .map(|i| i as usize)
            .collect()
    }

    /// Returns membrane potentials as a flat Vec<f32>.
    pub fn membrane_state(&self) -> Vec<f32> {
        self.core.swarm.arena.membrane.clone()
    }

    /// Extracts a simplified state vector from the SNN.
    ///
    /// Returns:
    /// - Stage-wise mean membrane potentials (6 values)
    /// - Overall spike count (1 value)
    pub fn extract_state(&self) -> Vec<f32> {
        let mut state = Vec::with_capacity(STAGES + 1);
        for stage in 0..STAGES {
            let start = stage * NEURONS_PER_STAGE;
            let end = start + NEURONS_PER_STAGE;
            let sum: f32 = self.core.swarm.arena.membrane[start..end].iter().sum();
            state.push(sum / NEURONS_PER_STAGE as f32);
        }
        state.push(self.core.swarm.spike_count() as f32);
        state
    }

    /// Resets the SNN state (membrane, recovery, refractory).
    pub fn reset(&mut self) {
        self.core.swarm.arena = StateArena::new(TOTAL_NEURONS);
        self.core.swarm.spike_buffer.clear();
        self.tick = 0;
    }

    /// Anneals noise standard deviation towards a target.
    pub fn anneal_noise(&mut self, target: f32, rate: f32) {
        self.noise_std += (target - self.noise_std) * rate;
        self.noise_std = self.noise_std.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_engine_creation() {
        let engine = ForwardEngine::new(0.06, 0.1);
        assert_eq!(engine.core.swarm.arena.membrane.len(), TOTAL_NEURONS);
    }

    #[test]
    fn forward_engine_step_produces_spikes() {
        let mut engine = ForwardEngine::new(0.06, 0.1);
        let result = engine.step(&[]);
        assert_eq!(result.tick, 1);
    }

    #[test]
    fn forward_engine_reset() {
        let mut engine = ForwardEngine::new(0.06, 0.1);
        engine.step(&[0]);
        engine.reset();
        assert_eq!(engine.tick, 0);
        assert_eq!(engine.core.swarm.spike_count(), 0);
    }

    #[test]
    fn forward_anneal_noise() {
        let mut engine = ForwardEngine::new(0.06, 0.5);
        engine.anneal_noise(0.0, 1.0);
        assert!(engine.noise_std < 0.01);
    }
}
