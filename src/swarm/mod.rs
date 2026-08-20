//! Swarm — QLIF Dynamics, Noise Injection & Async Spike Writing
//!
//! The swarm layer owns temporal evolution. It reads from `StateArena`,
//! applies QLIF dynamics with noise, and writes spikes into `SpikeBuffer`.

use crate::substrate::{SpikeBuffer, StateArena, WeightMatrix};

pub mod neuron;
pub mod snn_core;

/// Configuration for QLIF swarm dynamics.
#[derive(Debug, Clone, Copy)]
pub struct SwarmConfig {
    /// Membrane potential decay factor per timestep.
    pub decay: f32,
    /// Resting membrane potential (mV).
    pub resting_potential: f32,
    /// Standard deviation of Gaussian noise injected per timestep.
    pub noise_std: f32,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            decay: 0.95,
            resting_potential: -70.0,
            noise_std: 0.1,
        }
    }
}

/// QLIF swarm operating over flat state arenas.
#[derive(Debug, Clone)]
pub struct Swarm {
    /// Flat state arena for all neurons.
    pub arena: StateArena,
    /// Sparse weight matrix.
    pub weights: WeightMatrix,
    /// Spike output buffer for the current timestep.
    pub spike_buffer: SpikeBuffer,
    /// Swarm configuration.
    pub config: SwarmConfig,
}

impl Swarm {
    /// Creates a new swarm with pre-allocated capacity.
    ///
    /// * `capacity` — total number of neurons.
    /// * `input_size` — number of input neurons (top of the arena).
    /// * `output_size` — number of output neurons (bottom of the arena).
    pub fn new(capacity: usize, _input_size: usize, _output_size: usize) -> Self {
        let arena = StateArena::new(capacity);
        let weights = WeightMatrix::new(capacity, capacity);
        let spike_buffer = SpikeBuffer::new(capacity);
        let config = SwarmConfig::default();
        Self {
            arena,
            weights,
            spike_buffer,
            config,
        }
    }

    /// Advances the swarm by one timestep.
    ///
    /// # Steps
    ///
    /// 1. Decay membrane potentials and update recovery variables.
    /// 2. Inject input spikes and synaptic currents from the weight matrix.
    /// 3. Add Gaussian noise.
    /// 4. Detect spikes where membrane >= threshold.
    /// 5. Reset spiking neurons, set refractory periods, record spikes.
    pub fn step(&mut self, input_spikes: &[usize]) {
        let n = self.arena.membrane.len();
        let decay = self.config.decay;
        let rest = self.config.resting_potential;
        let noise_std = self.config.noise_std;

        // 1. Decay + noise (passive membrane dynamics)
        for i in 0..n {
            if self.arena.refractory[i] > 0 {
                self.arena.refractory[i] -= 1;
                self.arena.membrane[i] *= 0.5; // hard reset during refractory
                continue;
            }
            self.arena.membrane[i] = decay * self.arena.membrane[i] + rest * (1.0 - decay);
            self.arena.membrane[i] += (rand::random::<f32>() - 0.5) * 2.0 * noise_std;
        }

        // 2. Inject external input spikes
        for &neuron_idx in input_spikes {
            if neuron_idx < n {
                self.arena.membrane[neuron_idx] += 0.5; // excitatory boost
            }
        }

        // 3. Synaptic propagation from previous spikes (simple feed-forward)
        //    We use the weight matrix to compute input current for each neuron.
        //    This is O(n^2) but acceptable for 180 neurons.
        let mut synaptic_input = vec![0.0f32; n];
        for post in 0..n {
            let mut sum = 0.0;
            for pre in 0..n {
                let w = self.weights.get(pre, post);
                if w.abs() > 1e-6 && self.arena.refractory[pre] == 0 {
                    // Only propagate if pre neuron is above threshold (spiked last step)
                    // We approximate by checking if membrane is high
                    if self.arena.membrane[pre] > self.arena.threshold[pre] * 0.5 {
                        sum += w;
                    }
                }
            }
            synaptic_input[post] = sum;
        }
        for i in 0..n {
            self.arena.membrane[i] += synaptic_input[i] * 0.1;
        }

        // 4. Spike detection + recording
        self.spike_buffer.clear();
        for i in 0..n {
            if self.arena.refractory[i] == 0 && self.arena.membrane[i] >= self.arena.threshold[i] {
                self.spike_buffer.push(i as u32).ok();
                self.arena.membrane[i] = 0.0; // reset
                self.arena.refractory[i] = 3; // refractory period
                self.arena.recovery[i] += 0.1; // adaptation
            }
        }
    }

    /// Returns the number of spikes in the current timestep.
    pub fn spike_count(&self) -> usize {
        self.spike_buffer.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swarm_creation() {
        let swarm = Swarm::new(10, 2, 2);
        assert_eq!(swarm.arena.membrane.len(), 10);
        assert_eq!(swarm.spike_buffer.count, 10);
    }

    #[test]
    fn swarm_step_runs_without_panic() {
        let mut swarm = Swarm::new(8, 2, 2);
        swarm.step(&[]);
    }

    #[test]
    fn swarm_step_produces_spikes_with_strong_input() {
        let mut swarm = Swarm::new(10, 2, 2);
        // Boost neuron 0 strongly
        swarm.arena.membrane[0] = 1.0;
        swarm.step(&[]);
        assert!(swarm.spike_count() > 0);
    }

    #[test]
    fn swarm_refractory_prevents_immediate_respike() {
        let mut swarm = Swarm::new(10, 2, 2);
        swarm.arena.membrane[0] = 1.0;
        swarm.step(&[]);
        let first_count = swarm.spike_count();
        // Immediately step again without new input
        swarm.step(&[]);
        // Should have fewer spikes due to refractory period
        assert!(swarm.spike_count() <= first_count);
    }
}
