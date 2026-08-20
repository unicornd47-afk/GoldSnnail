//! SNN Core — 180-Neuron QLIF Swarm with 6×30 Stages
//!
//! This is the concrete instantiation of the `Swarm` abstraction for the
//! GoldWorm desktop simulator. It fixes:
//!   - 180 neurons total
//!   - 6 stages × 30 neurons per stage
//!   - Stage-wise connectivity bias (feed-forward + lateral)
//!   - Serializable state for Tauri IPC
//!
//! The stage layout mirrors the pipeline pills in the UI:
//!   0: Sensor, 1: Attention, 2: Memory, 3: Compression, 4: World, 5: RL

use crate::substrate::{StateArena, WeightMatrix, SpikeBuffer, NeuronIdx, SpikeEvent};
use crate::swarm::{Swarm, SwarmConfig};

/// Fixed SNN topology constants.
pub const TOTAL_NEURONS: usize = 180;
pub const STAGES: usize = 6;
pub const NEURONS_PER_STAGE: usize = 30;
pub const INPUT_NEURONS: usize = 30;
pub const OUTPUT_NEURONS: usize = 30;

/// Stage names for UI rendering.
pub const STAGE_NAMES: [&str; STAGES] = [
    "Sensor",
    "Attention",
    "Memory",
    "Compression",
    "World",
    "RL",
];

/// Stage colors for UI rendering.
pub const STAGE_COLORS: [&str; STAGES] = [
    "#00d4ff",
    "#ff4d6d",
    "#2ecc71",
    "#a67cff",
    "#f1c40f",
    "#e67e22",
];

/// The SNN core engine.
#[derive(Debug, Clone)]
pub struct SnnCore {
    pub swarm: Swarm,
    pub tick: u64,
    pub density: f64,
}

impl SnnCore {
    /// Creates a new SNN core with 180 neurons in 6×30 stages.
    pub fn new(density: f64) -> Self {
        let mut swarm = Swarm::new(TOTAL_NEURONS, INPUT_NEURONS, OUTPUT_NEURONS);
        
        // Initialize thresholds with stage-wise variation
        for stage in 0..STAGES {
            for i in 0..NEURONS_PER_STAGE {
                let idx = stage * NEURONS_PER_STAGE + i;
                swarm.arena.threshold[idx] = -55.0 + (stage as f32) * 2.0;
            }
        }

        // Build sparse connectivity based on density and stage proximity
        build_connectivity(&mut swarm.weights, density);

        Self {
            swarm,
            tick: 0,
            density,
        }
    }

    /// Advances the SNN by one timestep.
    pub fn step(&mut self, input_spikes: &[usize]) -> SnnStepResult {
        self.tick += 1;
        self.swarm.step(input_spikes);

        let spike_count = self.swarm.spike_count();
        let active_synapses = self.count_active_synapses();
        let mean_weight = self.compute_mean_weight();

        SnnStepResult {
            tick: self.tick,
            spike_count,
            active_synapses,
            mean_weight,
            density: self.density,
        }
    }

    /// Returns the current spike buffer as a list of neuron indices.
    pub fn spike_indices(&self) -> Vec<u32> {
        self.swarm.spike_buffer.indices.clone()
    }

    /// Returns the number of active synapses (weight > threshold).
    pub fn count_active_synapses(&self) -> usize {
        let threshold = 0.05;
        let mut count = 0;
        for i in 0..TOTAL_NEURONS {
            for j in 0..TOTAL_NEURONS {
                if self.swarm.weights.get(i, j) > threshold {
                    count += 1;
                }
            }
        }
        count
    }

    /// Returns the mean weight value across all synapses.
    pub fn compute_mean_weight(&self) -> f32 {
        if self.swarm.weights.data.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.swarm.weights.data.iter().sum();
        sum / self.swarm.weights.data.len() as f32
    }

    /// Returns the stage index for a given neuron index.
    pub fn stage_of(neuron_idx: usize) -> usize {
        (neuron_idx / NEURONS_PER_STAGE).min(STAGES - 1)
    }

    /// Returns the position within the stage for a given neuron index.
    pub fn position_in_stage(neuron_idx: usize) -> usize {
        neuron_idx % NEURONS_PER_STAGE
    }
}

/// Result of a single SNN timestep.
#[derive(Debug, Clone)]
pub struct SnnStepResult {
    pub tick: u64,
    pub spike_count: usize,
    pub active_synapses: usize,
    pub mean_weight: f32,
    pub density: f64,
}

/// Builds sparse connectivity for the SNN weight matrix.
///
/// Connectivity rules:
/// - Within the same stage: density * 1.2
/// - Adjacent stages (feed-forward): density * 2.5
/// - Other stages: density * 0.12
fn build_connectivity(weights: &mut WeightMatrix, density: f64) {
    let n = TOTAL_NEURONS;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let si = SnnCore::stage_of(i);
            let sj = SnnCore::stage_of(j);
            let mut p = density as f32;
            if sj == si + 1 {
                p *= 2.5;
            } else if sj == si {
                p *= 1.2;
            } else if (sj as i32 - si as i32).abs() > 1 {
                p *= 0.12;
            }
            if rand::random::<f32>() < p {
                let w = 0.12 + rand::random::<f32>() * 0.35;
                weights.set(i, j, w);
            }
        }
    }
}

/// Serializes the SNN state for Tauri IPC.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnnStateDto {
    pub neurons: Vec<NeuronStateDto>,
    pub synapses: Vec<SynapseStateDto>,
    pub tick: u64,
    pub density: f64,
    pub pending_spikes: Vec<u32>,
}

/// Serialized neuron state for UI rendering.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuronStateDto {
    pub id: usize,
    pub stage: usize,
    pub x: f32,
    pub y: f32,
    pub v_m: f32,
    pub threshold: f32,
    pub refractory: u16,
    pub last_spike: i64,
}

/// Serialized synapse state for UI rendering.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynapseStateDto {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
    pub delay: u16,
}

impl From<&SnnCore> for SnnStateDto {
    fn from(core: &SnnCore) -> Self {
        let width = 800.0;
        let height = 600.0;
        let mut neurons = Vec::with_capacity(TOTAL_NEURONS);
        
        for s in 0..STAGES {
            for i in 0..NEURONS_PER_STAGE {
                let idx = s * NEURONS_PER_STAGE + i;
                let cx = width * (s as f32 + 0.5) / STAGES as f32;
                let cy = height * (0.12 + 0.76 * ((i as f32 + 0.5) / NEURONS_PER_STAGE as f32));
                neurons.push(NeuronStateDto {
                    id: idx,
                    stage: s,
                    x: cx + (rand::random::<f32>() - 0.5) * 0.55,
                    y: cy + (rand::random::<f32>() - 0.5) * 0.28,
                    v_m: core.swarm.arena.membrane[idx],
                    threshold: core.swarm.arena.threshold[idx],
                    refractory: core.swarm.arena.refractory[idx] as u16,
                    last_spike: 0,
                });
            }
        }

        let mut synapses = Vec::new();
        for i in 0..TOTAL_NEURONS {
            for j in 0..TOTAL_NEURONS {
                let w = core.swarm.weights.get(i, j);
                if w > 0.05 {
                    synapses.push(SynapseStateDto {
                        from: i,
                        to: j,
                        weight: w,
                        delay: 1,
                    });
                }
            }
        }

        Self {
            neurons,
            synapses,
            tick: core.tick,
            density: core.density,
            pending_spikes: core.swarm.spike_buffer.indices.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snn_core_creation() {
        let core = SnnCore::new(0.06);
        assert_eq!(core.swarm.arena.membrane.len(), TOTAL_NEURONS);
    }

    #[test]
    fn snn_core_step_runs() {
        let mut core = SnnCore::new(0.06);
        let result = core.step(&[]);
        assert_eq!(result.tick, 1);
    }

    #[test]
    fn snn_stage_constants() {
        assert_eq!(TOTAL_NEURONS, 180);
        assert_eq!(STAGES, 6);
        assert_eq!(NEURONS_PER_STAGE, 30);
    }

    #[test]
    fn snn_stage_of() {
        assert_eq!(SnnCore::stage_of(0), 0);
        assert_eq!(SnnCore::stage_of(29), 0);
        assert_eq!(SnnCore::stage_of(30), 1);
        assert_eq!(SnnCore::stage_of(179), 5);
    }
}
