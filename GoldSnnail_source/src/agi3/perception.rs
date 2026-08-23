//! Perception pipeline bridging ARC-AGI-3 pixel observations to the GoldWorm SNN.
//!
//! Converts raw 2D grid frames into spike patterns for the 180-neuron SNN's
//! Sensor stage (neurons 0-29).

use crate::agi3::Observation;
use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::swarm::snn_core::{SnnCore, SnnStepResult};
use crate::vision::{ArcGrid, GridEncoder};

#[derive(Debug, Clone)]
pub struct PerceptionPipeline {
    pub encoder: GridEncoder,
    pub ball: PoincareBall,
    pub snn_core: SnnCore,
    pub target_radius: f64,
    pub input_dim: usize,
}

impl PerceptionPipeline {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        target_radius: f64,
        snn_density: f64,
    ) -> Self {
        let encoder = GridEncoder::new(input_dim, hidden_dim, output_dim, target_radius);
        let ball = PoincareBall::new(1.0);
        let snn_core = SnnCore::new(snn_density);
        Self {
            encoder,
            ball,
            snn_core,
            target_radius,
            input_dim,
        }
    }

    pub fn encode_grid(&self, grid: &ArcGrid) -> Result<Vec<f64>, String> {
        let features_f32 = grid.to_feature_vector();
        if features_f32.len() != self.input_dim {
            return Err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.input_dim,
                features_f32.len()
            ));
        }
        let features: Vec<f64> = features_f32.iter().map(|&x| x as f64).collect();
        Ok(self.encoder.forward(&features))
    }

    pub fn grid_to_hyperbolic(&self, grid: &ArcGrid) -> Result<HyperbolicPoint, String> {
        let encoded = self.encode_grid(grid)?;
        let arr = ndarray::Array1::from_vec(encoded);
        self.ball
            .exp_map_origin(&arr)
            .map_err(|e| format!("Poincaré projection failed: {:?}", e))
    }

    pub fn grid_to_spikes(&self, grid: &ArcGrid) -> Vec<usize> {
        let features_f32 = grid.to_feature_vector();
        let features: Vec<f64> = features_f32.iter().map(|&x| x as f64).collect();
        let mut spikes = Vec::new();
        let threshold = 0.3;

        for group in 0..10 {
            let start = group * 10;
            let end = (start + 10).min(features.len());
            if start >= features.len() {
                break;
            }
            let group_features = &features[start..end];
            let max_val = group_features
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if max_val > threshold {
                spikes.push(group);
            }
        }

        let total_cells = (grid.width * grid.height).max(1);
        let unique_colors = grid.unique_colors();
        for &color in &unique_colors {
            let count = grid.data.iter().flatten().filter(|&&c| c == color).count();
            let ratio = count as f64 / total_cells as f64;
            if ratio > 0.05 {
                let neuron = 10 + color as usize;
                if neuron < 30 {
                    spikes.push(neuron);
                }
            }
        }

        spikes
    }

    pub fn step_snn(&mut self, input_spikes: &[usize]) -> SnnStepResult {
        self.snn_core.step(input_spikes)
    }

    pub fn sensor_activations(&self) -> Vec<f32> {
        self.snn_core.swarm.arena.membrane[..30].to_vec()
    }

    pub fn process_observation(
        &mut self,
        obs: &Observation,
    ) -> Result<HyperbolicPoint, String> {
        let grid = ArcGrid::from_2d(&obs.frame);
        let hyperbolic = self.grid_to_hyperbolic(&grid)?;
        let spikes = self.grid_to_spikes(&grid);
        self.step_snn(&spikes);
        Ok(hyperbolic)
    }

    pub fn train_encoder_step(
        &mut self,
        input_grid: &ArcGrid,
        output_grid: &ArcGrid,
        lr: f64,
    ) -> Result<f64, String> {
        self.encoder.train_step(input_grid, output_grid, lr)
    }
}
