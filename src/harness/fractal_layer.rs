//! FractalLayer — Self-Similar Wrapper Around a FrozenCore
//!
//! A FractalLayer is the fundamental building block of the scalable architecture.
//! It follows the 3-1-4-1 pattern:
//!
//!   3  = three sub-components: InputAdapter, FrozenCore, OutputAdapter
//!   1  = one unified spike contract (all I/O is spike-index lists)
//!   4  = four scale axes: width, depth, recurrence, plasticity
//!   1  = one frozen backbone per layer
//!
//! Each FractalLayer can be recursively composed: the output of one layer
//! feeds the input of the next, preserving the same interface at every depth.

use crate::harness::fractal_core::{FrozenCore, FrozenCoreConfig, FrozenCoreResult};
use rand::Rng;
use crate::harness::scale::ScaleProfile;

/// Trainable linear adapter: projects spike features to/from the core space.
///
/// DOD-compliant: flat Vec<f32>, no Box<dyn>, usize-indexed.
#[derive(Debug, Clone)]
pub struct SpikeAdapter {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl SpikeAdapter {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let weights: Vec<f32> = (0..input_dim * output_dim)
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let bias = vec![0.0f32; output_dim];
        Self { weights, bias, input_dim, output_dim }
    }

    /// Forward pass: input_features -> output_features
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim, "Adapter input dim mismatch");
        let mut out = vec![0.0f32; self.output_dim];
        for o in 0..self.output_dim {
            let mut sum = self.bias[o];
            for i in 0..self.input_dim {
                sum += self.weights[o * self.input_dim + i] * input[i];
            }
            out[o] = sum;
        }
        out
    }

    /// SGD update on a single gradient signal.
    pub fn update(&mut self, input: &[f32], grad_output: &[f32], lr: f32) {
        assert_eq!(input.len(), self.input_dim);
        assert_eq!(grad_output.len(), self.output_dim);
        for o in 0..self.output_dim {
            let g = grad_output[o] * lr;
            self.bias[o] -= g;
            for i in 0..self.input_dim {
                let idx = o * self.input_dim + i;
                self.weights[idx] -= g * input[i];
            }
        }
    }

    /// L2 norm of all weights (for telemetry).
    pub fn weight_norm(&self) -> f32 {
        self.weights.iter().map(|w| w * w).sum::<f32>().sqrt()
    }
}

/// A single FractalLayer: InputAdapter -> FrozenCore -> OutputAdapter.
///
/// This is the 3-1-4-1 unit:
///   - 3 components (input adapter, frozen core, output adapter)
///   - 1 unified spike contract
///   - 4 scale axes (width, depth, recurrence, plasticity)
///   - 1 frozen backbone
#[derive(Debug, Clone)]
pub struct FractalLayer {
    pub input_adapter: SpikeAdapter,
    pub core: FrozenCore,
    pub output_adapter: SpikeAdapter,
    pub scale: ScaleProfile,
    pub layer_id: usize,
}

impl FractalLayer {
    /// Creates a new FractalLayer with a given scale profile and I/O dimensions.
    pub fn new(layer_id: usize, scale: ScaleProfile, input_dim: usize, core_dim: usize, output_dim: usize) -> Self {
        let core_config = FrozenCoreConfig::default();
        let core = FrozenCore::new(core_config);
        let input_adapter = SpikeAdapter::new(input_dim, core_dim);
        let output_adapter = SpikeAdapter::new(6, output_dim);
        Self { input_adapter, core, output_adapter, scale, layer_id }
    }

    /// Forward pass through the layer.
    ///
    /// `input_features` are projected into spike space, processed by the frozen
    /// core for `scale.recurrence` timesteps, then projected back out.
    pub fn forward(&mut self, input_features: &[f32]) -> FractalLayerResult {
        // 1. Input adapter: features -> core input spikes (top-k selection)
        let core_input = self.input_adapter.forward(input_features);
        let input_spikes = top_k_indices(&core_input, 30); // 30 input neurons

        // 2. Frozen core: deterministic spike dynamics
        let core_result = self.core.run(&input_spikes, self.scale.recurrence);

        // 3. Output adapter: core stage_means -> output features
        let output_features = self.output_adapter.forward(&core_result.stage_means);

        FractalLayerResult {
            output_features,
            output_spikes: core_result.spikes,
            stage_means: core_result.stage_means,
            tick: core_result.tick,
            layer_id: self.layer_id,
        }
    }

    /// Updates adapters using a simple gradient signal (no core mutation).
    pub fn adapt(&mut self, input: &[f32], grad_output: &[f32], lr: f64) {
        let lr_f = lr as f32 * self.scale.plasticity as f32;
        // Update output adapter (direct gradient)
        self.output_adapter.update(&self.core.membrane_snapshot()[..6], grad_output, lr_f);
        // Update input adapter (through core, simplified)
        self.input_adapter.update(input, grad_output, lr_f * 0.5);
    }

    /// Returns total parameter count (adapters only; core is frozen).
    pub fn param_count(&self) -> usize {
        self.input_adapter.weights.len()
            + self.input_adapter.bias.len()
            + self.output_adapter.weights.len()
            + self.output_adapter.bias.len()
    }

    /// Returns compute cost proxy from scale profile.
    pub fn compute_cost(&self) -> usize {
        self.scale.compute_cost()
    }
}

/// Result of a FractalLayer forward pass.
#[derive(Debug, Clone, Default)]
pub struct FractalLayerResult {
    /// Output feature vector.
    pub output_features: Vec<f32>,
    /// Output spike indices from the frozen core.
    pub output_spikes: Vec<usize>,
    /// Mean membrane per stage (6D).
    pub stage_means: Vec<f32>,
    /// Tick count after processing.
    pub tick: u64,
    /// Which layer produced this result.
    pub layer_id: usize,
}

/// Selects the indices of the top-k values in `v`.
fn top_k_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = v.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractal_layer_creation() {
        let scale = ScaleProfile::base();
        let layer = FractalLayer::new(0, scale, 16, 32, 16);
        assert_eq!(layer.layer_id, 0);
        assert_eq!(layer.param_count(), 16 * 32 + 32 + 32 * 16 + 16);
    }

    #[test]
    fn fractal_layer_forward_runs() {
        let scale = ScaleProfile::base();
        let mut layer = FractalLayer::new(0, scale, 16, 32, 16);
        let input = vec![0.1f32; 16];
        let result = layer.forward(&input);
        assert_eq!(result.layer_id, 0);
        assert_eq!(result.stage_means.len(), 6);
    }

    #[test]
    fn fractal_layer_adapt_changes_weights() {
        let scale = ScaleProfile::base();
        let mut layer = FractalLayer::new(0, scale, 8, 16, 8);
        let input = vec![0.1f32; 8];
        let grad = vec![0.01f32; 8];
        let before = layer.output_adapter.weights[0];
        layer.adapt(&input, &grad, 0.1);
        let after = layer.output_adapter.weights[0];
        assert!((after - before).abs() > 1e-10, "Weights should change after adapt");
    }

    #[test]
    fn fractal_layer_compute_cost() {
        let scale = ScaleProfile { width: 2, depth: 3, recurrence: 8, plasticity: 1.0 };
        let layer = FractalLayer::new(0, scale, 8, 16, 8);
        assert_eq!(layer.compute_cost(), 2 * 3 * 8);
    }
}






