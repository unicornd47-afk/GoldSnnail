//! NoteCoreLayer — 3-1-4-1 Frontier-Inspired Fractal Block
//!
//! A NoteCoreLayer is the fundamental building block of the scalable architecture.
//! It follows the 3-1-4-1 pattern WITH frontier enhancements:
//!
//!   3  = three sub-components: InputAdapter, FrozenCore, OutputAdapter
//!   1  = one unified spike contract (all I/O is spike-index lists)
//!   4  = four scale axes: width, depth, recurrence, plasticity
//!   1  = one frozen backbone per layer
//!
//! Frontier enhancements:
//!   - Gated state update (Mamba/RWKV-inspired)
//!   - Spike-based attention pattern (Transformer-inspired)
//!   - Residual gating with learned coefficient
//!
//! Each NoteCoreLayer can be recursively composed: the output of one layer
//! feeds the input of the next, preserving the same interface at every depth.

use crate::harness::fractal_core::{FrozenCore, FrozenCoreConfig};
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
        let scale = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let weights: Vec<f32> = (0..input_dim * output_dim)
            .map(|_| rand::random::<f32>() * 2.0 * scale - scale)
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

/// A single NoteCoreLayer: InputAdapter -> FrozenCore -> OutputAdapter.
///
/// This is the 3-1-4-1 unit with frontier enhancements:
///   - 3 components (input adapter, frozen core, output adapter)
///   - 1 unified spike contract
///   - 4 scale axes (width, depth, recurrence, plasticity)
///   - 1 frozen backbone per layer
///   - Gated state update (Mamba/RWKV-inspired)
///   - Spike-based attention pattern (Transformer-inspired)
#[derive(Debug, Clone)]
pub struct NoteCoreLayer {
    pub input_adapter: SpikeAdapter,
    pub core: FrozenCore,
    pub output_adapter: SpikeAdapter,
    pub gate_adapter: SpikeAdapter, // Learned gating coefficient
    pub scale: ScaleProfile,
    pub layer_id: usize,
}

impl NoteCoreLayer {
    /// Creates a new NoteCoreLayer with a given scale profile and I/O dimensions.
    pub fn new(layer_id: usize, scale: ScaleProfile, input_dim: usize, core_dim: usize, output_dim: usize) -> Self {
        let core_config = FrozenCoreConfig::default();
        let core = FrozenCore::new(core_config);
        let input_adapter = SpikeAdapter::new(input_dim, core_dim);
        let output_adapter = SpikeAdapter::new(6, output_dim); // core outputs 6 stage_means
        let gate_adapter = SpikeAdapter::new(input_dim, 1); // scalar gate
        Self { input_adapter, core, output_adapter, gate_adapter, scale, layer_id }
    }

    /// Creates a NoteCoreLayer configured for ARC latent reasoning.
    ///
    /// - `input_dim` must equal `73 * scale_width` from `ArcTripartiteEncoder`
    /// - `output_dim` is set to `4 * scale_width` for the 4 semantic latent vectors
    pub fn new_arc(layer_id: usize, scale: ScaleProfile, input_dim: usize) -> Self {
        let core_dim = 32 * scale.width.min(4);
        let output_dim = 4 * scale.width; // 4 semantic regions
        Self::new(layer_id, scale, input_dim, core_dim, output_dim)
    }

    /// Forward pass through the layer with gating and residual.
    ///
    /// `input_features` are projected into spike space, processed by the frozen
    /// core for `scale.recurrence` timesteps, then projected back out with gating.
    pub fn forward(&mut self, input_features: &[f32]) -> NoteCoreLayerResult {
        // 1. Input adapter: features -> core input spikes (top-k selection)
        let core_input = self.input_adapter.forward(input_features);
        let input_spikes = top_k_indices(&core_input, 30);

        // 2. Frozen core: deterministic spike dynamics
        let core_result = self.core.run(&input_spikes, self.scale.recurrence);

        // 3. Output adapter: core stage_means -> output features
        let mut output_features = self.output_adapter.forward(&core_result.stage_means);

        // 4. Gated residual: combine with input (Mamba/RWKV-inspired gating)
        let gate_logit = self.gate_adapter.forward(input_features);
        let gate = sigmoid(gate_logit[0]);
        for i in 0..output_features.len().min(input_features.len()) {
            output_features[i] = gate * output_features[i] + (1.0 - gate) * input_features[i];
        }

        NoteCoreLayerResult {
            output_features,
            output_spikes: core_result.spikes,
            stage_means: core_result.stage_means,
            tick: core_result.tick,
            layer_id: self.layer_id,
            gate,
        }
    }

    /// ARC-specific forward pass.
    ///
    /// Returns 4 semantic latent vectors derived from the frozen core stage_means.
    /// No residual gating is applied; the raw semantic vectors are returned.
    pub fn forward_arc(&mut self, code: &[f32]) -> ArcReasoningResult {
        assert_eq!(code.len(), self.input_adapter.input_dim, "ARC code dim mismatch: expected {}, got {}", self.input_adapter.input_dim, code.len());

        // 1. Input adapter: tripartite code -> core input spikes
        let core_input = self.input_adapter.forward(code);
        let input_spikes = top_k_indices(&core_input, 30);

        // 2. Frozen core: deterministic spike dynamics
        let core_result = self.core.run(&input_spikes, self.scale.recurrence);

        // 3. Output adapter: 6 stage_means -> 4 semantic vectors
        let semantic_vectors = self.output_adapter.forward(&core_result.stage_means);

        ArcReasoningResult {
            semantic_vectors,
            stage_means: core_result.stage_means,
            output_spikes: core_result.spikes,
            tick: core_result.tick,
            layer_id: self.layer_id,
        }
    }

    /// Updates adapters using a simple gradient signal (no core mutation).
    pub fn adapt(&mut self, input: &[f32], grad_output: &[f32], lr: f64) {
        let lr_f = lr as f32 * self.scale.plasticity as f32;
        // Update output adapter
        self.output_adapter.update(&self.core.membrane_snapshot()[..6], grad_output, lr_f);
        // Update input adapter
        self.input_adapter.update(input, grad_output, lr_f * 0.5);
        // Update gate adapter
        let gate_grad = vec![grad_output.iter().map(|g| g * 0.1).sum::<f32>() / grad_output.len() as f32];
        self.gate_adapter.update(input, &gate_grad, lr_f * 0.2);
    }

    /// ARC-specific adapter update.
    ///
    /// Updates input and output adapters using semantic vector gradients.
    pub fn adapt_arc(&mut self, code: &[f32], grad_semantics: &[f32], lr: f64) {
        let lr_f = lr as f32 * self.scale.plasticity as f32;
        // Update output adapter with semantic gradient
        self.output_adapter.update(&self.core.membrane_snapshot()[..6], grad_semantics, lr_f);
        // Project semantic gradient to input adapter output space for backprop
        let grad_input: Vec<f32> = grad_semantics.iter()
            .cycle()
            .take(self.input_adapter.output_dim)
            .cloned()
            .collect();
        self.input_adapter.update(code, &grad_input, lr_f * 0.5);
    }

    /// Returns total parameter count (adapters only; core is frozen).
    pub fn param_count(&self) -> usize {
        self.input_adapter.weights.len()
            + self.input_adapter.bias.len()
            + self.output_adapter.weights.len()
            + self.output_adapter.bias.len()
            + self.gate_adapter.weights.len()
            + self.gate_adapter.bias.len()
    }

    /// Returns compute cost proxy from scale profile.
    pub fn compute_cost(&self) -> usize {
        self.scale.compute_cost()
    }
}

/// Result of a NoteCoreLayer forward pass.
#[derive(Debug, Clone, Default)]
pub struct NoteCoreLayerResult {
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
    /// Learned gating coefficient (0-1).
    pub gate: f32,
}

/// Result of an ARC reasoning forward pass.
#[derive(Debug, Clone, Default)]
pub struct ArcReasoningResult {
    /// 4 semantic latent vectors (4 * scale_width dims).
    pub semantic_vectors: Vec<f32>,
    /// Mean membrane per stage (6D rule memory).
    pub stage_means: Vec<f32>,
    /// Output spike indices from the frozen core.
    pub output_spikes: Vec<usize>,
    /// Tick count after processing.
    pub tick: u64,
    /// Which layer produced this result.
    pub layer_id: usize,
}

/// Sigmoid activation function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
    fn note_core_layer_creation() {
        let scale = ScaleProfile::base();
        let layer = NoteCoreLayer::new(0, scale, 16, 32, 16);
        assert_eq!(layer.layer_id, 0);
        assert_eq!(layer.param_count(), 16 * 32 + 32 + 6 * 16 + 16 + 16 * 1 + 1);
    }

    #[test]
    fn note_core_layer_forward_runs() {
        let scale = ScaleProfile::base();
        let mut layer = NoteCoreLayer::new(0, scale, 16, 32, 16);
        let input = vec![0.1f32; 16];
        let result = layer.forward(&input);
        assert_eq!(result.layer_id, 0);
        assert_eq!(result.stage_means.len(), 6);
        assert!(result.gate >= 0.0 && result.gate <= 1.0);
    }

    #[test]
    fn note_core_layer_adapt_changes_weights() {
        let scale = ScaleProfile::base();
        let mut layer = NoteCoreLayer::new(0, scale, 8, 8, 8);
        let input = vec![0.1f32; 8];
        let grad = vec![1.0f32; 8];
        layer.forward(&input);
        let before_weights: f32 = layer.output_adapter.weights.iter().sum();
        let before_biases: f32 = layer.output_adapter.bias.iter().sum();
        layer.adapt(&input, &grad, 1.0);
        let after_weights: f32 = layer.output_adapter.weights.iter().sum();
        let after_biases: f32 = layer.output_adapter.bias.iter().sum();
        let total_before = before_weights + before_biases;
        let total_after = after_weights + after_biases;
        assert!(total_before != total_after, "Parameters should change after adapt: before={}, after={}", total_before, total_after);
    }

    #[test]
    fn note_core_layer_compute_cost() {
        let scale = ScaleProfile { width: 2, depth: 3, recurrence: 8, plasticity: 1.0 };
        let layer = NoteCoreLayer::new(0, scale, 8, 16, 8);
        assert_eq!(layer.compute_cost(), 2 * 3 * 8);
    }

    #[test]
    fn note_core_layer_arc_creation() {
        let scale = ScaleProfile::base();
        let layer = NoteCoreLayer::new_arc(0, scale, 73);
        assert_eq!(layer.layer_id, 0);
        assert_eq!(layer.input_adapter.input_dim, 73);
        assert_eq!(layer.output_adapter.output_dim, 4); // 4 semantic vectors
    }

    #[test]
    fn note_core_layer_forward_arc_runs() {
        let scale = ScaleProfile::base();
        let mut layer = NoteCoreLayer::new_arc(0, scale, 73);
        let code = vec![0.1f32; 73];
        let result = layer.forward_arc(&code);
        assert_eq!(result.layer_id, 0);
        assert_eq!(result.stage_means.len(), 6);
        assert_eq!(result.semantic_vectors.len(), 4);
        assert!(!result.output_spikes.is_empty(), "Core should emit spikes");
    }

    #[test]
    fn note_core_layer_arc_adapt_changes_weights() {
        let scale = ScaleProfile::base();
        let mut layer = NoteCoreLayer::new_arc(0, scale, 73);
        let code = vec![0.1f32; 73];
        let grad = vec![1.0f32; 4];
        layer.forward_arc(&code);
        let before_weights: f32 = layer.output_adapter.weights.iter().sum();
        let before_biases: f32 = layer.output_adapter.bias.iter().sum();
        layer.adapt_arc(&code, &grad, 1.0);
        let after_weights: f32 = layer.output_adapter.weights.iter().sum();
        let after_biases: f32 = layer.output_adapter.bias.iter().sum();
        let total_before = before_weights + before_biases;
        let total_after = after_weights + after_biases;
        assert!(total_before != total_after, "Parameters should change after adapt_arc: before={}, after={}", total_before, total_after);
    }

    #[test]
    fn note_core_layer_arc_semantic_dims_match_scale_width() {
        for width in [1, 2, 4] {
            let scale = ScaleProfile { width, depth: 1, recurrence: 4, plasticity: 1.0 };
            let layer = NoteCoreLayer::new_arc(0, scale, 73 * width);
            assert_eq!(layer.output_adapter.output_dim, 4 * width);
        }
    }
}
