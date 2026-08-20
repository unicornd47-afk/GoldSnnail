//! FractalNetwork — Recursive Stack of FractalLayers
//!
//! A FractalNetwork is a self-similar stack of FractalLayers.
//! It implements the frontier "FractalNet" pattern:
//!   - Same 3-1-4-1 block repeated at every depth level
//!   - Residual connections between layers
//!   - Local learning in adapters only (frozen cores untouched)
//!
//! Architecture mapping to frontier templates:
//!   Transformer   -> Attention(Quaternion) + FrozenCore + Adapter FFN
//!   Mamba/RWKV    -> Temporal Recurrence + Gating + State update
//!   FractalNet    -> Recursive FractalLayer composition
//!   Universal Approx -> Frozen basis (core) + Linear adapters = function space

use crate::harness::fractal_core::FrozenCore;
use crate::harness::fractal_layer::FractalLayer;
use crate::harness::scale::ScaleProfile;

/// A recursive fractal network with uniform layer scales.
///
/// All layers share the same scale profile for equal fractal scaling.
/// Residual connections link consecutive layers.
#[derive(Debug, Clone)]
pub struct FractalNetwork {
    pub layers: Vec<FractalLayer>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub base_scale: ScaleProfile,
}

impl FractalNetwork {
    /// Creates a new FractalNetwork with `depth` uniformly-scaled layers.
    ///
    /// All layers share the same `base_scale` for true fractal equality.
    pub fn new(input_dim: usize, output_dim: usize, depth: usize, base_scale: ScaleProfile) -> Self {
        assert!(depth > 0, "FractalNetwork depth must be >= 1");
        assert!(input_dim > 0 && output_dim > 0, "I/O dimensions must be positive");

        let mut layers = Vec::with_capacity(depth);
        for d in 0..depth {
            let core_dim = 32 * base_scale.width.min(4);
            let layer_scale = ScaleProfile {
                width: base_scale.width,
                depth: base_scale.depth,
                recurrence: base_scale.recurrence,
                plasticity: base_scale.plasticity,
            };
            let layer = FractalLayer::new(d, layer_scale, input_dim, core_dim, output_dim);
            layers.push(layer);
        }

        Self { layers, input_dim, output_dim, base_scale }
    }

    /// Forward pass through all layers with residual connections.
    ///
    /// Each layer processes the sum of (layer_input + previous_output),
    /// implementing the frontier "residual block" pattern.
    pub fn forward(&mut self, mut input: Vec<f32>) -> FractalNetworkResult {
        assert_eq!(input.len(), self.input_dim, "Network input dim mismatch");
        let mut residual = vec![0.0f32; self.output_dim];
        let mut all_spikes = Vec::new();
        let mut total_ticks = 0;

        for layer in &mut self.layers {
            // Residual: combine current input with previous layer output
            let mut combined = input.clone();
            if combined.len() == residual.len() {
                for i in 0..combined.len() {
                    combined[i] += residual[i] * 0.5; // gated residual
                }
            }

            let result = layer.forward(&combined);
            residual = result.output_features.clone();
            input = result.output_features.clone();
            all_spikes.extend(result.output_spikes);
            total_ticks += result.tick;
        }

        FractalNetworkResult {
            output: residual,
            total_spikes: all_spikes.len(),
            total_ticks,
            layer_count: self.layers.len(),
        }
    }

    /// Adapts all layers using a shared gradient signal.
    pub fn adapt(&mut self, input: &[f32], grad_output: &[f32], lr: f64) {
        let mut current_grad = grad_output.to_vec();
        for layer in &mut self.layers {
            layer.adapt(input, &current_grad, lr);
            // Simplified gradient propagation backward
            current_grad = current_grad.iter().map(|g| g * 0.9).collect();
        }
    }

    /// Returns total parameter count across all adapters.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    /// Returns total compute cost proxy.
    pub fn compute_cost(&self) -> usize {
        self.layers.iter().map(|l| l.compute_cost()).sum()
    }

    /// Returns the scale profile of layer `i`.
    pub fn layer_scale(&self, i: usize) -> Option<&ScaleProfile> {
        self.layers.get(i).map(|l| &l.scale)
    }
}

/// Result of a FractalNetwork forward pass.
#[derive(Debug, Clone, Default)]
pub struct FractalNetworkResult {
    /// Final output feature vector.
    pub output: Vec<f32>,
    /// Total spikes emitted across all layers.
    pub total_spikes: usize,
    /// Total SNN ticks processed.
    pub total_ticks: u64,
    /// Number of layers in the network.
    pub layer_count: usize,
}

/// Builds a "3141" fractal network — the canonical minimal configuration.
///
/// The name encodes the topology:
///   3 = 3 scale axes used (width, recurrence, plasticity; depth=1)
///   1 = 1 layer
///   4 = 4 width multiplier
///   1 = 1 frozen core
///
/// This is the smallest non-trivial fractal unit.
pub fn build_3141_fractal(input_dim: usize, output_dim: usize) -> FractalNetwork {
    let scale = ScaleProfile {
        width: 1,
        depth: 1,
        recurrence: 4,
        plasticity: 1.0,
    };
    FractalNetwork::new(input_dim, output_dim, 1, scale)
}

/// Scales a 3141 fractal network to a larger configuration while preserving
/// the self-similar 3-1-4-1 pattern.
///
/// Rules:
///   - Width  doubles: 1 -> 2 -> 4 (capped)
///   - Depth  doubles: 1 -> 2 -> 4 (capped)
///   - Recurrence doubles: 4 -> 8 -> 16 (capped)
///   - Plasticity stays at 1.0 (frozen core + trainable adapters)
pub fn scale_3141(base: &FractalNetwork, factor: usize) -> FractalNetwork {
    assert!(factor > 0, "Scale factor must be >= 1");
    let f = factor.min(8); // safety cap

    let width = 1 << f.min(2); // 1, 2, 4
    let depth = 1 << f.min(2); // 1, 2, 4
    let recurrence = 4 * (1 << f.min(3)); // 4, 8, 16, 32

    let scale = ScaleProfile {
        width: width as usize,
        depth: depth as usize,
        recurrence: recurrence as usize,
        plasticity: 1.0,
    };

    FractalNetwork::new(base.input_dim, base.output_dim, depth as usize, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_3141_fractal_creates_single_layer() {
        let net = build_3141_fractal(16, 16);
        assert_eq!(net.layers.len(), 1);
        assert_eq!(net.layers[0].scale.width, 1);
        assert_eq!(net.layers[0].scale.recurrence, 4);
    }

    #[test]
    fn scale_3141_doubles_width_depth() {
        let base = build_3141_fractal(16, 16);
        let scaled = scale_3141(&base, 2);
        assert_eq!(scaled.layers.len(), 4); // depth doubled twice: 1->2->4
        assert_eq!(scaled.layers[0].scale.width, 4); // width doubled twice: 1->2->4
    }

    #[test]
    fn fractal_network_forward_runs() {
        let mut net = build_3141_fractal(16, 16);
        let input = vec![0.1f32; 16];
        let result = net.forward(input);
        assert_eq!(result.output.len(), 16);
        assert!(result.total_ticks > 0);
    }

    #[test]
    fn fractal_network_param_count() {
        let net = build_3141_fractal(16, 16);
        assert!(net.param_count() > 0);
    }

    #[test]
    fn fractal_network_residual_produces_output() {
        let mut net = FractalNetwork::new(8, 8, 2, ScaleProfile::base());
        let input = vec![0.5f32; 8];
        let result = net.forward(input);
        assert_eq!(result.output.len(), 8);
        assert!(result.layer_count == 2);
    }
}
