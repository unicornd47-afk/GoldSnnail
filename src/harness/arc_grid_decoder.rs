//! ArcGridDecoder — Decode Semantic Vectors to ARC Grids
//!
//! Decodes the 4 semantic latent vectors from NoteCoreLayer back into
//! a predicted ARC output grid.
//!
//! Uses a learned linear projection from semantic space to pixel space.

use crate::harness::note_core_layer::NoteCoreLayer;
use crate::harness::scale::ScaleProfile;

/// Decoder that maps 4 semantic latent vectors to an ARC grid.
///
/// Uses a simple linear layer per color channel, trained by SGD.
#[derive(Debug, Clone)]
pub struct ArcGridDecoder {
    /// Weights for decoding: (output_dim x 4) where output_dim = width * height
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub output_dim: usize,
}

impl ArcGridDecoder {
    /// Creates a new decoder for the given grid dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let output_dim = width * height;
        let scale = (6.0f32 / 4.0).sqrt(); // 4 semantic inputs
        let weights: Vec<f32> = (0..output_dim * 4)
            .map(|_| rand::random::<f32>() * 2.0 * scale - scale)
            .collect();
        let bias = vec![0.0f32; output_dim];
        Self { weights, bias, width, height, output_dim }
    }

    /// Creates a decoder matching a NoteCoreLayer's output dimensions.
    pub fn from_layer(layer: &NoteCoreLayer) -> Self {
        let output_dim = layer.output_adapter.output_dim;
        let width = output_dim / 4; // approximate
        let height = 1;
        Self::new(width.max(1), height.max(1))
    }

    /// Forward pass: semantic_vectors -> flat grid values.
    pub fn forward(&self, semantic_vectors: &[f32]) -> Vec<f32> {
        assert_eq!(semantic_vectors.len(), 4, "Expected 4 semantic vectors, got {}", semantic_vectors.len());
        let mut out = vec![0.0f32; self.output_dim];
        for o in 0..self.output_dim {
            let mut sum = self.bias[o];
            for i in 0..4 {
                sum += self.weights[o * 4 + i] * semantic_vectors[i];
            }
            out[o] = sum;
        }
        out
    }

    /// Decodes semantic vectors to an ARC grid.
    ///
    /// Maps the continuous values to discrete colors 0..=9.
    pub fn decode_to_grid(&self, semantic_vectors: &[f32]) -> Vec<Vec<u8>> {
        let flat = self.forward(semantic_vectors);
        let mut grid = vec![vec![0u8; self.width]; self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let val = flat[idx];
                // Map continuous value to 0..=9
                let color = ((val * 10.0).round() as i32 % 10).abs() as u8;
                grid[y][x] = color.min(9);
            }
        }
        grid
    }

    /// SGD update using grid-level gradient.
    pub fn update(&mut self, semantic_vectors: &[f32], grad_grid: &[Vec<f32>], lr: f32) {
        let flat_grad: Vec<f32> = grad_grid.iter().flatten().cloned().collect();
        assert_eq!(flat_grad.len(), self.output_dim, "Gradient dim mismatch");
        for o in 0..self.output_dim {
            let g = flat_grad[o] * lr;
            self.bias[o] -= g;
            for i in 0..4 {
                self.weights[o * 4 + i] -= g * semantic_vectors[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_creation() {
        let decoder = ArcGridDecoder::new(10, 10);
        assert_eq!(decoder.width, 10);
        assert_eq!(decoder.height, 10);
        assert_eq!(decoder.output_dim, 100);
        assert_eq!(decoder.weights.len(), 400); // 100 * 4
    }

    #[test]
    fn decoder_forward_runs() {
        let decoder = ArcGridDecoder::new(5, 5);
        let semantics = vec![0.5, -0.3, 0.8, -0.1];
        let flat = decoder.forward(&semantics);
        assert_eq!(flat.len(), 25);
    }

    #[test]
    fn decoder_produces_valid_grid() {
        let decoder = ArcGridDecoder::new(4, 3);
        let semantics = vec![0.5, -0.3, 0.8, -0.1];
        let grid = decoder.decode_to_grid(&semantics);
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].len(), 4);
        for row in &grid {
            for &c in row {
                assert!(c <= 9, "Color {} out of bounds", c);
            }
        }
    }

    #[test]
    fn decoder_update_changes_weights() {
        let mut decoder = ArcGridDecoder::new(4, 4);
        let semantics = vec![0.5, -0.3, 0.8, -0.1];
        let grad_grid = vec![vec![0.01f32; 4]; 4];
        let before = decoder.weights[0];
        decoder.update(&semantics, &grad_grid, 0.1);
        let after = decoder.weights[0];
        assert!(before != after, "Weights should change after update");
    }
}
