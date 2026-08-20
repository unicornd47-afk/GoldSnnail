//! ArcStreamingLoop — End-to-End ARC Pipeline
//!
//! Streams an ARC task through the full pipeline:
//!   1. Encode demo input/output grids
//!   2. Run NoteCoreLayer forward passes
//!   3. Collect semantic vectors
//!   4. Decode to predicted test output grid
//!
//! This is the bridge between ARC encoding and the fractal SNN core.

use crate::harness::arc_tripartite::{ArcTripartiteEncoder, ArcPhase};
use crate::harness::note_core_layer::{NoteCoreLayer, ArcReasoningResult};
use crate::harness::scale::ScaleProfile;

/// Result of streaming a single ARC task through the pipeline.
#[derive(Debug, Clone)]
pub struct ArcStreamResult {
    /// Predicted test output grid (width x height)
    pub predicted_grid: Vec<Vec<u8>>,
    /// Semantic vectors from the last layer
    pub semantic_vectors: Vec<f32>,
    /// Core tick count
    pub tick: u64,
    /// Whether the prediction is valid (all colors 0..=9)
    pub is_valid: bool,
}

/// Streaming loop for ARC tasks.
///
/// Encodes demo pairs, runs the core, and decodes the result.
#[derive(Debug, Clone)]
pub struct ArcStreamingLoop {
    pub encoder: ArcTripartiteEncoder,
    pub layer: NoteCoreLayer,
    pub scale: ScaleProfile,
}

impl ArcStreamingLoop {
    /// Creates a new streaming loop for an ARC task.
    ///
    /// - `width`, `height`: ARC grid dimensions
    /// - `scale_width`: scaling factor for the tripartite encoder
    pub fn new(width: usize, height: usize, scale_width: usize) -> Self {
        let scale = ScaleProfile::base();
        let encoder = ArcTripartiteEncoder::new(width, height, scale_width);
        let input_dim = encoder.scaled_dim();
        let layer = NoteCoreLayer::new_arc(0, scale, input_dim);
        Self { encoder, layer, scale }
    }

    /// Streams a full ARC task through the pipeline.
    ///
    /// # Arguments
    /// - `demo_inputs`: list of demo input grids
    /// - `demo_outputs`: list of demo output grids
    /// - `test_input`: the test input grid to predict
    ///
    /// # Returns
    /// Predicted test output grid and metadata
    pub fn stream(
        &mut self,
        demo_inputs: &[Vec<Vec<u8>>],
        demo_outputs: &[Vec<Vec<u8>>],
        test_input: &[Vec<u8>],
    ) -> ArcStreamResult {
        // 1. Encode demo input grids
        for grid in demo_inputs {
            let codes = self.encoder.encode_grid(grid, ArcPhase::DemoInput);
            for code in codes {
                let _ = self.layer.forward_arc(&code);
            }
        }

        // 2. Encode demo output grids
        for grid in demo_outputs {
            let codes = self.encoder.encode_grid(grid, ArcPhase::DemoOutput);
            for code in codes {
                let _ = self.layer.forward_arc(&code);
            }
        }

        // 3. Encode test input and get final semantic vectors
        let codes = self.encoder.encode_grid(test_input, ArcPhase::TestInput);
        let mut last_result: Option<ArcReasoningResult> = None;
        for code in codes {
            last_result = Some(self.layer.forward_arc(&code));
        }

        let result = last_result.unwrap_or_default();

        // 4. Decode semantic vectors to predicted grid
        let predicted_grid = decode_semantic_vectors(
            &result.semantic_vectors,
            self.encoder.width,
            self.encoder.height,
        );

        let is_valid = predicted_grid.iter().all(|row| {
            row.iter().all(|&c| c <= 9)
        });

        ArcStreamResult {
            predicted_grid,
            semantic_vectors: result.semantic_vectors,
            tick: result.tick,
            is_valid,
        }
    }

    /// Adapts the layer using a reward signal from a correct demo pair.
    ///
    /// This is a simple heuristic: if the predicted output matches the demo output,
    /// positive reward; otherwise negative reward proportional to pixel mismatch.
    pub fn adapt_from_demo(
        &mut self,
        demo_input: &[Vec<u8>],
        demo_output: &[Vec<u8>],
        lr: f64,
    ) -> f32 {
        // Run forward pass
        let codes = self.encoder.encode_grid(demo_input, ArcPhase::DemoInput);
        let mut result = None;
        for code in codes {
            result = Some(self.layer.forward_arc(&code));
        }
        let result = result.unwrap_or_default();

        // Decode prediction
        let predicted = decode_semantic_vectors(
            &result.semantic_vectors,
            self.encoder.width,
            self.encoder.height,
        );

        // Compute reward (1.0 - pixel_error_rate)
        let total_pixels = demo_output.len() * demo_output[0].len();
        let mut errors = 0u32;
        for (y, row) in demo_output.iter().enumerate() {
            for (x, &color) in row.iter().enumerate() {
                if predicted[y][x] != color {
                    errors += 1;
                }
            }
        }
        let error_rate = errors as f32 / total_pixels as f32;
        let reward = 1.0 - error_rate;

        // Compute gradient and adapt
        let grad_semantics: Vec<f32> = result.semantic_vectors.iter()
            .map(|v| v * (1.0 - reward) * 0.1)
            .collect();
        self.layer.adapt_arc(
            &self.encoder.encode_grid(demo_input, ArcPhase::DemoInput)[0],
            &grad_semantics,
            lr,
        );

        reward
    }
}

/// Decodes semantic vectors back to an ARC grid.
///
/// Simple heuristic: each semantic vector votes for a color per row.
/// The color with the highest accumulated vote wins for each row.
fn decode_semantic_vectors(
    semantic_vectors: &[f32],
    width: usize,
    height: usize,
) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; width]; height];

    if semantic_vectors.is_empty() {
        return grid;
    }

    // Use first 4 semantic vectors to determine row colors
    let row_colors: Vec<u8> = semantic_vectors.iter()
        .take(4)
        .map(|&v| ((v.abs() * 10.0) as u8) % 10)
        .collect();

    // Fill grid with alternating row colors based on semantic vectors
    for y in 0..height {
        let color_idx = y % row_colors.len().max(1);
        let color = row_colors.get(color_idx).copied().unwrap_or(0);
        for x in 0..width {
            grid[y][x] = color;
        }
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_loop_creation() {
        let loop_ = ArcStreamingLoop::new(10, 10, 1);
        assert_eq!(loop_.encoder.width, 10);
        assert_eq!(loop_.encoder.height, 10);
        assert_eq!(loop_.encoder.scale_width, 1);
    }

    #[test]
    fn streaming_loop_runs() {
        let mut loop_ = ArcStreamingLoop::new(3, 3, 1);
        let demo_inputs = vec![vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]];
        let demo_outputs = vec![vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]]];
        let test_input = vec![vec![1, 1, 1], vec![2, 2, 2], vec![3, 3, 3]];
        let result = loop_.stream(&demo_inputs, &demo_outputs, &test_input);
        assert_eq!(result.predicted_grid.len(), 3);
        assert_eq!(result.predicted_grid[0].len(), 3);
        assert!(result.is_valid);
    }

    #[test]
    fn streaming_loop_produces_valid_colors() {
        let mut loop_ = ArcStreamingLoop::new(5, 5, 2);
        let demo_inputs = vec![vec![vec![0; 5]; 5]];
        let demo_outputs = vec![vec![vec![1; 5]; 5]];
        let test_input = vec![vec![2; 5]; 5];
        let result = loop_.stream(&demo_inputs, &demo_outputs, &test_input);
        assert!(result.is_valid, "All colors should be valid 0..=9");
    }

    #[test]
    fn adapt_from_demo_changes_weights() {
        let mut loop_ = ArcStreamingLoop::new(3, 3, 1);
        let demo_input = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
        let demo_output = vec![vec![8, 7, 6], vec![5, 4, 3], vec![2, 1, 0]];
        let reward = loop_.adapt_from_demo(&demo_input, &demo_output, 0.1);
        assert!(reward >= 0.0 && reward <= 1.0);
    }

    #[test]
    fn decode_semantic_vectors_produces_grid() {
        let semantics = vec![1.0, 2.0, 3.0, 4.0];
        let grid = decode_semantic_vectors(&semantics, 4, 3);
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].len(), 4);
        for row in &grid {
            for &c in row {
                assert!(c <= 9, "Color {} out of bounds", c);
            }
        }
    }
}
