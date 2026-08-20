//! ArcTripartiteEncoder — 3-Region Spike Code for ARC-AGI
//!
//! Maps a single ARC pixel + phase into a flat spike code for the NoteCoreLayer.
//!
//! ## Contract
//! - Preconditions: width <= 30, height <= 30, scale_width >= 1
//! - Postconditions: encode() returns Vec<f32> of exact length 73 * scale_width
//! - Invariants: color in 0..=9, x in 0..width, y in 0..height
//! - Boundary Guards: assert!(color <= 9), assert!(x < width), assert!(y < height)



/// ARC task phase — drives the tonic Context signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcPhase {
    DemoInput,
    DemoOutput,
    TestInput,
}

impl ArcPhase {
    /// Returns the 3-dimensional tonic spike code for this phase.
    /// Encoding:
    ///   DemoInput  -> [1.0, 0.0, 0.0]
    ///   DemoOutput -> [0.0, 1.0, 0.0]
    ///   TestInput  -> [0.0, 0.0, 1.0]
    pub fn tonic_code(self) -> [f32; 3] {
        match self {
            ArcPhase::DemoInput  => [1.0, 0.0, 0.0],
            ArcPhase::DemoOutput => [0.0, 1.0, 0.0],
            ArcPhase::TestInput  => [0.0, 0.0, 1.0],
        }
    }
}

/// Tripartite encoder for ARC grids.
///
/// Splits the spike code into 3 regions:
///   Region 1 (Identity): 10 neurons — one-hot color spike
///   Region 2 (Spatial):  60 neurons — Gaussian place cells for X and Y
///   Region 3 (Context):  3 neurons  — tonic phase flag
///
/// Base dimension = 73. Scaled by ScaleProfile.width.
#[derive(Debug, Clone)]
pub struct ArcTripartiteEncoder {
    pub width: usize,
    pub height: usize,
    pub scale_width: usize,
}

impl ArcTripartiteEncoder {
    /// Creates a new encoder with strict boundary checks.
    ///
    /// # Panics
    /// - width == 0 or width > 30
    /// - height == 0 or height > 30
    /// - scale_width == 0
    pub fn new(width: usize, height: usize, scale_width: usize) -> Self {
        assert!(width > 0 && width <= 30, "width must be in 1..=30, got {}", width);
        assert!(height > 0 && height <= 30, "height must be in 1..=30, got {}", height);
        assert!(scale_width >= 1, "scale_width must be >= 1, got {}", scale_width);
        Self { width, height, scale_width }
    }

    /// Returns the base input dimension (before scaling).
    pub fn base_dim(&self) -> usize {
        73 // 10 + 60 + 3
    }

    /// Returns the scaled input dimension = base_dim * scale_width.
    pub fn scaled_dim(&self) -> usize {
        self.base_dim() * self.scale_width
    }

    /// Encodes a single pixel + phase into a flat spike code.
    ///
    /// # Panics
    /// - color > 9
    /// - x >= width
    /// - y >= height
    pub fn encode(&self, color: u8, x: usize, y: usize, phase: ArcPhase) -> Vec<f32> {
        // Boundary guards
        assert!(color <= 9, "ARC color must be 0..=9, got {}", color);
        assert!(x < self.width, "x coordinate {} out of bounds for width {}", x, self.width);
        assert!(y < self.height, "y coordinate {} out of bounds for height {}", y, self.height);

        let mut code = Vec::with_capacity(self.scaled_dim());

        // Region 1: Identity (10 neurons per scale unit)
        for _ in 0..self.scale_width {
            code.extend_from_slice(&Self::color_to_spike(color));
        }

        // Region 2: Spatial (60 neurons per scale unit: 30 X + 30 Y)
        for _ in 0..self.scale_width {
            code.extend_from_slice(&Self::position_to_spike(x, y, self.width, self.height));
        }

        // Region 3: Context (3 neurons per scale unit)
        for _ in 0..self.scale_width {
            code.extend_from_slice(&phase.tonic_code());
        }

        debug_assert_eq!(code.len(), self.scaled_dim(), "Encoded dimension mismatch");
        code
    }

    /// Encodes a full grid in raster order, returning a Vec of per-pixel codes.
    ///
    /// # Panics
    /// - grid dimensions mismatch encoder dimensions
    pub fn encode_grid(&self, grid: &[Vec<u8>], phase: ArcPhase) -> Vec<Vec<f32>> {
        assert_eq!(grid.len(), self.height, "Grid height mismatch: {} vs {}", grid.len(), self.height);
        for (y, row) in grid.iter().enumerate() {
            assert_eq!(row.len(), self.width, "Row {} width mismatch: {} vs {}", y, row.len(), self.width);
        }

        let mut codes = Vec::with_capacity(self.height * self.width);
        for y in 0..self.height {
            for x in 0..self.width {
                let color = grid[y][x];
                codes.push(self.encode(color, x, y, phase));
            }
        }
        codes
    }

    /// One-hot spike code for ARC color (10 neurons).
    fn color_to_spike(color: u8) -> [f32; 10] {
        let mut spike = [0.0f32; 10];
        spike[color as usize] = 1.0;
        spike
    }

    /// Gaussian place-cell spike code for X and Y coordinates (60 neurons).
    ///
    /// Uses 30 neurons for X and 30 for Y, each with a Gaussian tuning curve
    /// centered at regular intervals across the grid dimensions.
    fn position_to_spike(x: usize, y: usize, width: usize, height: usize) -> [f32; 60] {
        let mut spike = [0.0f32; 60];

        // X place cells: 30 neurons, Gaussian tuning with sigma = width/6
        let sigma_x = (width as f32) / 6.0;
        let centers_x: Vec<f32> = (0..30).map(|i| (i as f32 + 0.5) * (width as f32) / 30.0).collect();
        for i in 0..30 {
            let dist = (x as f32 - centers_x[i]).abs();
            spike[i] = (-dist * dist / (2.0 * sigma_x * sigma_x)).exp();
        }

        // Y place cells: 30 neurons, Gaussian tuning with sigma = height/6
        let sigma_y = (height as f32) / 6.0;
        let centers_y: Vec<f32> = (0..30).map(|i| (i as f32 + 0.5) * (height as f32) / 30.0).collect();
        for i in 0..30 {
            let dist = (y as f32 - centers_y[i]).abs();
            spike[30 + i] = (-dist * dist / (2.0 * sigma_y * sigma_y)).exp();
        }

        spike
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let enc = ArcTripartiteEncoder::new(30, 30, 1);
        assert_eq!(enc.width, 30);
        assert_eq!(enc.height, 30);
        assert_eq!(enc.scale_width, 1);
        assert_eq!(enc.base_dim(), 73);
        assert_eq!(enc.scaled_dim(), 73);
    }

    #[test]
    fn test_encoder_scaled_dim() {
        let enc = ArcTripartiteEncoder::new(30, 30, 4);
        assert_eq!(enc.base_dim(), 73);
        assert_eq!(enc.scaled_dim(), 292);
    }

    #[test]
    fn test_color_to_spike_one_hot() {
        let spike = ArcTripartiteEncoder::color_to_spike(0);
        assert_eq!(spike[0], 1.0);
        for i in 1..10 { assert_eq!(spike[i], 0.0); }

        let spike9 = ArcTripartiteEncoder::color_to_spike(9);
        assert_eq!(spike9[9], 1.0);
        for i in 0..9 { assert_eq!(spike9[i], 0.0); }
    }

    #[test]
    fn test_position_to_spike_centered() {
        let spike = ArcTripartiteEncoder::position_to_spike(15, 15, 30, 30);
        // X place cells: center of 30-wide grid is at 14.5
        // Y place cells: center of 30-high grid is at 14.5
        assert!(spike[15] > 0.5, "X place cell 15 should be highly active for x=15");
        assert!(spike[45] > 0.5, "Y place cell 15 should be highly active for y=15");
    }

    #[test]
    fn test_position_to_spike_corner() {
        let spike = ArcTripartiteEncoder::position_to_spike(0, 0, 30, 30);
        // Corner should activate cells near 0
        assert!(spike[0] > 0.5, "X place cell 0 should be highly active for x=0");
        assert!(spike[30] > 0.5, "Y place cell 0 should be highly active for y=0");
    }

    #[test]
    fn test_encode_boundary_guards() {
        let enc = ArcTripartiteEncoder::new(10, 10, 1);

        // Valid encode should succeed
        let code = enc.encode(5, 3, 7, ArcPhase::DemoInput);
        assert_eq!(code.len(), 73);

        // Invalid color should panic
        let result = std::panic::catch_unwind(|| enc.encode(10, 0, 0, ArcPhase::DemoInput));
        assert!(result.is_err(), "color=10 should panic");

        // Invalid x should panic
        let result = std::panic::catch_unwind(|| enc.encode(0, 10, 0, ArcPhase::DemoInput));
        assert!(result.is_err(), "x=10 should panic for width=10");

        // Invalid y should panic
        let result = std::panic::catch_unwind(|| enc.encode(0, 0, 10, ArcPhase::DemoOutput));
        assert!(result.is_err(), "y=10 should panic for height=10");
    }

    #[test]
    fn test_encode_phase_codes() {
        let enc = ArcTripartiteEncoder::new(10, 10, 1);

        let code_input = enc.encode(0, 0, 0, ArcPhase::DemoInput);
        let code_output = enc.encode(0, 0, 0, ArcPhase::DemoOutput);
        let code_test = enc.encode(0, 0, 0, ArcPhase::TestInput);

        // Context region is last 3 neurons per scale unit
        // For scale_width=1: indices 70, 71, 72
        assert_eq!(code_input[70], 1.0);
        assert_eq!(code_input[71], 0.0);
        assert_eq!(code_input[72], 0.0);

        assert_eq!(code_output[70], 0.0);
        assert_eq!(code_output[71], 1.0);
        assert_eq!(code_output[72], 0.0);

        assert_eq!(code_test[70], 0.0);
        assert_eq!(code_test[71], 0.0);
        assert_eq!(code_test[72], 1.0);
    }

    #[test]
    fn test_encode_grid_raster_order() {
        let enc = ArcTripartiteEncoder::new(2, 2, 1);
        let grid = vec![vec![0, 1], vec![2, 3]];
        let codes = enc.encode_grid(&grid, ArcPhase::DemoInput);
        assert_eq!(codes.len(), 4);

        // First pixel (0,0) color=0
        assert_eq!(codes[0][0], 1.0);
        // Second pixel (1,0) color=1
        assert_eq!(codes[1][1], 1.0);
        // Third pixel (0,1) color=2
        assert_eq!(codes[2][2], 1.0);
        // Fourth pixel (1,1) color=3
        assert_eq!(codes[3][3], 1.0);
    }

    #[test]
    fn test_encode_grid_dimension_mismatch() {
        let enc = ArcTripartiteEncoder::new(10, 10, 1);
        let grid = vec![vec![0, 1, 2], vec![3, 4, 5]]; // 3x2, not 10x10
        let result = std::panic::catch_unwind(|| enc.encode_grid(&grid, ArcPhase::DemoInput));
        assert!(result.is_err(), "Dimension mismatch should panic");
    }

    #[test]
    fn test_scale_width_affects_dim() {
        let enc1 = ArcTripartiteEncoder::new(10, 10, 1);
        let enc2 = ArcTripartiteEncoder::new(10, 10, 2);
        let code1 = enc1.encode(5, 3, 3, ArcPhase::DemoInput);
        let code2 = enc2.encode(5, 3, 3, ArcPhase::DemoInput);
        assert_eq!(code1.len(), 73);
        assert_eq!(code2.len(), 146);
        // Identity block (10 dims) is repeated at the start for each scale unit
        for i in 0..10 {
            assert_eq!(code1[i], code2[i], "Identity dim {} mismatch", i);
            assert_eq!(code1[i], code2[10 + i], "Identity dim {} not repeated", i);
        }
        // Spatial block (60 dims) is repeated after identity blocks
        for i in 0..60 {
            assert_eq!(code1[10 + i], code2[20 + i], "Spatial dim {} mismatch", i);
        }
        // Context block (3 dims) is repeated at the end
        for i in 0..3 {
            assert_eq!(code1[70 + i], code2[140 + i], "Context dim {} mismatch", i);
            assert_eq!(code1[70 + i], code2[143 + i], "Context dim {} not repeated", i);
        }
    }
}

