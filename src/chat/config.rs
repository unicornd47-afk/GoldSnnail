//! Chat Configuration — Centralized parameters for the SNN-LLM chat engine
//!
//! Eliminates magic numbers and provides type-safe geometry contracts.

use crate::geometry::HyperbolicPoint;
use ndarray::Array1;

// =============================================================================
// 1. ChatConfig — Global chat engine parameters
// =============================================================================

/// Centralized configuration for the chat engine.
#[derive(Debug, Clone, Copy)]
pub struct ChatConfig {
    /// Maximum number of conversation turns to keep in the buffer.
    pub conversation_capacity: usize,
    /// Spike rate multiplier for encoding (higher = more spikes per word).
    pub spike_rate: f32,
    /// Temporal window size (timesteps per word position).
    pub temporal_window: usize,
    /// World model latent dimension (must match lexicon embedding dimension).
    pub world_model_latent: usize,
    /// World model hidden layer size.
    pub world_model_hidden: usize,
    /// Preferred SIMD chunk size for parallel processing (L1-cache optimized).
    pub simd_chunk_size: usize,
}

impl ChatConfig {
    /// Default configuration for general-purpose chat.
    pub const DEFAULT: Self = Self {
        conversation_capacity: 50,
        spike_rate: 3.0,
        temporal_window: 5,
        world_model_latent: 2,
        world_model_hidden: 4,
        simd_chunk_size: 1024,
    };

    /// SHD-optimized configuration (stateless audio classification).
    pub const SHD_BENCHMARK: Self = Self {
        conversation_capacity: 0,
        spike_rate: 1.0,
        temporal_window: 32,
        world_model_latent: 2,
        world_model_hidden: 4,
        simd_chunk_size: 1024,
    };

    /// DVS128-optimized configuration (spatial gesture recognition).
    pub const DVS128_BENCHMARK: Self = Self {
        conversation_capacity: 0,
        spike_rate: 1.0,
        temporal_window: 64,
        world_model_latent: 2,
        world_model_hidden: 8,
        simd_chunk_size: 1024,
    };

    /// Validate that the world model latent dimension matches the lexicon dimension.
    pub fn validate_geometry(&self, lexicon_dim: usize) -> Result<(), GeometryError> {
        if self.world_model_latent != lexicon_dim {
            return Err(GeometryError::DimensionMismatch {
                expected: lexicon_dim,
                got: self.world_model_latent,
            });
        }
        Ok(())
    }
}

// =============================================================================
// 2. WorldGeometry — Explicit hyperbolic space contract
// =============================================================================

/// Explicit geometric parameters for the hyperbolic world model.
#[derive(Debug, Clone, Copy)]
pub struct WorldGeometry {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub curvature: f32,
}

impl WorldGeometry {
    pub const fn new(latent_dim: usize, hidden_dim: usize, curvature: f32) -> Self {
        Self { latent_dim, hidden_dim, curvature }
    }

    /// Validate against a known lexicon dimension.
    pub fn validate(&self, lexicon_dim: usize) -> Result<(), GeometryError> {
        if self.latent_dim != lexicon_dim {
            return Err(GeometryError::DimensionMismatch {
                expected: lexicon_dim,
                got: self.latent_dim,
            });
        }
        Ok(())
    }

    /// Create a `HyperbolicPoint` with zero coordinates of the correct dimension.
    pub fn zero_point(&self) -> HyperbolicPoint {
        HyperbolicPoint::new(Array1::from(vec![0.0f64; self.latent_dim]))
            .expect("zero point must be valid")
    }
}

// =============================================================================
// 3. Errors
// =============================================================================

#[derive(Debug, Clone)]
pub enum GeometryError {
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for GeometryError {}

// =============================================================================
// 4. Re-exports
// =============================================================================

pub use crate::LabError;
