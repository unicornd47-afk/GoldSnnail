//! Scale Profile — 4-dimensional scaling envelope for fractal layers
//!
//! Every FractalLayer is defined by four independent scale axes:
//!   Width      — parallel sub-units inside one layer
//!   Depth      — number of stacked FractalLayers
//!   Recurrence — timesteps the core processes per forward call
//!   Plasticity — learning-rate multiplier for adapter weights
//!
//! The frozen core itself is never modified; only adapters scale.

use serde::{Deserialize, Serialize};

/// 4-dimensional scaling profile for a fractal layer.
///
/// All values are positive integers or positive floats.
/// `None` means "inherit from parent scale".
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScaleProfile {
    /// Width: number of parallel sub-units (e.g., parallel SNN cores).
    /// Range: [1, 32].
    pub width: usize,

    /// Depth: number of stacked FractalLayers in this block.
    /// Range: [1, 8].
    pub depth: usize,

    /// Recurrence: timesteps processed per forward() call.
    /// More recurrence = longer temporal integration.
    /// Range: [1, 64].
    pub recurrence: usize,

    /// Plasticity: multiplier on the base learning rate for adapters.
    /// 1.0 = normal training, 0.0 = frozen adapters.
    pub plasticity: f64,
}

impl ScaleProfile {
    /// Base profile: single unit, single layer, minimal recurrence, full plasticity.
    pub const fn base() -> Self {
        Self {
            width: 1,
            depth: 1,
            recurrence: 4,
            plasticity: 1.0,
        }
    }

    /// Doubles width, keeps other axes constant.
    pub fn widen(self) -> Self {
        Self { width: self.width * 2, ..self }
    }

    /// Doubles depth, keeps other axes constant.
    pub fn deepen(self) -> Self {
        Self { depth: self.depth * 2, ..self }
    }

    /// Doubles recurrence, keeps other axes constant.
    pub fn prolong(self) -> Self {
        Self { recurrence: self.recurrence * 2, ..self }
    }

    /// Scales plasticity by factor, clamped to [0.0, 2.0].
    pub fn scale_plasticity(self, factor: f64) -> Self {
        Self {
            plasticity: (self.plasticity * factor).clamp(0.0, 2.0),
            ..self
        }
    }

    /// Returns total compute cost proxy (width x depth x recurrence).
    pub fn compute_cost(&self) -> usize {
        self.width * self.depth * self.recurrence
    }

    /// Returns whether this profile is within safe operating bounds.
    pub fn is_safe(&self) -> bool {
        self.width > 0
            && self.depth > 0
            && self.recurrence > 0
            && self.plasticity >= 0.0
            && self.compute_cost() <= 1024
    }
}

impl Default for ScaleProfile {
    fn default() -> Self {
        Self::base()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_profile_is_safe() {
        let p = ScaleProfile::base();
        assert!(p.is_safe());
        assert_eq!(p.compute_cost(), 1 * 1 * 4);
    }

    #[test]
    fn widen_doubles_width() {
        let p = ScaleProfile::base().widen();
        assert_eq!(p.width, 2);
        assert_eq!(p.depth, 1);
    }

    #[test]
    fn deepen_doubles_depth() {
        let p = ScaleProfile::base().deepen();
        assert_eq!(p.depth, 2);
        assert_eq!(p.width, 1);
    }

    #[test]
    fn prolong_doubles_recurrence() {
        let p = ScaleProfile::base().prolong();
        assert_eq!(p.recurrence, 8);
    }

    #[test]
    fn scale_plasticity_clamps() {
        let p = ScaleProfile::base().scale_plasticity(3.0);
        assert_eq!(p.plasticity, 2.0);
    }
}
