//! Fractal Scaling — Equal Up/Down Scaling for 3-1-4-1 Networks
//!
//! Provides mathematically equal scaling where all axes double/halve together,
//! preserving the self-similar 3-1-4-1 pattern at every level.

use crate::harness::fractal_network::FractalNetwork;
use crate::harness::scale::ScaleProfile;

/// Scaling direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDir {
    Up,
    Down,
    Equal,
}

/// Scales a FractalNetwork equally in all dimensions.
///
/// Rules:
///   - width      doubles/halves: 1 -> 2 -> 4 -> 8 ...
///   - depth      doubles/halves: 1 -> 2 -> 4 -> 8 ...
///   - recurrence doubles/halves: 4 -> 8 -> 16 -> 32 ...
///   - number of layers doubles/halves
///   - plasticity stays at 1.0 (frozen core + trainable adapters)
///
/// The scaling is symmetric: up(x) then down(x) returns the original network.
pub fn scale_network(net: &FractalNetwork, factor: usize, dir: ScaleDir) -> FractalNetwork {
    let f = 1usize << factor.min(3);
    let base = net.layers[0].scale;
    let num_layers = net.layers.len();

    let (width, depth, recurrence) = match dir {
        ScaleDir::Up => (
            base.width * f,
            base.depth * f,
            base.recurrence * f,
        ),
        ScaleDir::Down => (
            base.width / f,
            base.depth / f,
            base.recurrence / f,
        ),
        ScaleDir::Equal => (
            base.width * f,
            base.depth * f,
            base.recurrence * f,
        ),
    };

    let scale = ScaleProfile {
        width: width.max(1),
        depth: depth.max(1),
        recurrence: recurrence.max(1),
        plasticity: 1.0,
    };

    let new_num_layers = match dir {
        ScaleDir::Up => num_layers * f,
        ScaleDir::Down => num_layers / f,
        ScaleDir::Equal => num_layers * f,
    };

    FractalNetwork::new(net.input_dim, net.output_dim, new_num_layers.max(1), scale)
}

/// Returns the scale factor needed to transform `from` to `to` network depths.
pub fn scale_factor(from_depth: usize, to_depth: usize) -> usize {
    if from_depth == 0 || to_depth == 0 {
        return 0;
    }
    let ratio = (to_depth as f64) / (from_depth as f64);
    (ratio.log2().round() as usize).max(0)
}

/// Scales network to match a target compute budget.
pub fn scale_to_budget(net: &FractalNetwork, budget: usize) -> FractalNetwork {
    let current_cost = net.compute_cost();
    if current_cost <= budget {
        // Scale up to fill budget
        let factor = (budget / current_cost.max(1)).ilog2() as usize;
        scale_network(net, factor.min(3), ScaleDir::Up)
    } else {
        // Scale down to fit budget
        let factor = (current_cost / budget.max(1)).ilog2() as usize;
        scale_network(net, factor.min(3), ScaleDir::Down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_up_doubles_dimensions() {
        let base = FractalNetwork::new(16, 16, 1, ScaleProfile::base());
        let scaled = scale_network(&base, 1, ScaleDir::Up);
        assert_eq!(scaled.layers.len(), 2);
        assert_eq!(scaled.layers[0].scale.width, 2);
        assert_eq!(scaled.layers[0].scale.recurrence, 8);
    }

    #[test]
    fn scale_down_halves_dimensions() {
        let base = FractalNetwork::new(16, 16, 4, ScaleProfile::base());
        let scaled = scale_network(&base, 1, ScaleDir::Down);
        assert_eq!(scaled.layers.len(), 2);
        assert_eq!(scaled.layers[0].scale.width, 1);
        assert_eq!(scaled.layers[0].scale.recurrence, 4);
    }

    #[test]
    fn equal_scaling_preserves_symmetry() {
        let base = FractalNetwork::new(8, 8, 2, ScaleProfile::base());
        let up = scale_network(&base, 1, ScaleDir::Up);
        assert_eq!(up.layers.len(), 4);
        let down = scale_network(&up, 1, ScaleDir::Down);
        assert_eq!(down.layers.len(), 2);
        assert_eq!(down.layers[0].scale.width, 1);
    }

    #[test]
    fn scale_factor_computation() {
        assert_eq!(scale_factor(1, 4), 2);
        assert_eq!(scale_factor(4, 1), 2);
        assert_eq!(scale_factor(2, 2), 0);
    }
}
