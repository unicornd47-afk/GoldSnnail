//! Reward Engine — Extrinsic + Intrinsic (Curiosity) Reward
//!
//! Computes multi-faceted reward signals:
//! - Extrinsic: task-specific performance (grid similarity, classification accuracy)
//! - Intrinsic: prediction error, novelty, progress

use crate::vision::arc_loader::ArcGrid;

/// Reward weights balancing extrinsic and intrinsic signals.
#[derive(Debug, Clone, Copy)]
pub struct RewardWeights {
    pub extrinsic: f64,
    pub intrinsic_prediction_error: f64,
    pub intrinsic_novelty: f64,
    pub intrinsic_progress: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            extrinsic: 1.0,
            intrinsic_prediction_error: 0.2,
            intrinsic_novelty: 0.1,
            intrinsic_progress: 0.3,
        }
    }
}

/// Reward engine for computing training signals.
#[derive(Debug, Clone)]
pub struct RewardEngine {
    pub weights: RewardWeights,
    pub last_reward: f64,
    pub best_reward: f64,
}

impl RewardEngine {
    /// Creates a new reward engine with default weights.
    pub fn new() -> Self {
        Self::with_weights(RewardWeights::default())
    }

    /// Creates a reward engine with explicit weights.
    pub fn with_weights(weights: RewardWeights) -> Self {
        Self {
            weights,
            last_reward: 0.0,
            best_reward: 0.0,
        }
    }

    /// Computes grid similarity reward (exact match or cell-wise accuracy).
    pub fn grid_similarity(&self, predicted: &ArcGrid, expected: &ArcGrid) -> f64 {
        if predicted.width != expected.width || predicted.height != expected.height {
            return -1.0;
        }
        let total = (predicted.width * predicted.height).max(1);
        let mut correct = 0usize;
        for r in 0..predicted.height {
            for c in 0..predicted.width {
                if predicted.data[r][c] == expected.data[r][c] {
                    correct += 1;
                }
            }
        }
        (correct as f64) / (total as f64)
    }

    /// Computes intrinsic prediction-error reward based on spike count change.
    pub fn prediction_error(&self, prev_spikes: usize, curr_spikes: usize) -> f64 {
        let diff = (curr_spikes as f64 - prev_spikes as f64).abs();
        (-diff / 50.0).exp() // decays with large changes
    }

    /// Computes novelty reward (inverse of frequency — simplified as random baseline).
    pub fn novelty(&self) -> f64 {
        rand::random::<f64>() * 0.2
    }

    /// Computes progress reward (improvement over last reward).
    pub fn progress(&self, current_reward: f64) -> f64 {
        let delta = current_reward - self.last_reward;
        delta.clamp(-1.0, 1.0)
    }

    /// Computes the total reward for a task step.
    pub fn compute_total(
        &mut self,
        extrinsic: f64,
        prev_spike_count: usize,
        curr_spike_count: usize,
    ) -> f64 {
        let pred_err = self.prediction_error(prev_spike_count, curr_spike_count);
        let novelty = self.novelty();
        let progress = self.progress(extrinsic);

        let total = self.weights.extrinsic * extrinsic
            + self.weights.intrinsic_prediction_error * pred_err
            + self.weights.intrinsic_novelty * novelty
            + self.weights.intrinsic_progress * progress;

        self.last_reward = extrinsic;
        if extrinsic > self.best_reward {
            self.best_reward = extrinsic;
        }

        total.clamp(-2.0, 2.0)
    }

    /// Resets episode-level tracking.
    pub fn reset_episode(&mut self) {
        self.last_reward = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::arc_loader::ArcGrid;

    #[test]
    fn grid_similarity_exact_match() {
        let engine = RewardEngine::new();
        let a = ArcGrid::from_data(vec![vec![0, 1], vec![1, 0]]).unwrap();
        let b = a.clone();
        assert_eq!(engine.grid_similarity(&a, &b), 1.0);
    }

    #[test]
    fn grid_similarity_mismatch() {
        let engine = RewardEngine::new();
        let a = ArcGrid::from_data(vec![vec![0, 1], vec![1, 0]]).unwrap();
        let b = ArcGrid::from_data(vec![vec![0, 0], vec![0, 0]]).unwrap();
        let sim = engine.grid_similarity(&a, &b);
        assert!((sim - 0.5).abs() < 1e-10);
    }

    #[test]
    fn reward_total_bounded() {
        let mut engine = RewardEngine::new();
        let total = engine.compute_total(1.0, 10, 20);
        assert!(total <= 2.0);
        assert!(total >= -2.0);
    }
}
