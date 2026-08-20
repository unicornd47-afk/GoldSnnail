//! Meta-Controller — Mode Switching, Checkpointing, Hyperparameter Adaptation
//!
//! The "brain of the harness" decides WHAT and HOW to learn.

use crate::harness::{HarnessMode, EvalMetrics};

/// Meta-controller configuration.
#[derive(Debug, Clone, Copy)]
pub struct MetaConfig {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Minimum learning rate.
    pub min_lr: f64,
    /// Initial noise standard deviation.
    pub initial_noise: f32,
    /// Minimum noise standard deviation.
    pub min_noise: f32,
    /// Noise annealing rate per epoch.
    pub noise_anneal_rate: f32,
    /// Exploration factor (epsilon for epsilon-greedy, entropy bonus).
    pub exploration: f64,
    /// Checkpoint every N epochs.
    pub checkpoint_interval: u64,
    /// Forgetting threshold to trigger rollback (relative accuracy drop).
    pub forgetting_threshold: f64,
    /// Plateau epochs before increasing exploration.
    pub plateau_patience: u64,
}

impl Default for MetaConfig {
    fn default() -> Self {
        Self {
            initial_lr: 0.01,
            min_lr: 0.0001,
            initial_noise: 0.5,
            min_noise: 0.01,
            noise_anneal_rate: 0.99,
            exploration: 0.3,
            checkpoint_interval: 50,
            forgetting_threshold: 0.2,
            plateau_patience: 10,
        }
    }
}

/// Meta-controller state and decision logic.
#[derive(Debug, Clone)]
pub struct MetaController {
    pub mode: HarnessMode,
    pub epoch: u64,
    pub lr: f64,
    pub noise_std: f32,
    pub exploration: f64,
    pub config: MetaConfig,
    pub best_accuracy: f64,
    pub plateau_counter: u64,
    pub last_checkpoint_epoch: u64,
}

impl MetaController {
    /// Creates a new meta-controller with default configuration.
    pub fn new() -> Self {
        Self::with_config(MetaConfig::default())
    }

    /// Creates a meta-controller with explicit configuration.
    pub fn with_config(config: MetaConfig) -> Self {
        Self {
            mode: HarnessMode::Train,
            epoch: 0,
            lr: config.initial_lr,
            noise_std: config.initial_noise,
            exploration: config.exploration,
            config,
            best_accuracy: 0.0,
            plateau_counter: 0,
            last_checkpoint_epoch: 0,
        }
    }

    /// Advances the epoch counter and updates hyperparameters.
    pub fn tick(&mut self) {
        self.epoch += 1;
        self.noise_std *= self.config.noise_anneal_rate as f32;
        self.noise_std = self.noise_std.max(self.config.min_noise);

        // Cosine-annealed learning rate (simplified)
        let progress = (self.epoch as f64 / 1000.0).min(1.0);
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.lr = self.config.min_lr + (self.config.initial_lr - self.config.min_lr) * cosine;
        self.lr = self.lr.max(self.config.min_lr);
    }

    /// Called at epoch end to update mode and detect forgetting/plateau.
    pub fn on_epoch_end(&mut self, metrics: &EvalMetrics) -> HarnessMode {
        let prev_mode = self.mode;

        // Update best accuracy
        if metrics.accuracy > self.best_accuracy {
            self.best_accuracy = metrics.accuracy;
            self.plateau_counter = 0;
        } else {
            self.plateau_counter += 1;
        }

        // Mode switching logic
        let new_mode = if metrics.forgetting > self.config.forgetting_threshold {
            // Forgetting detected -> consolidate
            HarnessMode::Consolidate
        } else if self.plateau_counter > self.config.plateau_patience {
            // Plateau -> increase exploration, switch to dream
            self.exploration = (self.exploration * 1.5).min(1.0);
            HarnessMode::Dream
        } else {
            // Normal training
            HarnessMode::Train
        };

        self.mode = new_mode;
        new_mode
    }

    /// Returns true if a checkpoint should be saved this epoch.
    pub fn should_checkpoint(&self) -> bool {
        self.epoch > 0
            && self.epoch >= self.last_checkpoint_epoch + self.config.checkpoint_interval
    }

    /// Marks that a checkpoint was saved at the current epoch.
    pub fn mark_checkpoint(&mut self, _path: impl Into<String>) {
        self.last_checkpoint_epoch = self.epoch;
    }

    /// Returns the current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Returns the current noise standard deviation.
    pub fn noise_std(&self) -> f32 {
        self.noise_std
    }

    /// Returns the current exploration factor.
    pub fn exploration(&self) -> f64 {
        self.exploration
    }
}

impl Default for MetaController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_controller_creation() {
        let meta = MetaController::new();
        assert_eq!(meta.mode, HarnessMode::Train);
    }

    #[test]
    fn meta_tick_advances_epoch() {
        let mut meta = MetaController::new();
        meta.tick();
        assert_eq!(meta.epoch, 1);
    }

    #[test]
    fn meta_on_epoch_end_detects_forgetting() {
        let mut meta = MetaController::new();
        let metrics = EvalMetrics {
            epoch: 1,
            mode: HarnessMode::Train,
            accuracy: 0.1,
            avg_reward: 0.0,
            avg_loss: 0.0,
            forgetting: 0.3,
            firing_rate: 10.0,
            active_synapses: 100,
        };
        let mode = meta.on_epoch_end(&metrics);
        assert_eq!(mode, HarnessMode::Consolidate);
    }

    #[test]
    fn meta_checkpoint_interval() {
        let mut meta = MetaController::new();
        meta.epoch = 50;
        assert!(meta.should_checkpoint());
        meta.mark_checkpoint("/tmp/ckpt");
        assert!(!meta.should_checkpoint());
    }
}
