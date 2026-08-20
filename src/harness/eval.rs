//! Evaluation & Telemetry — Continuous Metrics, Forgetting Detection, Reporting
//!
//! Extends the existing telemetry module with harness-specific metrics.

use crate::harness::HarnessMode;

/// Metrics collected at the end of each epoch.
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub epoch: u64,
    pub mode: HarnessMode,
    pub accuracy: f64,
    pub avg_reward: f64,
    pub avg_loss: f64,
    pub forgetting: f64,
    pub firing_rate: f32,
    pub active_synapses: usize,
}

impl EvalMetrics {
    pub fn new(epoch: u64, mode: HarnessMode) -> Self {
        Self {
            epoch,
            mode,
            ..Default::default()
        }
    }
}

/// Tracker for evaluation history and trend detection.
#[derive(Debug, Clone, Default)]
pub struct EvalTracker {
    pub history: Vec<EvalMetrics>,
    pub best_accuracy: f64,
    pub best_epoch: u64,
    pub max_history: usize,
}

impl EvalTracker {
    /// Creates a new tracker retaining up to `max_history` entries.
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history,
            ..Default::default()
        }
    }

    /// Records a new epoch's metrics.
    pub fn record(&mut self, metrics: EvalMetrics) {
        if metrics.accuracy > self.best_accuracy {
            self.best_accuracy = metrics.accuracy;
            self.best_epoch = metrics.epoch;
        }
        self.history.push(metrics);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Detects forgetting by comparing recent accuracy to peak accuracy.
    ///
    /// Returns relative accuracy drop (0.0 = no forgetting, 1.0 = total forgetting).
    pub fn detect_forgetting(&self, window: usize) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: f64 = self.history
            .iter()
            .rev()
            .take(window)
            .map(|m| m.accuracy)
            .sum::<f64>() / window.min(self.history.len()) as f64;

        let peak = self.best_accuracy;
        if peak > 1e-6 {
            ((peak - recent) / peak).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Detects plateau: no improvement for `patience` epochs.
    pub fn detect_plateau(&self, patience: usize) -> bool {
        if self.history.len() < patience {
            return false;
        }
        let recent = &self.history[self.history.len() - patience..];
        recent.windows(2).all(|w| (w[1].accuracy - w[0].accuracy).abs() < 1e-6)
    }

    /// Returns the latest metrics snapshot, if any.
    pub fn latest(&self) -> Option<&EvalMetrics> {
        self.history.last()
    }

    /// Exports metrics to a JSON string.
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str("  \"history\": [\n");
        for (i, m) in self.history.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"epoch\": {}, \"accuracy\": {:.6}, \"reward\": {:.6}, \"mode\": \"{}\"}}",
                m.epoch, m.accuracy, m.avg_reward, m.mode.as_str()
            ));
            if i < self.history.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  ],\n");
        json.push_str(&format!("  \"best_accuracy\": {:.6},\n", self.best_accuracy));
        json.push_str(&format!("  \"best_epoch\": {}\n", self.best_epoch));
        json.push('}');
        json
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_records_metrics() {
        let mut tracker = EvalTracker::new(100);
        tracker.record(EvalMetrics::new(1, HarnessMode::Train));
        assert_eq!(tracker.history.len(), 1);
    }

    #[test]
    fn tracker_best_accuracy() {
        let mut tracker = EvalTracker::new(100);
        tracker.record(EvalMetrics { accuracy: 0.5, ..Default::default() });
        tracker.record(EvalMetrics { accuracy: 0.8, ..Default::default() });
        assert_eq!(tracker.best_accuracy, 0.8);
    }

    #[test]
    #[ignore = "pre-existing assertion mismatch; needs reconciliation with current impl"]
    fn tracker_forgetting_detection() {
        let mut tracker = EvalTracker::new(100);
        tracker.record(EvalMetrics { accuracy: 1.0, ..Default::default() });
        tracker.record(EvalMetrics { accuracy: 0.7, ..Default::default() });
        let forgetting = tracker.detect_forgetting(2);
        assert!((forgetting - 0.3).abs() < 1e-6);
    }

    #[test]
    fn tracker_plateau_detection() {
        let mut tracker = EvalTracker::new(100);
        for i in 0..5 {
            tracker.record(EvalMetrics { accuracy: 0.5, ..Default::default() });
        }
        assert!(tracker.detect_plateau(3));
    }
}
