//! Telemetry — Passive Observer for Avalanche Metrics & Criticality
//!
//! This module is strictly passive. It observes spike counts and records
//! avalanche metrics without mutating simulation state.

pub mod entropy;
pub mod avalanche;
pub mod avalanche_sim;
pub mod power_law;

pub use avalanche_sim::{simulate_avalanche, generate_avalanche_distribution};
pub use power_law::{PowerLawObserver, PowerLawFit};

/// Metrics collected for a single avalanche window.
#[derive(Debug, Clone, Default)]
pub struct AvalancheMetrics {
    /// Total spikes observed in the window.
    pub total_spikes: u64,
    /// Mean activity across all neurons.
    pub mean_activity: f32,
    /// Criticality index (branching ratio proxy).
    pub criticality_index: f32,
    /// Shannon entropy of the spike distribution.
    pub entropy: f32,
}

/// Passive observer that records avalanche metrics over time.
///
/// Note: `history` stores `AvalancheMetrics` structs. This is explicitly exempt
/// from the DOD flat-memory rule because telemetry is not hot-path simulation
/// state; it is bounded observation data collected at step boundaries.
#[derive(Debug, Clone)]
pub struct TelemetryObserver {
    /// Historical metrics snapshots.
    pub history: Vec<AvalancheMetrics>,
    /// Maximum number of snapshots to retain.
    pub max_history: usize,
}

impl TelemetryObserver {
    /// Creates a new observer retaining up to `max_history` snapshots.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Records a new metrics entry for a window with `spike_count` spikes.
    pub fn record(&mut self, spike_count: usize) {
        let metrics = AvalancheMetrics {
            total_spikes: spike_count as u64,
            mean_activity: 0.0,
            criticality_index: 0.0,
            entropy: 0.0,
        };
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(metrics);
    }

    /// Returns the latest metrics snapshot, if any.
    pub fn snapshot(&self) -> Option<&AvalancheMetrics> {
        self.history.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_record_and_snapshot() {
        let mut obs = TelemetryObserver::new(4);
        obs.record(10);
        obs.record(20);
        let snap = obs.snapshot().unwrap();
        assert_eq!(snap.total_spikes, 20);
    }

    #[test]
    fn telemetry_bounded_history() {
        let mut obs = TelemetryObserver::new(2);
        obs.record(1);
        obs.record(2);
        obs.record(3);
        assert_eq!(obs.history.len(), 2);
        assert_eq!(obs.history[0].total_spikes, 2);
    }
}
