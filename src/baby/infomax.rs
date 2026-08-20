//! Infomax Reward — Mutual Information as Intrinsic Curiosity
//!
//! Implements the "Baby Principle": learn not by prediction error,
//! but by maximizing information flow between sensor and hidden state.
//!
//! Reward = ΔMI(sensor; hidden)
//! High reward when the agent discovers new structure in its environment.

use ndarray::{Array1, Array2};

/// Mutual Information estimator with histogram binning.
/// Tracks joint and marginal distributions of (sensor, hidden) pairs.
pub struct InfomaxReward {
    joint: Array2<f64>,
    sensor_marginal: Array1<f64>,
    hidden_marginal: Array1<f64>,
    total: f64,
    bins: usize,
    prev_mi: f64,
}

impl InfomaxReward {
    /// Create a new InfomaxReward with the given number of bins.
    pub fn new(bins: usize) -> Self {
        Self {
            joint: Array2::zeros((bins, bins)),
            sensor_marginal: Array1::zeros(bins),
            hidden_marginal: Array1::zeros(bins),
            total: 0.0,
            bins,
            prev_mi: 0.0,
        }
    }

    /// Observe a (sensor, hidden) pair and update histograms.
    /// `sensor` and `hidden` should be normalized to [-1, 1].
    pub fn observe(&mut self, sensor: f64, hidden: f64) {
        let s = ((sensor + 1.0) / 2.0 * (self.bins as f64 - 1.0)).clamp(0.0, self.bins as f64 - 1.0) as usize;
        let h = ((hidden + 1.0) / 2.0 * (self.bins as f64 - 1.0)).clamp(0.0, self.bins as f64 - 1.0) as usize;
        
        self.joint[[s, h]] += 1.0;
        self.sensor_marginal[s] += 1.0;
        self.hidden_marginal[h] += 1.0;
        self.total += 1.0;
    }

    /// Compute current Mutual Information I(Sensor; Hidden).
    pub fn mi(&self) -> f64 {
        if self.total == 0.0 {
            return 0.0;
        }
        
        let mut sum = 0.0;
        for s in 0..self.bins {
            for h in 0..self.bins {
                let p_j = self.joint[[s, h]] / self.total;
                let p_s = self.sensor_marginal[s] / self.total;
                let p_h = self.hidden_marginal[h] / self.total;
                
                if p_j > 1e-12 && p_s > 1e-12 && p_h > 1e-12 {
                    sum += p_j * (p_j / (p_s * p_h)).ln();
                }
            }
        }
        sum
    }

    /// Observe and return the change in MI as reward.
    /// Positive reward = increasing information flow (discovery).
    pub fn reward_delta(&mut self, sensor: f64, hidden: f64) -> f64 {
        let mi_before = self.mi();
        self.observe(sensor, hidden);
        let mi_after = self.mi();
        
        let delta = mi_after - mi_before;
        self.prev_mi = mi_after;
        delta
    }

    /// Get current MI value.
    pub fn current_mi(&self) -> f64 {
        self.prev_mi
    }

    /// Reset the estimator (e.g., at episode boundary).
    pub fn reset(&mut self) {
        self.joint.fill(0.0);
        self.sensor_marginal.fill(0.0);
        self.hidden_marginal.fill(0.0);
        self.total = 0.0;
        self.prev_mi = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn infomax_rewards_structured_correlation() {
        let mut infomax = InfomaxReward::new(10);
        
        // Highly correlated sensor-hidden pairs should yield positive MI
        for i in 0..100 {
            let s = (i % 10) as f64 / 10.0;
            let h = s; // perfect correlation
            infomax.reward_delta(s, h);
        }
        
        assert!(infomax.current_mi() > 0.1, "Correlated signals should have positive MI");
    }

    #[test]
    fn infomax_low_for_independent_signals() {
        use rand::thread_rng;
        let mut rng = thread_rng();
        let mut infomax = InfomaxReward::new(10);
        
        // Truly independent signals
        for _ in 0..200 {
            let s = rng.r#gen::<f64>();
            let h = rng.r#gen::<f64>();
            infomax.reward_delta(s, h);
        }
        
        assert!(infomax.current_mi() < 0.5, "Independent signals should have low MI, got {}", infomax.current_mi());
    }

    #[test]
    fn reward_delta_positive_on_discovery() {
        let mut infomax = InfomaxReward::new(5);
        
        // First observations should give positive reward (new information)
        let r1 = infomax.reward_delta(0.5, 0.5);
        let r2 = infomax.reward_delta(0.3, 0.3);
        
        assert!(r1 >= 0.0, "Initial discovery should give non-negative reward");
        assert!(r2 >= 0.0, "Initial discovery should give non-negative reward");
    }

    #[test]
    fn reset_clears_state() {
        let mut infomax = InfomaxReward::new(5);
        infomax.reward_delta(0.5, 0.5);
        infomax.reset();
        
        assert_eq!(infomax.total, 0.0);
        assert_eq!(infomax.current_mi(), 0.0);
    }
}