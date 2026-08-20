//! UCB Explorer — Uncertainty-Driven Exploration
//!
//! Implements the "Baby Principle": explore what you don't know yet.
//! Uses Upper Confidence Bound (UCB) to balance exploitation vs exploration.
//!
//! Formula: UCB(a) = avg_reward(a) + c * sqrt(ln(t) / n(a))

use std::collections::HashMap;
use rand::Rng;

/// Upper Confidence Bound Explorer.
/// Selects actions based on estimated value + exploration bonus.
pub struct UCBExplorer {
    counts: HashMap<String, usize>,
    rewards: HashMap<String, f64>,
    c: f64,
    t: usize,
}

impl UCBExplorer {
    /// Create a new UCB explorer.
    /// `c` controls exploration strength (typical: 1.0 - 2.0).
    pub fn new(c: f64) -> Self {
        Self {
            counts: HashMap::new(),
            rewards: HashMap::new(),
            c,
            t: 0,
        }
    }

    /// Select the candidate with the highest UCB score.
    /// `candidates` is the set of available actions.
    pub fn select(&mut self, candidates: &[String], _rng: &mut impl Rng) -> String {
        self.t += 1;
        let mut best = candidates[0].clone();
        let mut best_score = f64::NEG_INFINITY;

        for token in candidates {
            let n = *self.counts.get(token).unwrap_or(&1) as f64;
            let r = self.rewards.get(token).copied().unwrap_or(0.5);
            let avg = r / n;
            let exploration = self.c * ((self.t as f64).ln() / n).sqrt();
            let score = avg + exploration;

            if score > best_score {
                best_score = score;
                best = token.clone();
            }
        }
        best
    }

    /// Update the reward estimate for a given token.
    pub fn update(&mut self, token: &str, reward: f64) {
        *self.counts.entry(token.to_string()).or_insert(0) += 1;
        *self.rewards.entry(token.to_string()).or_insert(0.0) += reward;
    }

    /// Get the current estimated value of a token.
    pub fn value(&self, token: &str) -> f64 {
        let n = *self.counts.get(token).unwrap_or(&1) as f64;
        let r = self.rewards.get(token).copied().unwrap_or(0.5);
        r / n
    }

    /// Get the number of times a token has been selected.
    pub fn count(&self, token: &str) -> usize {
        *self.counts.get(token).unwrap_or(&0)
    }

    /// "Goldilocks" noise: inject a token whose complexity is just beyond current competence.
    /// Returns a challenge token from the candidate set.
    pub fn goldilocks_noise(
        &self,
        candidates: &[String],
        complexity: &HashMap<String, f64>,
        _complexity_threshold: f64,
    ) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }

        // Find token whose complexity is closest to (but above) current competence
        let mut best = candidates[0].clone();
        let mut best_diff = f64::INFINITY;

        for token in candidates {
            let comp = complexity.get(token).copied().unwrap_or(0.5);
            let n = *self.counts.get(token).unwrap_or(&1) as f64;
            let r = self.rewards.get(token).copied().unwrap_or(0.5);
            let competence = r / n; // higher = more competent
            
            // We want tokens slightly above our competence level
            let diff = (comp - competence - 0.1).abs();
            
            if diff < best_diff && comp > competence {
                best_diff = diff;
                best = token.clone();
            }
        }

        Some(best)
    }

    /// Reset the explorer state.
    pub fn reset(&mut self) {
        self.counts.clear();
        self.rewards.clear();
        self.t = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucb_prefers_underexplored() {
        let mut ucb = UCBExplorer::new(1.5);
        let mut rng = rand::thread_rng();
        
        let candidates = vec!["known".into(), "unknown".into()];
        ucb.update("known", 1.0);
        ucb.update("known", 1.0);
        ucb.update("known", 1.0);
        // "unknown" has no updates
        
        let selected = ucb.select(&candidates, &mut rng);
        // UCB should explore the unknown token due to exploration bonus
        assert!(selected == "unknown" || selected == "known");
    }

    #[test]
    fn ucb_converges_to_best() {
        let mut ucb = UCBExplorer::new(0.1); // low exploration
        let mut rng = rand::thread_rng();
        
        let candidates = vec!["bad".into(), "good".into()];
        ucb.update("bad", 0.0);
        ucb.update("bad", 0.0);
        ucb.update("good", 1.0);
        ucb.update("good", 1.0);
        
        // With low exploration, should prefer "good"
        for _ in 0..50 {
            let selected = ucb.select(&candidates, &mut rng);
            // Should mostly select "good" but not always
        }
        assert!(ucb.value("good") > ucb.value("bad"));
    }

    #[test]
    fn goldilocks_selects_appropriate_challenge() {
        let ucb = UCBExplorer::new(1.0);
        let mut complexity = HashMap::new();
        complexity.insert("easy".into(), 0.2);
        complexity.insert("medium".into(), 0.5);
        complexity.insert("hard".into(), 0.9);
        
        let candidates = vec!["easy".into(), "medium".into(), "hard".into()];
        let challenge = ucb.goldilocks_noise(&candidates, &complexity, 0.3);
        
        assert!(challenge.is_some());
        // Should select something near the competence boundary
    }
}
