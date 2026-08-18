//! Transitional Probability Learner — Grammar Acquisition from Sequences
//!
//! Implements the "Baby Principle": segment the world by learning
//! transition probabilities P(next | current) from observed sequences.
//!
//! This is the foundation of language acquisition: children learn
//! grammar by tracking which words follow which other words.

use std::collections::HashMap;
use rand::Rng;

/// Transitional Probability Learner.
/// Learns P(next | current) from observed sequences.
pub struct TransitionalLearner {
    /// counts[current][next] = number of times next followed current
    counts: HashMap<String, HashMap<String, usize>>,
    /// totals[current] = total transitions from current
    totals: HashMap<String, usize>,
}

impl TransitionalLearner {
    /// Create a new TransitionalLearner with empty statistics.
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            totals: HashMap::new(),
        }
    }

    /// Observe a sequence and update transition counts.
    pub fn observe(&mut self, sequence: &[String]) {
        for window in sequence.windows(2) {
            let current = &window[0];
            let next = &window[1];
            
            *self.counts.entry(current.clone()).or_default()
                .entry(next.clone()).or_insert(0) += 1;
            *self.totals.entry(current.clone()).or_insert(0) += 1;
        }
    }

    /// Get P(next | current).
    pub fn probability(&self, current: &str, next: &str) -> f64 {
        let c = self.counts.get(current)
            .and_then(|m| m.get(next)).copied().unwrap_or(0);
        let t = *self.totals.get(current).unwrap_or(&1).max(&1);
        c as f64 / t as f64
    }

    /// Get the total count for a current token.
    pub fn count(&self, current: &str) -> usize {
        *self.totals.get(current).unwrap_or(&0)
    }

    /// Generate a sequence starting from `start` using learned probabilities.
    pub fn generate(&self, start: &str, length: usize, rng: &mut impl Rng) -> Vec<String> {
        let mut seq = vec![start.to_string()];
        for _ in 0..length - 1 {
            let current = seq.last().unwrap();
            let next = self.sample_next(current, rng);
            seq.push(next);
        }
        seq
    }

    /// Sample next token according to learned probabilities.
    fn sample_next(&self, current: &str, rng: &mut impl Rng) -> String {
        let map = match self.counts.get(current) {
            Some(m) if !m.is_empty() => m,
            _ => return "???".to_string(),
        };
        
        let total: usize = map.values().sum();
        let mut thresh = rng.r#gen::<f64>() * total as f64;
        
        for (word, count) in map {
            thresh -= *count as f64;
            if thresh <= 0.0 {
                return word.clone();
            }
        }
        map.keys().next().unwrap().clone()
    }

    /// Get the most likely next token for a given current token.
    pub fn most_likely_next(&self, current: &str) -> Option<String> {
        let map = self.counts.get(current)?;
        map.iter().max_by_key(|(_, count)| *count).map(|(s, _)| s.clone())
    }

    /// Get transition entropy for a token (higher = more uncertain next word).
    pub fn entropy(&self, current: &str) -> f64 {
        let map = match self.counts.get(current) {
            Some(m) if !m.is_empty() => m,
            _ => return 0.0,
        };
        
        let total: f64 = map.values().sum::<usize>() as f64;
        if total == 0.0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &count in map.values() {
            let p = count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Reset the learner state.
    pub fn reset(&mut self) {
        self.counts.clear();
        self.totals.clear();
    }

    /// Get number of learned transitions.
    pub fn size(&self) -> usize {
        self.counts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_simple_sequence() {
        let mut learner = TransitionalLearner::new();
        learner.observe(&vec!["a".into(), "b".into(), "c".into()]);
        learner.observe(&vec!["a".into(), "b".into(), "d".into()]);
        
        assert_eq!(learner.probability("a", "b"), 1.0);
        assert_eq!(learner.probability("b", "c"), 0.5);
        assert_eq!(learner.probability("b", "d"), 0.5);
    }

    #[test]
    fn generates_grammatical_sequences() {
        let mut learner = TransitionalLearner::new();
        for _ in 0..100 {
            learner.observe(&vec!["the".into(), "cat".into(), "sees".into(), "mouse".into()]);
        }
        
        let mut rng = rand::thread_rng();
        let seq = learner.generate("the", 4, &mut rng);
        
        assert_eq!(seq[0], "the");
        for window in seq.windows(2) {
            let p = learner.probability(&window[0], &window[1]);
            assert!(p > 0.0, "Generated sequence should follow learned transitions");
        }
    }

    #[test]
    fn entropy_reflects_uncertainty() {
        let mut learner = TransitionalLearner::new();
        
        learner.observe(&vec!["a".into(), "b".into()]);
        learner.observe(&vec!["a".into(), "b".into()]);
        
        learner.observe(&vec!["x".into(), "y".into()]);
        learner.observe(&vec!["x".into(), "z".into()]);
        
        assert!(learner.entropy("a") < learner.entropy("x"));
    }

    #[test]
    fn most_likely_next_returns_best() {
        let mut learner = TransitionalLearner::new();
        learner.observe(&vec!["a".into(), "b".into()]);
        learner.observe(&vec!["a".into(), "b".into()]);
        learner.observe(&vec!["a".into(), "c".into()]);
        
        assert_eq!(learner.most_likely_next("a"), Some("b".into()));
    }

    #[test]
    fn reset_clears_state() {
        let mut learner = TransitionalLearner::new();
        learner.observe(&vec!["a".into(), "b".into()]);
        learner.reset();
        
        assert_eq!(learner.size(), 0);
        assert_eq!(learner.probability("a", "b"), 0.0);
    }
}
