//! Replay Buffer — Ring Buffer with Prioritized & Balanced Sampling
//!
//! Stores transitions for off-policy learning. Three-level design:
//! 1. Episodic buffer: raw transitions
//! 2. Semantic cache: consolidated concepts (placeholder for Phase 3)
//! 3. Dream buffer: generated sequences (placeholder for Phase 3)
//!
//! DOD-compliant: flat `Vec<Transition>`, no `Box<dyn>`, usize indices.

use serde::{Deserialize, Serialize};

/// A single training transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Input spike indices for this timestep.
    pub input_spikes: Vec<usize>,
    /// Output spike indices for this timestep.
    pub output_spikes: Vec<usize>,
    /// Scalar reward received.
    pub reward: f64,
    /// Next-step input spikes.
    pub next_input_spikes: Vec<usize>,
    /// Next-step output spikes.
    pub next_output_spikes: Vec<usize>,
    /// Whether the episode terminated.
    pub done: bool,
    /// TD-error for prioritized sampling (updated during training).
    pub td_error: f64,
}

impl Transition {
    /// Creates a new transition with default TD-error.
    pub fn new(
        input_spikes: Vec<usize>,
        output_spikes: Vec<usize>,
        reward: f64,
        next_input_spikes: Vec<usize>,
        next_output_spikes: Vec<usize>,
        done: bool,
    ) -> Self {
        Self {
            input_spikes,
            output_spikes,
            reward,
            next_input_spikes,
            next_output_spikes,
            done,
            td_error: 1.0,
        }
    }
}

/// Sampling strategy for the replay buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Uniform random sampling.
    Uniform,
    /// Prioritized by absolute TD-error.
    Prioritized,
    /// Force equal class distribution (not implemented in Phase 1).
    Balanced,
}

/// Configuration for the replay buffer.
#[derive(Debug, Clone, Copy)]
pub struct ReplayConfig {
    /// Maximum number of transitions to store.
    pub capacity: usize,
    /// Default sampling strategy.
    pub default_strategy: SamplingStrategy,
    /// Priority exponent (alpha) for PER.
    pub priority_alpha: f64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            default_strategy: SamplingStrategy::Uniform,
            priority_alpha: 0.6,
        }
    }
}

/// Ring-buffer replay store with flat Vec storage.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    pub transitions: Vec<Transition>,
    pub priorities: Vec<f64>,
    pub config: ReplayConfig,
    pub position: usize,
    pub filled: usize,
}

impl ReplayBuffer {
    /// Creates a new replay buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self::with_config(ReplayConfig {
            capacity,
            ..Default::default()
        })
    }

    /// Creates a new replay buffer with explicit configuration.
    pub fn with_config(config: ReplayConfig) -> Self {
        let capacity = config.capacity;
        Self {
            transitions: Vec::with_capacity(capacity),
            priorities: Vec::with_capacity(capacity),
            config,
            position: 0,
            filled: 0,
        }
    }

    /// Pushes a transition into the buffer (ring overwrite when full).
    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() < self.config.capacity {
            self.transitions.push(transition);
            self.priorities.push(1.0);
        } else {
            self.transitions[self.position] = transition;
            self.priorities[self.position] = 1.0;
        }
        self.position = (self.position + 1) % self.config.capacity;
        self.filled = self.filled.min(self.config.capacity) + 1;
        if self.filled > self.config.capacity {
            self.filled = self.config.capacity;
        }
    }

    /// Returns the number of stored transitions.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Returns true if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.transitions.len() >= self.config.capacity
    }

    /// Samples a batch of transitions using the given strategy.
    pub fn sample(&self, batch_size: usize, strategy: SamplingStrategy) -> Vec<&Transition> {
        if self.transitions.is_empty() {
            return Vec::new();
        }

        let n = self.transitions.len();
        let batch_size = batch_size.min(n);

        match strategy {
            SamplingStrategy::Uniform => self.sample_uniform(batch_size),
            SamplingStrategy::Prioritized => self.sample_prioritized(batch_size),
            SamplingStrategy::Balanced => self.sample_uniform(batch_size), // Phase 1 fallback
        }
    }

    /// Updates priorities for sampled transitions based on TD-error.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        for (&idx, &td_err) in indices.iter().zip(td_errors.iter()) {
            if idx < self.priorities.len() {
                let priority = (td_err.abs() + 1e-6).powf(self.config.priority_alpha);
                self.priorities[idx] = priority;
            }
        }
    }

    /// Clears all stored transitions.
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.priorities.clear();
        self.position = 0;
        self.filled = 0;
    }

    /// Returns the capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    // -------------------------------------------------------------------------
    // Private sampling helpers
    // -------------------------------------------------------------------------

    fn sample_uniform(&self, batch_size: usize) -> Vec<&Transition> {
        let mut out = Vec::with_capacity(batch_size);
        let n = self.transitions.len();
        for _ in 0..batch_size {
            let idx = rand::random::<usize>() % n;
            out.push(&self.transitions[idx]);
        }
        out
    }

    fn sample_prioritized(&self, batch_size: usize) -> Vec<&Transition> {
        let n = self.transitions.len();
        let sum_p: f64 = self.priorities.iter().sum();
        if sum_p <= 0.0 || n == 0 {
            return self.sample_uniform(batch_size);
        }

        let mut out = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut r = rand::random::<f64>() * sum_p;
            let mut idx = 0;
            for (i, &p) in self.priorities.iter().enumerate().take(n) {
                r -= p;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
                idx = i;
            }
            out.push(&self.transitions[idx]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_push_and_len() {
        let mut buf = ReplayBuffer::new(4);
        buf.push(Transition::new(vec![0], vec![1], 1.0, vec![], vec![], false));
        buf.push(Transition::new(vec![2], vec![3], -1.0, vec![], vec![], true));
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());
        assert!(!buf.is_full());
    }

    #[test]
    fn replay_ring_overwrite() {
        let mut buf = ReplayBuffer::new(2);
        buf.push(Transition::new(vec![0], vec![1], 1.0, vec![], vec![], false));
        buf.push(Transition::new(vec![2], vec![3], 2.0, vec![], vec![], false));
        buf.push(Transition::new(vec![4], vec![5], 3.0, vec![], vec![], false));
        assert_eq!(buf.len(), 2);
        assert!(buf.is_full());
    }

    #[test]
    fn replay_sample_uniform() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0..10 {
            buf.push(Transition::new(vec![i], vec![i], i as f64, vec![], vec![], false));
        }
        let batch = buf.sample(3, SamplingStrategy::Uniform);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn replay_sample_prioritized() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0..10 {
            buf.push(Transition::new(vec![i], vec![i], i as f64, vec![], vec![], false));
        }
        let batch = buf.sample(3, SamplingStrategy::Prioritized);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn replay_clear() {
        let mut buf = ReplayBuffer::new(10);
        buf.push(Transition::new(vec![0], vec![1], 1.0, vec![], vec![], false));
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }
}
