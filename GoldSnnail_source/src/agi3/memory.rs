//! Working Memory for the ARC-AGI-3 Agent
//!
//! Maintains the agent's recent observations, actions, and rewards to support
//! sequential decision-making. Stores encoded observation embeddings in
//! hyperbolic space and SNN spike patterns for temporal context.

use crate::agi3::{Action, Observation};
use crate::geometry::{HyperbolicPoint, PoincareBall};

/// Working memory for the ARC-AGI-3 agent.
///
/// Maintains bounded buffers of recent experience: observations, actions,
/// rewards, hyperbolic state embeddings, and SNN spike patterns. When the
/// capacity is exceeded the oldest entries are evicted first.
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    capacity: usize,
    pub observations: Vec<Observation>,
    pub actions: Vec<Action>,
    pub rewards: Vec<f64>,
    pub hyperbolic_states: Vec<HyperbolicPoint>,
    pub spike_history: Vec<Vec<usize>>,
}

impl WorkingMemory {
    /// Creates a new working memory buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            observations: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            hyperbolic_states: Vec::with_capacity(capacity),
            spike_history: Vec::with_capacity(capacity),
        }
    }

    /// Pushes a new step into working memory.
    ///
    /// If the buffer is at capacity the oldest entry is evicted first.
    pub fn push(
        &mut self,
        obs: Observation,
        action: Action,
        reward: f64,
        h_state: HyperbolicPoint,
        spikes: Vec<usize>,
    ) {
        if self.observations.len() >= self.capacity {
            self.observations.remove(0);
            self.actions.remove(0);
            self.rewards.remove(0);
            self.hyperbolic_states.remove(0);
            self.spike_history.remove(0);
        }
        self.observations.push(obs);
        self.actions.push(action);
        self.rewards.push(reward);
        self.hyperbolic_states.push(h_state);
        self.spike_history.push(spikes);
    }

    /// Returns the most recent observation, if any.
    pub fn recent_observation(&self) -> Option<&Observation> {
        self.observations.last()
    }

    /// Returns the most recent hyperbolic state, if any.
    pub fn recent_state(&self) -> Option<&HyperbolicPoint> {
        self.hyperbolic_states.last()
    }

    /// Returns the last `n` hyperbolic states.
    pub fn last_n_states(&self, n: usize) -> Vec<&HyperbolicPoint> {
        let start = self.hyperbolic_states.len().saturating_sub(n);
        self.hyperbolic_states[start..].iter().collect()
    }

    /// Returns all stored hyperbolic states.
    pub fn state_sequence(&self) -> Vec<HyperbolicPoint> {
        self.hyperbolic_states.clone()
    }

    /// Number of steps stored in memory.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether memory is empty.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Clears all memory buffers.
    pub fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.hyperbolic_states.clear();
        self.spike_history.clear();
    }

    /// Detects an attractor state by checking whether the last 3 hyperbolic
    /// states are all within hyperbolic distance 0.1 of each other.
    ///
    /// Returns the current turn count (`self.len()`) when an attractor is
    /// detected, otherwise `None`.
    pub fn detect_attractor(&self) -> Option<usize> {
        if self.hyperbolic_states.len() < 3 {
            return None;
        }
        let ball = PoincareBall::new(-1.0);
        let last = &self.hyperbolic_states[self.hyperbolic_states.len() - 3..];
        for i in 0..3 {
            for j in i + 1..3 {
                if ball.distance(&last[i], &last[j]).ok()? > 0.1 {
                    return None;
                }
            }
        }
        Some(self.len())
    }

    /// Returns a human-readable summary of the memory contents.
    pub fn summary(&self) -> String {
        let last_reward = self.rewards.last().copied().unwrap_or(0.0);
        let last_spikes = self.spike_history.last().map(|v| v.len()).unwrap_or(0);
        format!(
            "WorkingMemory[{}/{}] last_reward={:.3} states={} spikes={}",
            self.len(),
            self.capacity,
            last_reward,
            self.hyperbolic_states.len(),
            last_spikes,
        )
    }
}

/// Compressed representation of a sequence of hyperbolic states.
///
/// Encodes each state by its coordinate mean, quantized to 8-bit. The first
/// 12 values are stored inline in `seed`; additional values overflow into
/// `residual`.
#[derive(Debug, Clone)]
pub struct MemorySeed {
    pub seed: [u8; 12],
    pub residual: Vec<u8>,
    pub checksum: u32,
}

impl MemorySeed {
    /// Compresses a sequence of hyperbolic states into a `MemorySeed`.
    ///
    /// For each state the mean of its coordinates is computed, clamped to
    /// `[-1.0, 1.0]`, and quantized to an 8-bit value. The first 12 values
    /// are stored in `seed` and any remaining values are stored in `residual`.
    pub fn compress(states: &[HyperbolicPoint]) -> Result<Self, String> {
        if states.is_empty() {
            return Err("Cannot compress empty state sequence".into());
        }

        let mut means = Vec::with_capacity(states.len());
        for state in states {
            if state.coords.is_empty() {
                return Err("Cannot compress state with zero coordinates".into());
            }
            let mean = state.coords.iter().sum::<f64>() / state.coords.len() as f64;
            if !mean.is_finite() {
                return Err(format!("Non-finite coordinate mean: {}", mean));
            }
            let clamped = mean.clamp(-1.0, 1.0);
            let quantized = ((clamped + 1.0) / 2.0 * 254.0).round() as u8;
            means.push(quantized);
        }

        let mut seed = [0u8; 12];
        let mut residual = Vec::new();

        for (i, val) in means.iter().enumerate() {
            if i < 12 {
                seed[i] = *val;
            } else {
                residual.push(*val);
            }
        }

        let checksum = seed
            .iter()
            .map(|&b| b as u32)
            .sum::<u32>()
            .wrapping_add(residual.iter().map(|&b| b as u32).sum::<u32>());

        Ok(Self {
            seed,
            residual,
            checksum,
        })
    }

    /// Decompresses the seed back into a sequence of hyperbolic states.
    ///
    /// Each reconstructed state has dimension `dim` with all coordinates
    /// set to the dequantized mean value.
    pub fn decompress(&self, dim: usize) -> Vec<HyperbolicPoint> {
        let mut values: Vec<f64> = self
            .seed
            .iter()
            .map(|&b| (b as f64 / 254.0) * 2.0 - 1.0)
            .collect();
        values.extend(
            self.residual
                .iter()
                .map(|&b| (b as f64 / 254.0) * 2.0 - 1.0),
        );

        values
            .into_iter()
            .map(|mean| {
                let mut coords = vec![0.0; dim];
                coords[0] = mean;
                let norm = coords.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm >= 1.0 {
                    let scale = 0.99 / norm;
                    for x in &mut coords {
                        *x *= scale;
                    }
                }
                HyperbolicPoint { coords }
            })
            .collect()
    }

    /// Computes cosine similarity between two `MemorySeed` instances.
    pub fn similarity(&self, other: &Self) -> f64 {
        let dot: f64 = self
            .seed
            .iter()
            .zip(other.seed.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let norm_a = self
            .seed
            .iter()
            .map(|&b| (b as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_b = other
            .seed
            .iter()
            .map(|&b| (b as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_evict_oldest() {
        let mut mem = WorkingMemory::new(2);
        let obs1 = Observation {
            frame: vec![vec![0]],
            width: 1,
            height: 1,
            turn: 0,
            episode_id: "e1".into(),
        };
        let obs2 = Observation {
            frame: vec![vec![1]],
            width: 1,
            height: 1,
            turn: 1,
            episode_id: "e1".into(),
        };
        let obs3 = Observation {
            frame: vec![vec![2]],
            width: 1,
            height: 1,
            turn: 2,
            episode_id: "e1".into(),
        };

        mem.push(obs1, Action::Noop, 0.0, HyperbolicPoint { coords: vec![0.1] }, vec![0]);
        mem.push(obs2, Action::Up, 1.0, HyperbolicPoint { coords: vec![0.2] }, vec![1]);
        mem.push(obs3, Action::Down, -1.0, HyperbolicPoint { coords: vec![0.3] }, vec![2]);

        assert_eq!(mem.len(), 2);
        assert_eq!(mem.recent_observation().unwrap().turn, 2);
        assert_eq!(mem.actions[0], Action::Up);
    }

    #[test]
    fn detect_attractor_with_close_states() {
        let mut mem = WorkingMemory::new(10);
        for coords in &[vec![0.1], vec![0.11], vec![0.09]] {
            mem.push(
                Observation {
                    frame: vec![vec![0]],
                    width: 1,
                    height: 1,
                    turn: 0,
                    episode_id: "e1".into(),
                },
                Action::Noop,
                0.0,
                HyperbolicPoint { coords: coords.clone() },
                vec![],
            );
        }
        assert!(mem.detect_attractor().is_some());
    }

    #[test]
    fn detect_attractor_with_distant_states() {
        let mut mem = WorkingMemory::new(10);
        mem.push(
            Observation {
                frame: vec![vec![0]],
                width: 1,
                height: 1,
                turn: 0,
                episode_id: "e1".into(),
            },
            Action::Noop,
            0.0,
            HyperbolicPoint { coords: vec![0.0] },
            vec![],
        );
        mem.push(
            Observation {
                frame: vec![vec![0]],
                width: 1,
                height: 1,
                turn: 0,
                episode_id: "e1".into(),
            },
            Action::Noop,
            0.0,
            HyperbolicPoint { coords: vec![0.5] },
            vec![],
        );
        mem.push(
            Observation {
                frame: vec![vec![0]],
                width: 1,
                height: 1,
                turn: 0,
                episode_id: "e1".into(),
            },
            Action::Noop,
            0.0,
            HyperbolicPoint { coords: vec![0.0] },
            vec![],
        );
        assert!(mem.detect_attractor().is_none());
    }

    #[test]
    fn memory_seed_round_trip() {
        let states = vec![
            HyperbolicPoint { coords: vec![0.1, 0.2] },
            HyperbolicPoint { coords: vec![0.3, 0.4] },
        ];
        let seed = MemorySeed::compress(&states).unwrap();
        let recovered = seed.decompress(2);
        assert_eq!(recovered.len(), 12 + seed.residual.len());
        let mean0 = recovered[0].coords[0];
        let mean1 = recovered[1].coords[0];
        assert!((mean0 - 0.15).abs() < 0.01);
        assert!((mean1 - 0.35).abs() < 0.01);
    }

    #[test]
    fn memory_seed_similarity() {
        let seed_a = MemorySeed {
            seed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            residual: vec![],
            checksum: 0,
        };
        let seed_b = MemorySeed {
            seed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            residual: vec![],
            checksum: 0,
        };
        let seed_c = MemorySeed {
            seed: [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            residual: vec![],
            checksum: 0,
        };
        assert!((seed_a.similarity(&seed_b) - 1.0).abs() < 1e-6);
        assert!(seed_a.similarity(&seed_c) < 1.0);
    }
}
