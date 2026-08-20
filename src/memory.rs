//! Working Memory — Sparse Persistent Activity via Recurrent Clusters
//!
//! Uses sparse `RecurrentSynapse` storage instead of dense matrices.
//! Complexity: O(size + recurrent.len()) per step, not O(size²).

use crate::geometry::Quaternion;
use crate::swarm::neuron::QLIFNeuron;

/// Sparse recurrent connection: only existing synapses are stored
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecurrentSynapse {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
}

/// Working Memory with sparse recurrent graph.
/// No O(n²) anymore — scales to 100k+ neurons on CPU.
pub struct WorkingMemory {
    pub neurons: Vec<QLIFNeuron>,
    /// Only existing synapses. DOD: flat Vec, no indirection.
    pub recurrent: Vec<RecurrentSynapse>,
    pub input_weights: Vec<f64>,
    pub size: usize,
    /// Spike buffer from last timestep for recurrent inputs
    pub last_spikes: Vec<bool>,
    /// History for attractor detection
    spike_history: Vec<Vec<bool>>,
    pub history_capacity: usize,
}

impl WorkingMemory {
    pub fn new(size: usize, _beta: f64, _threshold: f64) -> Self {
        let neurons = (0..size)
            .map(|_| QLIFNeuron::new(0.9, 1.0))
            .collect();
        Self {
            neurons,
            recurrent: Vec::new(),
            input_weights: vec![0.0; size],
            size,
            last_spikes: vec![false; size],
            spike_history: Vec::with_capacity(100),
            history_capacity: 100,
        }
    }

    /// Adds a single recurrent synapse.
    /// For ring attractor: `add_recurrent(i, (i+1)%n, 0.4)`
    pub fn add_recurrent(&mut self, from: usize, to: usize, weight: f64) {
        assert!(from < self.size && to < self.size, "Index out of bounds");
        self.recurrent.push(RecurrentSynapse { source: from, target: to, weight });
    }

    /// Batch init from a connectivity pattern
    pub fn build_recurrent<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, usize) -> Option<f64>,
    {
        self.recurrent.clear();
        for i in 0..self.size {
            for j in 0..self.size {
                if let Some(w) = f(i, j) {
                    self.recurrent.push(RecurrentSynapse { source: i, target: j, weight: w });
                }
            }
        }
    }

    pub fn set_input_weight(&mut self, neuron: usize, weight: f64) {
        if neuron < self.size {
            self.input_weights[neuron] = weight;
        }
    }

    /// One timestep. O(size + recurrent.len()), not O(size²).
    pub fn step(
        &mut self,
        external_input: &[Quaternion],
        dt_ms: f64,
        _current_time_ms: f64,
    ) -> Vec<bool> {
        let mut spikes = vec![false; self.size];

        // 1. Collect recurrent inputs from last_spikes (PREVIOUS step!)
        // This prevents spikes from being forwarded in the same tick
        let mut recurrent_input = vec![Quaternion::new(0.0, 0.0, 0.0, 0.0); self.size];
        
        for syn in &self.recurrent {
            if self.last_spikes[syn.source] {
                let q = Quaternion::new(syn.weight as f32, 0.0, 0.0, 0.0);
                recurrent_input[syn.target] = recurrent_input[syn.target] + q;
            }
        }

        // 2. Update all neurons
        for i in 0..self.size {
            let mut total = external_input
                .get(i)
                .copied()
                .unwrap_or_else(|| Quaternion::new(0.0, 0.0, 0.0, 0.0));
            
            total = total * self.input_weights[i] as f32;
            total = total + recurrent_input[i];

            if self.neurons[i].step(&total, dt_ms, _current_time_ms).is_some() {
                spikes[i] = true;
            }
        }

        // 3. History & State Update
        self.last_spikes = spikes.clone();
        
        if self.spike_history.len() >= self.history_capacity {
            self.spike_history.remove(0);
        }
        self.spike_history.push(spikes.clone());

        spikes
    }

    pub fn reset(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
        self.last_spikes.fill(false);
        self.spike_history.clear();
    }

    /// Detects stable attractor states (>30% firing rate over window)
    pub fn detect_attractor(&self) -> Option<Vec<usize>> {
        if self.spike_history.len() < 5 {
            return None;
        }
        let window = self.spike_history.iter().rev().take(10).collect::<Vec<_>>();
        let mut counts = vec![0usize; self.size];
        
        for frame in &window {
            for (i, &fired) in frame.iter().enumerate() {
                if fired { counts[i] += 1; }
            }
        }
        
        let threshold = window.len() / 3;
        let active: Vec<_> = counts.iter().enumerate()
            .filter(|&(_, &c)| c >= threshold)
            .map(|(i, _)| i)
            .collect();
        
        if active.len() >= 2 { Some(active) } else { None }
    }

    /// Metrics for the current window
    pub fn firing_rate(&self) -> f64 {
        if self.spike_history.is_empty() { return 0.0; }
        let last = self.spike_history.last().unwrap();
        last.iter().filter(|&&b| b).count() as f64 / self.size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_memory_scales_to_4096() {
        let mut mem = WorkingMemory::new(4096, 0.9, 1.0);
        // Only 0.5% connectivity = ~84k synapses instead of 16M
        for i in 0..4096 {
            let target = (i + 1) % 4096;
            mem.add_recurrent(i, target, 0.15);
            if i % 2 == 0 {
                mem.add_recurrent(i, (i + 7) % 4096, 0.05);
            }
        }
        let input = vec![Quaternion::new(0.5, 0.0, 0.0, 0.0); 4096];
        let spikes = mem.step(&input, 1.0, 0.0);
        assert_eq!(spikes.len(), 4096);
    }

    #[test]
    fn ring_attractor_propagates_wave() {
        let mut mem = WorkingMemory::new(8, 0.9, 0.5);
        for i in 0..8 {
            mem.add_recurrent(i, (i + 1) % 8, 20.0);
        }
        mem.set_input_weight(0, 1.0);
        
        let mut fired_count = vec![0usize; 8];
        for t in 0..50 {
            let input = if t == 0 {
                vec![Quaternion::new(10.0, 0.0, 0.0, 0.0); 8]
            } else {
                vec![Quaternion::new(0.0, 0.0, 0.0, 0.0); 8]
            };
            let spikes = mem.step(&input, 1.0, t as f64);
            for (i, &fired) in spikes.iter().enumerate() {
                if fired { fired_count[i] += 1; }
            }
        }
        
        assert!(fired_count.iter().all(|&c| c > 0), "Wave should visit all neurons, counts: {:?}", fired_count);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut mem = WorkingMemory::new(16, 0.9, 1.0);
        mem.add_recurrent(0, 1, 0.5);
        let input = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); 16];
        let _ = mem.step(&input, 1.0, 0.0);
        mem.reset();
        assert!(mem.last_spikes.iter().all(|&b| !b));
        assert!(mem.spike_history.is_empty());
    }
}
