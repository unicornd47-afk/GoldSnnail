//! Substrate — Flat Memory Arenas & Data-Oriented Design
//!
//! This module embodies DOD principles: neurons are identified by `usize`
//! indices. No pointers, no `Box<dyn Trait>`, no heap fragmentation in hot paths.

pub mod avx2;

#[repr(C)]
#[derive(Debug, Clone)]
/// Flat arena storing per-neuron state in parallel arrays.
///
/// Neurons are identified by `usize` indices. No pointers, no Boxed traits.
pub struct StateArena {
    /// Flat array of membrane potentials.
    pub membrane: Vec<f32>,
    /// Flat array of recovery variables (e.g., u in Izhikevich).
    pub recovery: Vec<f32>,
    /// Flat array of firing thresholds.
    pub threshold: Vec<f32>,
    /// Flat array of refractory countdown timers.
    pub refractory: Vec<u32>,
}

impl StateArena {
    /// Creates a new arena with `capacity` neurons, initialised to default values.
    pub fn new(capacity: usize) -> Self {
        Self {
            membrane: vec![0.0; capacity],
            recovery: vec![0.0; capacity],
            threshold: vec![-55.0; capacity],
            refractory: vec![0; capacity],
        }
    }

    /// Grows all arrays by `additional` elements, initialised to default values.
    pub fn extend(&mut self, additional: usize) {
        let len = self.membrane.len();
        self.membrane.resize(len + additional, 0.0);
        self.recovery.resize(len + additional, 0.0);
        self.threshold.resize(len + additional, -55.0);
        self.refractory.resize(len + additional, 0);
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
/// Row-major flat weight matrix.
pub struct WeightMatrix {
    /// Flat storage for all weights (row-major).
    pub data: Vec<f32>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl WeightMatrix {
    /// Creates a new matrix of shape `rows x cols`, filled with zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Returns the linear index for `(row, col)`.
    #[inline(always)]
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    /// Returns the weight at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Sets the weight at `(row, col)`.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Returns an immutable slice for the entire row.
    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Returns a mutable slice for the entire row.
    pub fn row_mut(&mut self, row: usize) -> &mut [f32] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Computes the dot product of row `row` with `other` using AVX2 if available.
    pub fn dot_row(&self, row: usize, other: &[f32]) -> Option<f32> {
        if other.len() != self.cols {
            return None;
        }
        let a = self.row(row);
        crate::substrate::avx2::dot_product(a, other)
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
/// Fixed-capacity buffer for spike indices emitted in a single timestep.
pub struct SpikeBuffer {
    /// Indices of neurons that spiked in this timestep.
    pub indices: Vec<u32>,
    /// Maximum number of spikes this buffer can hold.
    pub count: usize,
}

impl SpikeBuffer {
    /// Creates a new buffer with capacity `max_spikes`.
    pub fn new(max_spikes: usize) -> Self {
        Self {
            indices: Vec::with_capacity(max_spikes),
            count: max_spikes,
        }
    }

    /// Clears all recorded spikes.
    pub fn clear(&mut self) {
        self.indices.clear();
    }

    /// Records a spike for `neuron_idx`.
    ///
    /// Returns `Err` if the buffer is already full.
    pub fn push(&mut self, neuron_idx: u32) -> Result<(), &'static str> {
        if self.indices.len() >= self.count {
            return Err("SpikeBuffer full");
        }
        self.indices.push(neuron_idx);
        Ok(())
    }

    /// Returns an iterator over recorded spike indices.
    pub fn iter(&self) -> impl Iterator<Item = &u32> {
        self.indices.iter()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// New-type wrapper for neuron identifiers.
pub struct NeuronIdx(pub usize);

#[repr(C)]
#[derive(Debug, Clone)]
/// Minimal spike event for inter-node / inter-GPU transmission.
pub struct SpikeEvent {
    /// Source neuron.
    pub src: NeuronIdx,
    /// Destination neuron.
    pub dst: NeuronIdx,
    /// Synaptic delay in ticks.
    pub delay_ticks: u16,
    /// Quantised amplitude (0-255).
    pub amplitude_u8: u8,
    /// Flag byte (reserved for future use).
    pub flags: u8,
}

// =============================================================================
// ChatArena — Flat storage for chat engine objects
// =============================================================================

/// Flat arena for chat engine objects.
///
/// Each object type is stored in its own Vec, indexed by `usize`.
/// This follows DOD principles: flat arrays, usize indices, no Box<dyn Trait>.
pub struct ChatArena {
    pub trainers: Vec<crate::semantics::SemanticTrainer>,
    pub encoders: Vec<crate::chat::spike_token_bridge::TokenSpikeEncoder>,
    pub decoders: Vec<crate::chat::spike_token_bridge::SpikeTokenDecoder>,
}

impl ChatArena {
    /// Creates a new empty arena.
    pub fn new() -> Self {
        Self {
            trainers: Vec::new(),
            encoders: Vec::new(),
            decoders: Vec::new(),
        }
    }

    /// Pushes a (trainer, encoder, decoder) triple and returns the shared index.
    pub fn push(
        &mut self,
        trainer: crate::semantics::SemanticTrainer,
        encoder: crate::chat::spike_token_bridge::TokenSpikeEncoder,
        decoder: crate::chat::spike_token_bridge::SpikeTokenDecoder,
    ) -> usize {
        let idx = self.trainers.len();
        self.trainers.push(trainer);
        self.encoders.push(encoder);
        self.decoders.push(decoder);
        idx
    }

    /// Gets a mutable reference to the trainer at `idx`.
    pub fn trainer_mut(&mut self, idx: usize) -> Option<&mut crate::semantics::SemanticTrainer> {
        self.trainers.get_mut(idx)
    }

    /// Gets a mutable reference to the encoder at `idx`.
    pub fn encoder_mut(&mut self, idx: usize) -> Option<&mut crate::chat::spike_token_bridge::TokenSpikeEncoder> {
        self.encoders.get_mut(idx)
    }

    /// Gets a mutable reference to the decoder at `idx`.
    pub fn decoder_mut(&mut self, idx: usize) -> Option<&mut crate::chat::spike_token_bridge::SpikeTokenDecoder> {
        self.decoders.get_mut(idx)
    }
}

pub use avx2::{has_avx2, has_fma, batch_euclidean_distances, batch_euclidean_distances_scalar, batch_argmax, dot_product};
#[cfg(feature = "rayon")]
pub use avx2::batch_distances_parallel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_arena_defaults() {
        let arena = StateArena::new(4);
        assert_eq!(arena.membrane, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(arena.recovery, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(arena.threshold, vec![-55.0, -55.0, -55.0, -55.0]);
        assert_eq!(arena.refractory, vec![0, 0, 0, 0]);
    }

    #[test]
    fn state_arena_extend() {
        let mut arena = StateArena::new(2);
        arena.extend(2);
        assert_eq!(arena.membrane.len(), 4);
        assert_eq!(arena.threshold, vec![-55.0, -55.0, -55.0, -55.0]);
    }

    #[test]
    fn weight_matrix_index() {
        let wm = WeightMatrix::new(3, 4);
        assert_eq!(wm.index(1, 2), 6);
        assert_eq!(wm.get(1, 2), 0.0);
    }

    #[test]
    fn weight_matrix_row_access() {
        let mut wm = WeightMatrix::new(2, 3);
        wm.set(0, 1, 0.5);
        assert_eq!(wm.get(0, 1), 0.5);
        assert_eq!(wm.row(0), &[0.0, 0.5, 0.0]);
    }

    #[test]
    fn spike_buffer_push_and_iter() {
        let mut buf = SpikeBuffer::new(2);
        buf.push(0).unwrap();
        buf.push(1).unwrap();
        assert_eq!(buf.iter().cloned().collect::<Vec<_>>(), vec![0, 1]);
        assert!(buf.push(2).is_err());
    }
}
