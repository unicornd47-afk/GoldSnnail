//! Routing — MoA Expert Indexing & SHD-CCP Compression
//!
//! Responsible for LongLongMoA routing tables and Sparse Hybrid Distributed
//! Compressed Column Pointer (SHD-CCP) spike packet compression.

pub mod moa;
pub mod shd_ccp;
pub mod multi_region;
pub mod datatype_universal;

use crate::substrate::WeightMatrix;

/// Stores expert routing indices and scores for MoA (Mixture of Agents).
#[derive(Debug, Clone)]
pub struct MoAIndex {
    /// Expert index assigned to each neuron.
    pub expert_indices: Vec<u32>,
    /// Routing score per neuron.
    pub scores: Vec<f32>,
}

impl MoAIndex {
    /// Creates a new index with capacity for `capacity` neurons.
    pub fn new(capacity: usize) -> Self {
        Self {
            expert_indices: vec![0; capacity],
            scores: vec![0.0; capacity],
        }
    }

    /// Updates the routing entry for `neuron_idx`.
    pub fn update(&mut self, neuron_idx: usize, expert_idx: u32, score: f32) {
        if neuron_idx < self.expert_indices.len() {
            self.expert_indices[neuron_idx] = expert_idx;
            self.scores[neuron_idx] = score;
        }
    }
}

/// Sparse matrix compression interface (SHD-CCP).
#[derive(Debug, Clone, Default)]
pub struct SHDCCP {
    /// Non-zero weight values.
    pub values: Vec<f32>,
    /// Column indices for each non-zero value.
    pub col_indices: Vec<u32>,
    /// Row pointers into `col_indices` and `values`.
    pub row_ptr: Vec<usize>,
}

impl SHDCCP {
    /// Creates an empty compressor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stub: compresses a dense `WeightMatrix` into CSR-like format.
    pub fn compress(&mut self, _weights: &WeightMatrix) {
        // TODO: implement SHD-CCP compression
    }

    /// Returns an iterator over non-zero entries in `row`.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = (u32, f32)> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        self.col_indices[start..end]
            .iter()
            .copied()
            .zip(self.values[start..end].iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moa_index_update() {
        let mut idx = MoAIndex::new(4);
        idx.update(1, 7, 0.9);
        assert_eq!(idx.expert_indices[1], 7);
        assert_eq!(idx.scores[1], 0.9);
    }

    #[test]
    fn shd_ccp_iter_row() {
        let mut comp = SHDCCP::new();
        comp.row_ptr = vec![0, 2, 3];
        comp.col_indices = vec![1, 3, 0];
        comp.values = vec![0.5, -0.2, 1.0];
        let row0: Vec<_> = comp.iter_row(0).collect();
        assert_eq!(row0, vec![(1, 0.5), (3, -0.2)]);
    }
}

