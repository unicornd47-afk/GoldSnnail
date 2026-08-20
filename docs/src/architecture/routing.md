# Routing Layer

The routing layer manages expert selection and sparse weight compression for large-scale SNN inference.

## Core Types

### MoAIndex

Mixture of Agents (MoA) expert routing index. Each neuron is assigned to an expert based on routing scores.

```rust
pub struct MoAIndex {
    pub expert_indices: Vec<u32>,
    pub scores: Vec<f32>,
}
```

Usage:
```rust
let mut index = MoAIndex::new(capacity);
index.update(neuron_idx, expert_idx, score);
```

### SHDCCP

Sparse Hybrid Distributed Compressed Column Pointer format for spike packet compression. CSR-like structure for sparse weight matrices.

```rust
pub struct SHDCCP {
    pub values: Vec<f32>,      // Non-zero weight values
    pub col_indices: Vec<u32>, // Column indices for each non-zero
    pub row_ptr: Vec<usize>,   // Row pointers into col_indices/values
}
```

Row iteration:
```rust
for (col, weight) in comp.iter_row(row) {
    // process non-zero entry
}
```

### Multi-Region

Multi-region routing for distributed SNN simulation across CPU/GPU boundaries.

## Design Rationale

Sparse connectivity is central to GoldWorm's efficiency claim. A fully connected `WeightMatrix` of size `N x N` requires `N²` weights. For `N = 10,000`, this is 100M weights (~400 MB for `f32`). Sparse formats reduce this to `O(E)` where `E` is the number of actual synapses.

The `SHDCCP` format mirrors CSR (Compressed Sparse Row) but is optimized for spike-event access patterns:

- `values` stores synaptic weights
- `col_indices` stores post-synaptic neuron indices
- `row_ptr` enables O(1) row start/end lookup

## Current Status

- `MoAIndex::new` and `update` are fully implemented
- `SHDCCP::iter_row` is fully implemented
- `SHDCCP::compress` is a stub (TODO: implement CSR compression from `WeightMatrix`)

## Performance Notes

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `MoAIndex::update` | O(1) | Direct index assignment |
| `SHDCCP::iter_row` | O(nnz_row) | Non-zero entries in row |
| `SHDCCP::compress` | O(N*M) | TODO: should be O(N*M) but with early termination for near-zero weights |

The `WeightMatrix` currently stores dense `rows x cols` data. Future optimization should lazily compress to `SHDCCP` during `step()` when sparsity exceeds a threshold (e.g., 90% zeros).

## Anti-Patterns

```rust
// FORBIDDEN: Dense matrix for sparse connectivity
let weights = vec![vec![f32; cols]; rows]; // O(N*M) memory waste

// FORBIDDEN: HashMap for sparse weights
let weights: HashMap<(usize, usize), f32> = // No locality, allocation overhead

// CORRECT: Flat CSR with explicit stride
pub struct SHDCCP {
    pub values: Vec<f32>,
    pub col_indices: Vec<u32>,
    pub row_ptr: Vec<usize>,
}
```
