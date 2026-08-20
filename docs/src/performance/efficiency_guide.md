# Efficiency Guide

This document describes the performance characteristics, optimization strategies, and known bottlenecks in GoldSnnail. It is intended for contributors who want to maintain or improve the 0.92 MB / 72 µs efficiency target.

## Verified Performance Targets

| Metric | Target | Verified |
|--------|--------|----------|
| Model Size | < 1 MB | 0.92 MB |
| Inference Latency | < 100 µs | 72 µs |
| N-MNIST 10-Digit | > 75% | 80.2% |
| Multi-Modal Relevance | > 75% | 83.3% |

## Data-Oriented Design (DoD) Principles

All hot-path code follows DoD/SoA principles:

1. **Flat `Vec<T>` with `usize` indices** — No `Vec<Struct>`, no `Box<dyn Trait>`, no pointer chasing
2. **Pre-allocation** — Arenas and buffers are allocated once and reused
3. **Zero-copy GPU bridge** — Flat memory maps directly to `cudaMemcpy` / `vkCmdCopyBuffer`
4. **SIMD-friendly layouts** — Contiguous `f32` arrays enable auto-vectorization and explicit AVX2

## Hot-Path Optimizations

### AVX2 SIMD

The `substrate::avx2` module provides runtime-dispatched AVX2/FMA implementations:

| Operation | SIMD Width | Speedup |
|-----------|-----------|---------|
| `dot_product` | 8x f32 (FMA) | ~4x scalar |
| `batch_euclidean_distances` | 8x f32 | ~3x scalar |
| `batch_argmax` | 8x f32 | ~2x scalar (limited by horizontal reduction) |

Runtime dispatch uses `std::is_x86_feature_detected!`. On non-x86 targets, scalar fallbacks are used automatically.

### Parallel Batch Operations

With the `rayon` feature, `batch_distances_parallel` distributes distance computation across CPU cores:

```rust
#[cfg(feature = "rayon")]
pub fn batch_distances_parallel(query: &[f32], database: &[f32], dim: usize) -> Vec<f32> {
    database.par_chunks(dim)
        .map(|point| {
            let mut sum_sq = 0.0f32;
            for j in 0..dim {
                let diff = point[j] - query[j];
                sum_sq += diff * diff;
            }
            sum_sq.sqrt()
        })
        .collect()
}
```

## Known Bottlenecks

### 1. `batch_argmax_avx2` — Ineffective Horizontal Reduction

The AVX2 argmax loads 8 values into `_mm256` registers but immediately stores them to a stack array and scans scalar-wise. The SIMD width is wasted; a true horizontal min/max reduction using `_mm256_hmin_ps` (AVX-512) or shuffles would be needed for actual speedup.

**Impact:** Limited — scalar argmax is already fast for typical buffer sizes (< 1000 elements).

### 2. Per-Query Allocations in `QuaternionAttention::forward`

The attention mechanism allocates `scores` and `weights` `Vec`s inside the loop for every query. This is not zero-allocation hot-path code.

**Impact:** Moderate — causes heap pressure and cache misses during inference.

**Fix:** Pre-allocate attention buffers outside the inference loop and pass them as mutable references.

### 3. `GradBuffers` Inside `GridEncoder::train_step`

Although `grad_buffers()` creates reusable buffers, `train_step` calls `self.grad_buffers()` fresh each invocation instead of taking a pre-allocated buffer.

**Impact:** Moderate — heap allocation per training step slows down gradient descent.

**Fix:** Cache `GradBuffers` as a field in `GridEncoder` and reuse across `train_step` calls.

### 4. `WorkingMemory` — Non-DoD Layout

`WorkingMemory` stores `Vec<QLIFNeuron>` (a struct per neuron) instead of flat `StateArena`-style arrays. This defeats the project's own flat-memory principle and causes pointer chasing.

**Impact:** High — cache-unfriendly access pattern for recurrent working memory.

**Fix:** Refactor `WorkingMemory` to use `StateArena` with `NeuronIdx` indices.

### 5. `ChatArena` — Heap-Heavy Semantic Storage

`ChatArena` stores `Vec<SemanticTrainer>` where `SemanticTrainer` contains `HashMap`, `HashSet`, and `Vec<LexiconToken>`. This is heap-heavy and pointer-chasing.

**Impact:** Moderate — affects chat inference latency.

**Fix:** Flatten `LexiconToken` storage into a `Vec<LexiconToken>` with `usize` indices, and replace `HashMap` lookups with flat vectors or perfect hashing.

### 6. `TelemetryObserver::record` — O(n) Removal

`self.history.remove(0)` is O(n) per record because it shifts all remaining elements.

**Impact:** Low for small `max_history`, but scales poorly for long simulations.

**Fix:** Replace `Vec` with `VecDeque` or implement a ring buffer.

### 7. `PoincareBall::distance` — Euclidean Proxy

The `distance` method computes Euclidean distance, not true hyperbolic distance. This is mathematically inconsistent with the project's hyperbolic premise.

**Impact:** Moderate — affects clustering and retrieval quality.

**Fix:** Implement proper hyperbolic distance: `d(x, y) = atanh(||(-x) ⊕ y||) * 2`.

## Floating-Point Precision

Mixed `f32`/`f64` usage causes implicit casts:

| Type | Precision | Usage |
|------|-----------|-------|
| `StateArena`, `PoincareDisk`, `Quaternion` | `f32` | Hot-path simulation |
| `WeightMatrix`, `HyperbolicPoint`, `WorldModel`, `GridEncoder` | `f64` | Offline training, geometric stability |

**Recommendation:** Standardize hot-path geometry on `f32` with compensated arithmetic (Kahan summation) for critical distance computations. Reserve `f64` for training and verification.

## Memory Budget

Target: 0.92 MB total model size.

| Component | Estimated Size | Notes |
|-----------|---------------|-------|
| `StateArena` (N neurons) | 4 * N * 4 bytes | membrane, recovery, threshold, refractory |
| `WeightMatrix` (N x N) | N * N * 4 bytes | Currently dense |
| `SpikeBuffer` | capacity * 4 bytes | Fixed capacity |
| `Lexicon` | ~10-50 KB | Depends on vocabulary size |
| `WorldModel` state | ~1-10 KB | Hidden state vectors |

For `N = 10,000` neurons:
- StateArena: 160 KB
- WeightMatrix (dense): 400 MB — **exceeds target**
- SpikeBuffer: 40 KB

**Recommendation:** Switch to sparse `SHDCCP` format for `WeightMatrix`. With 1% sparsity, weight storage drops to ~4 MB. With 0.1% sparsity, it drops to ~400 KB.

## Benchmarking

Run benchmarks with:

```bash
# All benchmarks
cargo bench

# Specific benchmark
cargo bench -- dod_substrate

# With flamegraph
cargo flamegraph --bench dod_substrate
```

Benchmark results are saved to `target/criterion/`. Compare against baseline with:

```bash
cargo bench -- --baseline main
```

## Profiling Tips

1. **Use `perf` on Linux or `Windows Performance Analyzer` on Windows** to identify cache misses and branch mispredictions.
2. **Check `cargo bloat`** to find unexpectedly large functions.
3. **Use `cargo asm`** to inspect generated SIMD code.
4. **Profile with `heaptrack` or `valgrind --tool=massif`** to identify allocation hotspots.

## Future Optimizations

| Optimization | Expected Gain | Priority |
|-------------|---------------|----------|
| Sparse `WeightMatrix` (SHDCCP) | 10-100x memory reduction | High |
| Pre-allocated attention buffers | Eliminate per-query allocations | Medium |
| Ring buffer for `TelemetryObserver` | O(1) insertion/removal | Low |
| `f32` standardization for geometry | Reduced memory bandwidth | Medium |
| CUDA kernel implementation | GPU acceleration for large networks | High |
| Horizontal argmax reduction (AVX-512) | 2x argmax speedup | Low |
