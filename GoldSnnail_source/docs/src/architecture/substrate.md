# Substrate Layer

The substrate is the foundation of GoldWorm's Data-Oriented Design (DoD) philosophy. It provides flat, contiguous, index-addressed memory structures that map directly to CPU cache lines and GPU device memory without serialization overhead.

## Core Types

### StateArena

Parallel flat arrays representing the continuous state of `N` neurons. A conceptual neuron is not a struct — it is a set of aligned scalar fields accessed by a shared `usize` index.

```rust
pub struct StateArena {
    pub membrane: Vec<f32>,
    pub recovery: Vec<f32>,
    pub threshold: Vec<f32>,
    pub refractory: Vec<u32>,
}
```

Memory layout:

```
Index:     0        1        2        ...      N-1
membrane: [0.12]   [0.04]   [0.33]   ...    [-0.01]
recovery: [0.01]   [0.02]   [0.00]   ...    [ 0.01]
threshold:[1.00]   [1.00]   [0.95]   ...    [ 1.00]
refractory:[0]     [0]      [0]      ...    [ 0]
```

Access pattern: `arena.membrane[idx]`

### WeightMatrix

Row-major flat weight matrix. The `data` vector stores all weights contiguously. Row slices are returned as `&[f32]` for SIMD-friendly hot-path access.

```rust
pub struct WeightMatrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}
```

Index calculation: `index = row * cols + col`

### SpikeBuffer

Fixed-capacity buffer for spike indices emitted in a single timestep. Returns `Err` if full, enforcing backpressure in the simulation loop.

```rust
pub struct SpikeBuffer {
    pub indices: Vec<u32>,
    pub count: usize,
}
```

### NeuronIdx

New-type wrapper for neuron identifiers, providing type safety without runtime overhead.

```rust
pub struct NeuronIdx(pub usize);
```

### ChatArena

Flat storage for chat engine objects. Each object type is stored in its own `Vec`, indexed by `usize`. This follows DoD principles: flat arrays, usize indices, no `Box<dyn Trait>`.

```rust
pub struct ChatArena {
    pub trainers: Vec<crate::semantics::SemanticTrainer>,
    pub encoders: Vec<crate::chat::spike_token_bridge::TokenSpikeEncoder>,
    pub decoders: Vec<crate::chat::spike_token_bridge::SpikeTokenDecoder>,
}
```

## Why usize Indices?

Pointer-free design is not an aesthetic choice. It is a hardware requirement.

1. **Cache locality:** Sequential access to `arena.membrane[idx]` for contiguous `idx` hits the same cache line. Pointer-chasing through `Vec<Neuron>` causes mandatory cache misses.
2. **GPU bridge:** Flat `Vec<f32>` maps directly to device memory via `cudaMemcpy`. No pointer fixup. No struct packing negotiation. No lifetime extension.
3. **SIMD friendliness:** Auto-vectorization and explicit SIMD intrinsics operate on contiguous memory. AoS layouts break vectorization.
4. **Serialization triviality:** Flat memory is inspectable with a hex editor. It is network-transferable with `memcpy`. It is debuggable with standard tooling.

## AVX2 Accelerations

The `avx2` submodule provides runtime-dispatched SIMD optimizations for hot-path operations:

| Function | Description |
|----------|-------------|
| `batch_euclidean_distances` | Euclidean distance from query to many database points |
| `batch_argmax` | Winner-take-all index search |
| `dot_product` | FMA-accelerated dot product |

All functions have scalar fallbacks and dispatch at runtime based on `std::is_x86_feature_detected!`. The `rayon` feature enables parallel distance computation via `batch_distances_parallel`.

## Formal Constraints

The following rules are mandatory for substrate types:

1. **No `Box<T>` in core structs.** Heap allocation per-element is forbidden in `StateArena`, `WeightMatrix`, `SpikeBuffer`, and all geometric types.
2. **No `Rc<T>` or `Arc<T>` in hot paths.** Shared ownership is not required for simulation state. If aliasing is needed, use indices into an arena allocator.
3. **No nested `Vec<Vec<T>>`.** Jagged arrays are forbidden. All 2D+ data must be flattened into a single `Vec<T>` with manual stride arithmetic.
4. **All structs must be `#[repr(C)]` compatible.** If a struct crosses the CPU-GPU boundary, it must have a defined memory layout compatible with C ABI.
5. **Maximum alignment: 4 bytes.** Types requiring stricter alignment are permitted only in local scopes, not in public API structs. This ensures `cudaMemcpy` operates on byte-addressable memory without pitch.

## Anti-Patterns

```rust
// FORBIDDEN: Vec of objects
pub struct Network {
    neurons: Vec<Neuron>, // Pointer chasing, cache unfriendly
}

// FORBIDDEN: HashMap for dense indexed access
pub membrane: HashMap<usize, f32>, // No locality, allocation overhead

// FORBIDDEN: dyn trait in hot path
pub state: Box<dyn State>, // vtable indirection, allocation

// FORBIDDEN: Jagged arrays
pub adjacency: Vec<Vec<usize>>, // No contiguous memory, pointer chasing
```
