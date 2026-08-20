# Data-Oriented Arrays & Memory Model Specification

**Version:** 0.3.0  
**Status:** Accepted  
**Crate:** `goldsnnail`

## 1. Introduction

The GoldSnnail SNN-AGI operates on large-scale spiking neural networks with strict latency and throughput requirements. Traditional object-oriented memory layouts introduce pointer indirection, cache line misses, and heap fragmentation that are incompatible with both CPU-side batch simulation and GPU-side kernel execution.

This specification defines the Data-Oriented Design (DoD) memory model used throughout the `goldsnnail` crate. The model enforces flat, contiguous, index-addressed storage. There are no nested allocations. There are no pointer indirections in hot paths. There is no jagged memory.

## 2. Core Principle: Structure of Arrays (SoA)

All multi-dimensional state is stored as parallel flat vectors. A conceptual "neuron" is not a struct. It is a set of aligned scalar fields accessed by a shared index `usize`.

```rust
pub struct StateArena {
    pub membrane: Vec<f32>,
    pub recovery: Vec<f32>,
    pub threshold: Vec<f32>,
    pub refractory: Vec<u32>,
}
```

This pattern is applied uniformly across:
- **Neuron state** (`StateArena`)
- **Synaptic weights** (`WeightMatrix`)
- **Spike events** (`SpikeBuffer`)
- **Geometric coordinates** (`PoincareDisk`)

## 3. Memory Layout

### 3.1 StateArena

Represents the continuous state of `N` neurons. Four parallel vectors of length `N`.

```
Index:     0        1        2        ...      N-1
membrane: [0.12]   [0.04]   [0.33]   ...    [-0.01]
recovery: [0.01]   [0.02]   [0.00]   ...    [ 0.01]
threshold:[1.00]   [1.00]   [0.95]   ...    [ 1.00]
refractory:[0]     [0]      [0]      ...    [ 0]
```

Access pattern: `arena.membrane[idx]`

### 3.2 WeightMatrix

Represents a fully connected or sparse weight matrix of shape `(pre, post)`. Stored as a row-major flat vector with explicit stride.

```rust
pub struct WeightMatrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl WeightMatrix {
    pub fn index(&self, pre: usize, post: usize) -> usize {
        pre * self.cols + post
    }
}
```

For sparse connectivity, the `WeightMatrix` may be replaced by a compressed format (see `MoAIndex` in the routing layer), but the underlying storage remains a flat `Vec<f32>`.

### 3.3 SpikeBuffer

Records spike events as a flat vector of `u32` neuron indices. No timestamps are stored per-spike in the base format; temporal resolution is implicit in simulation time steps.

```rust
pub struct SpikeBuffer {
    pub indices: Vec<u32>,
    pub count: usize,
}
```

## 4. Why usize Indices?

Pointer-free design is not an aesthetic choice. It is a hardware reality.

1. **Cache locality:** Sequential access to `arena.membrane[idx]` for contiguous `idx` hits the same cache line. Pointer-chasing through `Vec<Neuron>` causes mandatory cache misses.
2. **GPU bridge:** Flat `Vec<f32>` maps directly to device memory via `cudaMemcpy`. No pointer fixup. No struct packing negotiation. No lifetime extension.
3. **SIMD friendliness:** Auto-vectorization and explicit SIMD intrinsics operate on contiguous memory. AoS layouts break vectorization.
4. **Serialization triviality:** Flat memory is inspectable with a hex editor. It is network-transferable with `memcpy`. It is debuggable with standard tooling.

## 5. CUDA Bridge Specification

All core data structures in `goldsnnail` use only heap-allocated flat vectors (`Vec<f32>`, `Vec<u32>`, `Vec<usize>`). These vectors satisfy the following properties required for zero-copy GPU transfer:

1. **Contiguity:** Data is stored in a single heap allocation with no internal padding beyond natural alignment.
2. **Alignment:** All vectors are aligned to 4 bytes (`f32`, `u32`) or 8 bytes (`f64`). No SIMD-required alignments (16/32 bytes) are mandated at the API level, allowing `cudaMemcpy` without pitch calculations.
3. **Pointer stability:** `Vec<T>` guarantees that `data.as_ptr()` remains valid for the lifetime of the vector and is not invalidated by reallocation unless `push`/`resize` is called. The CUDA bridge operates on snapshots taken during `step()` boundaries.
4. **No pointers inside data:** No `Vec` element contains a reference, `Box`, or raw pointer. This eliminates the need for pointer patching during device transfer.

Transfer contract:

```rust
// CPU side
let membrane_gpu = device_malloc(arena.membrane.len() * size_of::<f32>());
cuda_memcpy(
    membrane_gpu,
    arena.membrane.as_ptr(),
    arena.membrane.len() * size_of::<f32>(),
    CudaMemcpyHostToDevice,
);
```

## 6. Formal Constraints

The following rules are mandatory. Violation of these rules constitutes a breaking change to the memory model and requires ADR review.

1. **No `Box<T>` in core structs.** Heap allocation per-element is forbidden in `StateArena`, `WeightMatrix`, `SpikeBuffer`, and all geometric types. Use flat vectors or stack allocation.
2. **No `Rc<T>` or `Arc<T>` in hot paths.** Shared ownership is not required for simulation state. If aliasing is needed, use indices into an arena allocator.
3. **No nested `Vec<Vec<T>>`.** Jagged arrays are forbidden. All 2D+ data must be flattened into a single `Vec<T>` with manual stride arithmetic.
4. **All structs must be `#[repr(C)]` compatible or trivially serializable.** If a struct crosses the CPU-GPU boundary, it must have a defined memory layout compatible with C ABI.
5. **Maximum alignment: 4 bytes.** Types requiring stricter alignment (e.g., `#[repr(align(16))]` for SIMD) are permitted only in local scopes, not in public API structs. This ensures `cudaMemcpy` operates on byte-addressable memory without pitch.

## 7. Anti-Patterns

The following patterns are explicitly forbidden in the `goldsnnail` codebase:

```rust
// FORBIDDEN: Vec of objects
pub struct Network {
    neurons: Vec<Neuron>, // Pointer chasing, cache unfriendly
}

// FORBIDDEN: HashMap for dense indexed access
pub membrane: HashMap<usize, f32>, // No locality, allocation overhead

// FORBIDDEN: Box in hot struct
pub weight: Box<f32>, // Per-element heap allocation

// FORBIDDEN: dyn trait in hot path
pub state: Box<dyn State>, // vtable indirection, allocation

// FORBIDDEN: Jagged arrays
pub adjacency: Vec<Vec<usize>>, // No contiguous memory, pointer chasing
```

## 8. Migration Notes

Existing code using AoS patterns must be refactored to SoA. The migration path is:

1. Identify all `Vec<T>` where `T` is a state-carrying struct.
2. Decompose `T` into its scalar fields.
3. Replace `Vec<T>` with parallel `Vec<f32>` / `Vec<u32>` / `Vec<usize>`.
4. Replace field access (`neuron.membrane`) with indexed access (`arena.membrane[idx]`).
5. Validate with `cargo test` and `cargo bench` to ensure no regression in throughput.

