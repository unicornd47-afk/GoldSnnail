# ADR-0003: CUDA-Ready Flat Arrays

**Status:** Accepted  
**Date:** 2026-08-09  
**Deciders:** Lead Architecture Team

## Context

GoldSnnail targets GPU acceleration via CUDA and Vulkan backends. GPU kernels require device memory to be allocated and copied from host memory. The standard Rust memory model (heap-allocated structs with pointers, trait objects, and nested vectors) is incompatible with zero-copy GPU transfer.

Specifically:
- `Vec<Neuron>` where `Neuron` contains `Box<dyn State>` cannot be copied to GPU without pointer patching.
- `Vec<Vec<f32>>` (jagged arrays) require pitch calculations for `cudaMemcpy2D`.
- `HashMap<usize, f32>` has no contiguous representation and cannot be transferred as a single buffer.

## Decision

All core data structures use only flat vectors (`Vec<f32>`, `Vec<u32>`, `Vec<usize>`) that satisfy the following properties:

1. **Contiguity:** Data is stored in a single heap allocation with no internal padding beyond natural alignment.
2. **Alignment:** All vectors are aligned to 4 bytes (`f32`, `u32`) or 8 bytes (`f64`). No SIMD-required alignments (16/32 bytes) are mandated at the API level.
3. **Pointer stability:** `Vec<T>` guarantees `data.as_ptr()` remains valid for the lifetime of the vector and is not invalidated by reallocation unless `push`/`resize` is called.
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

## Consequences

### Positive

- **Zero-copy GPU transfer:** Flat arrays copy to device memory with a single `cudaMemcpy` call.
- **Simplified kernels:** GPU kernels operate on raw pointers with known strides. No struct packing, no pointer chasing.
- **Cross-backend compatibility:** The same flat arrays work for Vulkan storage buffers and CUDA global memory.

### Negative

- **Manual stride arithmetic:** Row-column access requires explicit `index = row * cols + col` instead of `matrix[row][col]`.
- **No nested structures:** Complex topologies (e.g., dendritic trees) must be flattened into index arrays.

### Neutral

- The `vulkan` and `cuda` features are optional. The core library compiles without GPU backends and uses CPU-only flat arrays.

## References

- [Substrate Layer](../architecture/substrate.md)
- [Data-Oriented Arrays Specification](../math/dod_arrays.md)
- [Sandboxed Emergence Manifest](../architecture/manifesto.md)
