# ADR-0002: Data-Oriented Design (DoD) Memory Model

**Status:** Accepted  
**Date:** 2026-08-09  
**Deciders:** Lead Architecture Team

## Context

GoldWorm targets 0.92 MB model size and 72 µs inference latency. Traditional object-oriented memory layouts (`Vec<Neuron>`, `Box<dyn State>`, `HashMap<usize, f32>`) introduce pointer indirection, cache line misses, and heap fragmentation that are incompatible with these constraints.

Neural network simulation is fundamentally a numerical linear algebra problem. The natural representation is a set of parallel flat arrays (Structure of Arrays), not a collection of heap-allocated objects (Array of Structures).

## Decision

We adopt **Data-Oriented Design (DoD)** with **Structure of Arrays (SoA)** as the mandatory memory model for all hot-path simulation state.

Concrete rules:

1. **Neurons are `usize` indices.** A `NeuronIdx` is a newtype wrapper around `usize`. There is no `Neuron` struct in hot paths.
2. **State is parallel flat vectors.** `StateArena` stores four `Vec<f32>`: membrane, recovery, threshold, refractory. Length = number of neurons.
3. **Weights are flat matrices.** `WeightMatrix` stores a single `Vec<f32>` with explicit row-major stride. No `Vec<Vec<f32>>`.
4. **Spikes are flat event buffers.** `SpikeBuffer` stores `Vec<u32>` indices with a fixed capacity.
5. **No nested allocations.** No `Vec<Vec<T>>`, no `Box<T>` per element, no `HashMap` for dense indexed access.
6. **All structs are `#[repr(C)]`.** Public API structs crossing CPU-GPU boundaries must have C-compatible layout.
7. **Maximum alignment: 4 bytes.** No SIMD-required alignments in public structs.

## Consequences

### Positive

- **Cache locality:** Sequential access to flat arrays hits the same cache line. Pointer-chasing causes mandatory cache misses.
- **GPU bridge:** Flat `Vec<f32>` maps directly to `cudaMemcpy` / `vkCmdCopyBuffer`. No pointer fixup, no struct packing, no lifetime extension.
- **SIMD friendliness:** Auto-vectorization and explicit AVX2 intrinsics operate on contiguous memory. AoS layouts break vectorization.
- **Serialization triviality:** Flat memory is hex-editable, network-transferable, and debuggable with standard tooling.

### Negative

- **Less ergonomic API:** Index-based access (`arena.membrane[idx]`) is less readable than field access (`neuron.membrane`).
- **Refactoring cost:** Existing AoS code must be rewritten. This is a breaking change.
- **Debugging complexity:** Without object identity, stack traces and debug prints are less informative.

### Neutral

- The `TelemetryObserver` is explicitly exempt from DoD because it is bounded observation data, not hot-path simulation state. It uses `Vec<AvalancheMetrics>`.

## References

- [Substrate Layer](../architecture/substrate.md)
- [Data-Oriented Arrays Specification](../math/dod_arrays.md)
- [Sandboxed Emergence Manifest](../architecture/manifesto.md)
