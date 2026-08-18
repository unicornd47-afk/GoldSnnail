# GoldWorm Documentation

## Phase 1 Report

- **[GOLDWORM_REPORT.md](../GOLDWORM_REPORT.md)** — Complete research report with verified benchmarks, negative results, and Phase 1 findings

## Architecture Philosophy

- [Sandboxed Emergence Manifest](architecture/manifesto.md)

## Core Architecture

- [System Architecture Overview](architecture/overview.md)
- [ARC Solver Architecture](architecture/arc_solver.md)
- [Substrate Layer](architecture/substrate.md) — Flat memory arenas, weight matrices, AVX2 SIMD
- [Geometry Layer](architecture/geometry.md) — Poincaré-ball math, quaternion rotations, elastic boundaries
- [Swarm Layer](architecture/swarm.md) — QLIF dynamics, noise injection, spike propagation
- [Routing Layer](architecture/routing.md) — MoA expert indexing, SHD-CCP sparse compression
- [Telemetry Layer](architecture/telemetry.md) — Passive avalanche/power-law metrics

## Mathematics

- [Data-Oriented Arrays & Memory Model](math/dod_arrays.md)
- [Elastic Poincaré Geometry](math/poincare.md)

## Performance

- [Efficiency Guide](performance/efficiency_guide.md) — Hot-path optimizations, bottlenecks, profiling

## Architecture Decision Records

- [ADR-0001: Sandboxed Emergence](../adrs/0001-sandboxed-emergence.md)
- [ADR-0002: DOD Memory Model](../adrs/0002-dod-memory-model.md)
- [ADR-0003: CUDA-Ready Flat Arrays](../adrs/0003-cuda-ready-arrays.md)

## Development

- [Roadmap](../development/roadmap.md)
- [Report Outline](../development/REPORT_OUTLINE.md)
- [Phase 2 Status](../development/PHASE_2_STATUS.md)
- [ARC-AGI-2 Research](../development/ARC_AGI_2_RESEARCH.md)
- [ARC Failure Analysis](../development/arc_failure_analysis.md) — Benchmark analysis, failure classification, primitive recommendations

## Verified Phase 1 Metrics

| Metric | Value | Status |
|--------|-------|--------|
| N-MNIST 10-Digit (with Replay) | 80.2% | Verified |
| Multi-Modal Semantic Relevance | 83.3% | Verified (post-bugfix) |
| Model Size | 0.92 MB | Verified |
| Inference Latency | 72 µs | Verified |
| Criticality | τ = -1.92 | Verified |
| ARC Task Separation Ratio | 3.66 | Verified |
| ARC Transformation Silhouette | 0.189 | Verified (negative result) |
| ARC Retrieval Exact Match | 0% | Verified (negative result) |
| Forgetting (no replay) | 98.7% | Verified (negative result) |
