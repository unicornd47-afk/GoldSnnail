# GoldWorm

**A multi-modal AGI substrate: spiking neural networks, hyperbolic geometry, compositional reasoning, and desktop interaction — unified in Rust.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)](https://www.rust-lang.org/)
[![Tauri](https://img.shields.io/badge/tauri-v2-blue)](https://tauri.app/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is GoldWorm?

GoldWorm is a research codebase exploring the intersection of:

- **Spiking Neural Networks (SNN)** — QLIF dynamics with 180 neurons, 6 stages, deterministic tick evolution
- **Hyperbolic Geometry** — Poincaré-ball embeddings, quaternion rotations, elastic boundary enforcement
- **Compositional Reasoning** — ARC-AGI solver with 13 grid primitives and depth-first search
- **Desktop Interaction** — Tauri v2 shell with real-time SNN visualization, ARC debugger, and 3D rendering

The architecture follows **Data-Oriented Design (DoD)** principles: flat memory, Structure of Arrays, zero-allocation hot paths, and AVX2 SIMD acceleration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Tauri Desktop Shell                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ SNN Vis  │ │ARC Debug │ │ Monster  │ │ Audio / Chat   │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘ │
└───────┼───────────┼───────────┼───────────────┼───────────┘
        │           │           │               │
        └───────────┴───────────┴───────────────┘
                    Tauri IPC
┌─────────────────────────────────────────────────────────────┐
│  Rust Backend (goldworm crate)                               │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ ARC Solver  │  │ SNN Core    │  │ Universal DataType   │  │
│  │ (13 ops)    │  │ (180 neurons)│  │ Router (15 types)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Geometry    │  │ Semantics   │  │ Telemetry            │  │
│  │ (Poincaré)  │  │ (Concepts)  │  │ (Avalanche metrics)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Substrate (AVX2 SIMD, flat memory arenas)               │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### SNN Core
- **180 neurons** in 6 stages × 30 neurons each
- **QLIF dynamics:** leak, integrate, fire, refractory, adaptation, burst
- **Deterministic tick** evolution with optional noise injection
- **Real-time visualization** in Tauri frontend (raster, membrane curves, firing rates)

### ARC Solver
- **13 grid primitives:** Identity, Rotate, Flip, Move, Fill, Copy, Gravity, Mirror, Tile, Crop, ReplaceColor, Scale, CropContent
- **Compositional search:** Depth-first with parameter inference and fast-fail pruning
- **Color map composition:** Global color mappings inferred from training pairs
- **Benchmark:** 4.0% on ARC-AGI-1 training set (16/400 tasks)

### Geometry
- **Poincaré ball:** Curvature-parameterized hyperbolic space with elastic boundary enforcement
- **Quaternions:** Unit quaternion rotations with automatic re-normalization
- **GPU-ready:** Flat `Vec<f64>` coordinates map directly to CUDA/WGSL kernels

### Routing & Semantics
- **Universal DataType:** 15 type tags with hex/base64 serialization (SpikeStream, ArcGrid, HyperbolicPoint, etc.)
- **MoA routing:** Mixture-of-Agents expert indexing
- **SHD-CCP:** Sparse CSR compression for audio events
- **Concept graphs:** Semantic concepts embedded in Poincaré space

---

## Getting Started

### Prerequisites

- Rust 1.70+ (edition 2021)
- Node.js 18+ (for Tauri frontend)
- Git

### Build

```bash
# Clone
git clone https://github.com/your-org/goldworm.git
cd goldworm

# Build library + tests
cargo build --lib
cargo test --lib

# Run ARC benchmark
cargo run --example arc_compositional_solver -- --benchmark data/arc-agi-repo/data/training 3

# Run Tauri desktop app
cd app
npm install
npm run tauri dev
```

### Run Examples

```bash
# SNN core visualization
cargo run --example snn_core_demo

# ARC DSL oracle analysis
cargo run --example arc_dsl_oracle_analysis

# Audio encoding (SHD)
cargo run --example eval_shd
```

---

## Verified Phase 1 Metrics

| Metric | Value | Status |
|--------|-------|--------|
| N-MNIST 10-Digit (with Replay) | 80.2% | ✅ Verified |
| Multi-Modal Semantic Relevance | 83.3% | ✅ Verified |
| Model Size | 0.92 MB | ✅ Verified |
| Inference Latency | 72 µs | ✅ Verified |
| Criticality | τ = -1.92 | ✅ Verified |
| ARC Task Separation Ratio | 3.66 | ✅ Verified |
| ARC Transformation Silhouette | 0.189 | ⚠️ Negative result |
| ARC Retrieval Exact Match | 0% | ⚠️ Negative result |
| Forgetting (no replay) | 98.7% | ⚠️ Negative result |

---

## Project Structure

```
goldworm/
├── src/                    # Library source
│   ├── arc_*.rs           # ARC solver (program, apply, search, parser)
│   ├── swarm/             # SNN core (QLIF, noise, homeostasis)
│   ├── geometry/          # Poincaré ball, quaternions
│   ├── routing/           # MoA, SHD-CCP, universal datatype
│   ├── semantics/         # Concept graphs, token engine
│   ├── telemetry/         # Avalanche metrics, entropy
│   ├── substrate/         # Flat memory, AVX2 SIMD
│   └── vision/            # ARC loader, DSL solver, encoders
├── app/                    # Tauri desktop shell
│   ├── src/               # Frontend (HTML/CSS/JS + Three.js)
│   └── src-tauri/         # Rust backend (commands, SNN, ARC)
├── examples/               # ~55 runnable examples
├── benches/                # Criterion benchmarks
├── tests/                  # Integration tests
├── data/                   # Datasets (SHD, N-MNIST, ARC)
├── docs/                   # Architecture docs, ADRs, figures
└── models/                 # Pre-trained model artifacts
```

---

## Design Principles

1. **Elastic Boundaries over Hard Fails.** Every numerical boundary is soft. Out-of-bounds values are clamped, never panicked.
2. **Flat Memory, Zero Indirection.** Every neuron is an index. Every state is a vector. No `Box<dyn>`, no `HashMap` in hot paths.
3. **Sparse by Default.** Dense representations are converted to CSR/SpikeBuffer before storage.
4. **Type Tags First.** Every payload is prefixed with a `u8` type tag. No dynamic dispatch.
5. **Deterministic Tick.** All state advances are synchronous and reproducible.

---

## License

MIT — see [LICENSE](LICENSE) for details.
