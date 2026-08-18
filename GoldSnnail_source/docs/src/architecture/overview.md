# GoldWorm System Architecture

## Overview

GoldWorm is a multi-modal AGI substrate that unifies spiking neural networks (SNN), hyperbolic geometry, compositional reasoning, and desktop interaction in a single Rust codebase. The architecture is organized in five horizontal layers, each with strict data-oriented design (DoD) and zero-allocation hot paths.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tauri Desktop Shell (app/)                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────────┐  │
│  │ SNN Vis     │  │ ARC Debugger│  │ Monster 3D / Audio / Chat │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┬─────────────┘  │
│         │                │                       │                  │
├─────────┼────────────────┼───────────────────────┼──────────────────┤
│         │   IPC Bridge (commands.rs)            │                  │
├─────────┼────────────────┼───────────────────────┼──────────────────┤
│         │                │                       │                  │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────────────▼─────────────┐  │
│  │ ARC Search  │  │ ARC Apply   │  │ Universal DataType Router │  │
│  │ (arc/)      │  │ (arc/)      │  │ (routing/datatype_*)      │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┬─────────────┘  │
│         │                │                       │                  │
├─────────┼────────────────┼───────────────────────┼──────────────────┤
│         │                │                       │                  │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────────────▼─────────────┐  │
│  │ SNN Core    │  │ Geometry    │  │ Semantics / Telemetry     │  │
│  │ (swarm/)    │  │ (geometry/) │  │ (semantics/, telemetry/)  │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────────────────────┘  │
│         │                │                                                │
├─────────┼────────────────┼──────────────────────────────────────────────┤
│         │                │                                                │
│  ┌──────▼──────┐  ┌──────▼─────────────────────────────────────────┐  │
│  │ Substrate   │  │ AVX2 SIMD Intrinsics                            │  │
│  │ (substrate/)│  │ (substrate/avx2.rs)                             │  │
│  └─────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Substrate

**Responsibility:** Flat memory arenas, sparse weight matrices, AVX2 SIMD kernels.

The substrate is the only layer that touches raw memory. Everything above it works with indices and typed wrappers.

### Key Types

| Type | Description |
|------|-------------|
| `StateArena` | Four flat `Vec<f32>`: membrane, recovery, threshold, refractory |
| `WeightMatrix` | Row-major flat weight matrix with CSR sparse variant |
| `SpikeBuffer` | Delta-encoded spike indices for sparse communication |
| `SpikeEvent` | Single spike: src, dst, delay, amplitude, flags |

### Design Rules

- **No allocation in hot paths.** All buffers are pre-allocated at initialization.
- **Structure of Arrays (SoA).** Each field is a separate `Vec<T>`, not `Vec<Struct>`.
- **AVX2 first.** Hot loops use `_mm256_*` intrinsics. Scalar fallbacks exist but are slower.
- **Elastic failure.** Out-of-bounds numerics are clamped, never panicked.

---

## Layer 2: Geometry

**Responsibility:** Poincaré-ball hyperbolic geometry, quaternion rotations, elastic boundary enforcement.

### Key Types

| Type | Description |
|------|-------------|
| `PoincareBall` | Curvature parameter + norm enforcement (`norm() < 1.0`) |
| `HyperbolicPoint` | Coordinates on the Poincaré ball, exponential/Logarithmic maps |
| `Quaternion` | Unit quaternion for 3D rotations, norm-preserving multiplication |

### Design Rules

- **Elastic boundaries.** If a point exceeds the Poincaré disk boundary, it is projected back in. No panic.
- **Deterministic quaternions.** Every multiplication re-normalizes to prevent drift.
- **GPU-ready.** Flat `Vec<f64>` coordinates map directly to CUDA/WGSL kernels.

---

## Layer 3: Swarm

**Responsibility:** QLIF neuron dynamics, spike propagation, SNN state management.

### Key Types

| Type | Description |
|------|-------------|
| `SnnCore` | 180-neuron QLIF swarm: 6 stages × 30 neurons each |
| `QLIFNeuron` | Quad-partite Leaky Integrate-and-Fire with refractory, adaptation, burst |
| `Swarm` | Arena-level spike routing, noise injection, homeostasis |

### Design Rules

- **Fixed topology.** 180 neurons, no dynamic growth. Connectivity is static.
- **Deterministic tick.** Each `step()` advances all neurons synchronously.
- **State serialization.** `StateArena` is the only serialized form — no object pointers.

---

## Layer 4: Routing & Semantics

**Responsibility:** Multi-modal data routing, sparse compression, semantic grounding.

### Key Types

| Type | Description |
|------|-------------|
| `MoaIndex` | Mixture-of-Agents routing: top-k expert selection |
| `SHDCCP` | Sparse CSR matrix format for SHD audio events |
| `DataType` | Universal type tag enum: SpikeStream, ArcGrid, HyperbolicPoint, etc. |
| `ConceptGraph` | Semantic concept nodes embedded in Poincaré space |
| `LexiconToken` | Token with quaternion embedding + hyperbolic coordinates |

### Design Rules

- **Type tags first.** Every payload is prefixed with a `u8` type tag. No dynamic dispatch.
- **Sparse by default.** Dense representations are converted to CSR/SpikeBuffer before storage.
- **Hyperbolic semantics.** Concept distances follow Poincaré geometry, not Euclidean.

---

## Layer 5: ARC Solver

**Responsibility:** Compositional program search for ARC-AGI tasks.

### Key Types

| Type | Description |
|------|-------------|
| `ArcGrid` | 2D grid: `Vec<Vec<u8>>`, max 30×30 |
| `ArcTask` | Train pairs + test inputs/outputs |
| `ArcProgram` | Sequence of `ArcOpToken` operations |
| `ArcOpToken` | 8-byte token: opcode + 7 params |

### Operation Vocabulary

| Op | Code | Params | Description |
|----|------|--------|-------------|
| Identity | 0 | — | No-op |
| Rotate | 1 | angle | 90°/180°/270° |
| Flip | 2 | axis | Horizontal/vertical |
| Move | 3 | dx, dy | Shift content |
| Fill | 4 | color, x, y, w, h | Fill rectangle |
| Copy | 5 | src, dst, w, h | Copy region |
| Gravity | 6 | dir | Drop pixels (4-way) |
| Mirror | 7 | axis_x, axis_y | Mirror around point |
| Tile | 8 | n, m | Repeat grid n×m |
| Crop | 9 | x, y, w, h | Extract subrectangle |
| ReplaceColor | 10 | src, dst | Map color globally |
| Scale | 11 | factor | Nearest-neighbor upscale |
| CropContent | 12 | — | Auto-crop to non-background bbox |

### Search Strategy

1. **Color map first.** Infer global color mappings from training pairs.
2. **Parameter inference.** Extract exact op parameters from input/output dimensions before brute-force enumeration.
3. **Plausibility pre-filter.** Skip ops that cannot possibly match based on grid signature (dims, palette, object count).
4. **Tiered budget.** Depth 1: 60% budget, Depth 2: 30%, Depth 3: 10%.
5. **Fast-fail pruning.** Check only the first training pair before recursing.

### Current Performance

| Metric | Value |
|--------|-------|
| Training set accuracy | 4.0% (16/400) |
| DSL oracle upper bound (max_length=3) | 6.8% (27/400) |
| Avg time per task | 3.6s |
| Depth distribution | 90% depth 1, 10% depth 2-3 |

---

## Tauri Desktop Shell

**Responsibility:** Native desktop UI, IPC bridge to Rust backend.

### Architecture

```
┌─────────────────────────────────────────┐
│  Frontend (HTML/CSS/JS)                 │
│  - SNN Visualizer                       │
│  - ARC Debugger                         │
│  - Monster 3D (Three.js)                │
│  - Audio/Pyramid                        │
│  - Chat/WorldChat                       │
└──────────────────┬──────────────────────┘
                   │ Tauri IPC
┌──────────────────▼──────────────────────┐
│  Backend (Rust)                         │
│  - commands.rs: Tauri command handlers  │
│  - SNN Core init/step                   │
│  - ARC solver invocation                │
│  - DataType encode/decode               │
│  - Monster point generation             │
└─────────────────────────────────────────┘
```

### IPC Commands

| Command | Description |
|---------|-------------|
| `init_snn_core` | Initialize 180-neuron QLIF swarm |
| `step_snn` | Advance SNN by one tick with input spikes |
| `solve_arc_task` | Run compositional solver on ARC grid |
| `get_monster_points` | Generate 3D monster point cloud |
| `list_supported_types` | Enumerate Universal DataType variants |
| `encode_*` / `decode_payload` | Universal DataType serialization |

---

## Data Flow: ARC Task

```
User loads ARC JSON
        │
        ▼
ArcDataset::load_from_directory()
        │
        ▼
search_program(task, config)
        │
        ├──► infer_color_map() ──► color map alone?
        │
        ├──► GridSignature::from_grid() ──► pre-filter plausible ops
        │
        ├──► Depth 1: candidates_for_op() for each op
        │       └──► apply_program() ──► program_solves_train()?
        │
        ├──► Depth 2+: search_depth_first() with fast-fail pruning
        │       └──► partial_solves_train_prefix() ──► recurse or backtrack
        │
        ▼
ArcProgram (sequence of ArcOpToken)
        │
        ▼
apply_program(test_input, program) → predicted output
```

---

## Data Flow: SNN Tick

```
init_snn_core(density)
        │
        ▼
SnnCore::new() → StateArena (180 neurons × 4 fields)
        │
        ▼
step_snn(state, input_spikes)
        │
        ├──► Restore StateArena from DTO
        │
        ├──► For each tick:
        │       ├── Inject input spikes
        │       ├── QLIF membrane update (leak + input + noise)
        │       ├── Refractory check
        │       ├── Spike emission (if V > threshold)
        │       ├── Adaptation + homeostasis
        │       └── Advance refractory counters
        │
        ├── Collect output spikes
        │
        ▼
SnnStateDto → frontend visualization
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Neurons | 180 (6 stages × 30) |
| ARC max grid | 30×30 |
| ARC ops | 13 (0-12) |
| SNN tick latency | <1ms |
| ARC avg solve time | 3.6s |
| Model size | 0.92 MB |
| Inference latency | 72 µs |

---

## Dependencies

- **Rust** 1.70+ (edition 2021)
- **ndarray** — flat array math
- **serde/serde_json** — serialization
- **tauri** — desktop shell
- **three.js** — 3D rendering (frontend)
