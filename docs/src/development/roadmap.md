# GoldSnnail Development Roadmap

> *"We build in layers, from substrate to telemetry. Each phase is independently testable."*

**Project:** GoldSnnail SNN  
**Version:** v0.1.0-phase1  
**Language:** Rust (Edition 2021)

---

## Phase 1: Benchmark Foundation *(Complete)*

**Objective:** Verify the hyperbolic SNN substrate on real-world multi-modal and ARC benchmarks.

### Key Deliverables
- `StateArena`: four parallel `Vec<f32>` (membrane, recovery, threshold, refractory)
- `PoincareDisk`: 4D Poincaré ball with elastic boundary enforcement
- `GridEncoder`: 100D feature vectors → 32D hidden → 16D hyperbolic embeddings
- `ArcDataset`: ARC-AGI-1 loader with variable grid size support
- `TransformationAnalyzer`: k-means clustering and silhouette scoring on ARC tasks
- `HybridSolver`: k-NN router + heuristic solver (reached No-Go gate)
- `ArcIdentityBaseline`: efficiency leaderboard calibration entry

### Verified Metrics

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

### Key Learnings
1. **Hyperbolic space separates task identity, not task mechanism.** Compositional transformations produce monolithic vectors (Silhouette 0.189).
2. **Retrieval fails at 0% exact match.** Task similarity ≠ solution transferability.
3. **Geometry alone does not prevent forgetting.** Replay is necessary (98.7% forgetting without replay).
4. **Feature engineering is mandatory.** Raw pixels (ratio 1.28) → engineered features (ratio 3.66).
5. **Efficiency leaderboard is viable.** 0.92 MB / 72 µs positions GoldSnnail for cost-competitive submissions.

### Success Criteria
- `cargo test` passes for all substrate and vision tests.
- All benchmarks reproduced and documented in `docs/GOLDSNNAIL_REPORT.md`.
- Identity baseline submission generated at `data/arc_submission/submission_identity.json`.

### Dependencies
- None. This is the foundation.

---

## Phase 2: Efficiency Leaderboard *(No-Go Gate Triggered)*

**Objective:** Submit a cost-competitive entry to the ARC-AGI-1 efficiency leaderboard.

### Key Deliverables
- DSL brute-force solver (19 ops, program synthesis up to length 3)
- Cost metrics: $0.00004/task, 72 µs latency, 0.92 MB
- Kaggle submission pipeline (`submission_dsl_v1.json`)

### Status
**No-Go.** Brute-force DSL solver achieved 0.2% eval accuracy (1/400 tasks, Scale2x only). Training accuracy was 4.0% (16/400). New object-based ops (ExtractLargestComponent, RemoveIsolatedPixels, FillEnclosed) contributed 0% additional solves. Program length 3 did not improve eval accuracy. MirrorH/MirrorV bugs fixed but did not affect results.

### Decision Gate
> "If after testing length 3 + new ops accuracy is still <0.5%, pivot to ARC-AGI-2."

**Triggered.** Accuracy 0.2% < 0.5% threshold. Pivoting to ARC-AGI-2.

### Dependencies
- Phase 1 (ArcIdentityBaseline, verified metrics).

---

## Phase 3: Hybrid ARC Solver *(Pending — No-Go Gate Reached)*

**Objective:** Implement a router+DSL architecture for ARC-AGI-1.

### Key Deliverables
- Hyperbolic task-family router (verified ratio 3.66)
- Small DSL (rotate, flip, fill, count)
- Efficiency wrapper (router runs in 72 µs; only promising candidates evaluated)

### Status
**No-Go.** Hybrid solver achieved 0% accuracy on 20 10×10 ARC tasks with Identity/Rotate90/FlipHorizontal heuristics. Router finds neighbors but heuristics do not transfer.

### Path to Re-Entry
- Add 5+ new heuristics (FlipVertical, ColorMapping, ObjectCounting, SymmetryCompletion)
- Re-evaluate on full 400-task evaluation set
- Minimum gate: >5% accuracy with <10% compute overhead vs. identity baseline

### Dependencies
- Phase 1 (GridEncoder, ArcDataset).

---

## Phase 4: ARC-AGI-2 Research *(Active)*

**Objective:** Build ARC-AGI-2 solver targeting the ARC Prize 2025/2026 competition.

### Key Deliverables
- ARC-AGI-2 dataset integration (1000 training, 120 public eval tasks)
- Diagnostic analysis of ARC-AGI-2 task characteristics
- DSL solver baseline on ARC-AGI-2
- Research into compositional reasoning approaches

### Trigger Conditions (Met)
ARC-AGI-2 was officially released with public dataset:
1. ✅ ARC-AGI-2 evaluation dataset release (`github.com/arcprize/ARC-AGI-2`)
2. ✅ Public benchmark with 120 evaluation tasks
3. ✅ Official ARC Prize 2025/2026 announcement ($1M prizes)

### Success Criteria
- Non-zero accuracy on ARC-AGI-2 public eval set
- Cost per task < $0.10
- Full documentation of methodology

### No-Go Gate
If after 4 weeks:
- Accuracy <2% on ARC-AGI-2 public eval, OR
- Cost per task >$0.10

Then pivot to pure efficiency leaderboard and archive ARC-AGI-2 as a research curiosity only.

### Dependencies
- Phase 1 (verified metrics, GridEncoder, ArcDataset capability).
- Phase 2 (DSL solver framework, cost tracking).

---

## Phase Summary

| Phase | Name | Goal | Status |
|-------|------|------|--------|
| 1 | Benchmark Foundation | Verify substrate on real benchmarks | **Complete** |
| 2 | Efficiency Leaderboard | Submit cost-competitive ARC entry | **No-Go** |
| 3 | Hybrid ARC Solver | Router+DSL for ARC-AGI-1 | **No-Go** (re-entry pending) |
| 4 | ARC-AGI-2 Research | Compositional reasoning on ARC-AGI-2 | **Active** |
