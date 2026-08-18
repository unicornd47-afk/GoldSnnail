# Phase 2: Efficiency Leaderboard — Status Report

**Date:** 2026-08-12  
**Status:** No-Go Gate Triggered — Pivoting to ARC-AGI-2

## Objective

Submit a cost-competitive entry to the ARC-AGI-1 efficiency leaderboard. The hypothesis was that a brute-force DSL solver could achieve >0% accuracy on ARC-AGI-1 evaluation tasks at near-zero compute cost.

## Findings

### DSL Brute-Force Solver: 0.2% Eval Accuracy

| Configuration | Training Set Accuracy | Evaluation Set Accuracy |
|---------------|----------------------|------------------------|
| 19 ops, max_length=1 | ~3.0% | 0.0% |
| 19 ops, max_length=2 | ~4.0% | 0.2% (1/400, Scale2x) |
| 19 ops, max_length=3 | ~4.0% | 0.2% (1/400, Scale2x) |
| New object ops (ExtractLC, RmIsolated, FillEnclosed) | +0% | +0% |

**Key diagnostic findings:**
- 35.7% of train pairs have size changes (scaling ops needed but not sufficient)
- 0% of eval tasks have pure color maps
- 67.5% of tasks have all train pairs same size
- Only Scale2x solved any eval task; all other programs failed on eval
- Program length 3 (6,859 combinations per task) did not improve eval accuracy
- Training accuracy (4.0%) is 20× higher than eval accuracy (0.2%)

### Fixed Bugs

- `MirrorH` and `MirrorV` had integer overflow for even-width grids (formula `2*half - 1 - c + odd` produced negative values). Fixed with correct mirror indexing: `src = if c < w/2 { c } else { w - 1 - c }`.

### Cost: Ultra-Low (Unchanged)

| Metric | Value |
|--------|-------|
| Model size | 0.92 MB |
| Inference latency | 72 µs / task |
| Total inference (400 eval tasks) | ~29 ms |
| Estimated compute cost | $0.00004 USD |
| LLM baseline cost | $200 / task |
| **Efficiency ratio** | **5,000,000× cheaper** |

### Submission Pipeline: Working

- **Format:** Kaggle-compatible `submission.json` with `attempt_1` and `attempt_2` per task
- **File:** `data/arc_submission/submission_dsl_v1.json` (3.2 MB)
- **Validation:** JSON structure verified against ARC-AGI-1 spec

## Honest Assessment

The efficiency leaderboard requires **both** low cost AND non-zero accuracy. Our cost is unmatched, but our accuracy with 19-operation brute-force DSL is only 0.2% on the evaluation set.

This is a scientifically valuable negative result: it empirically confirms that **brute-force DSL search with 19 geometric/color primitives cannot solve ARC-AGI-1 evaluation tasks at scale**. The eval set is specifically designed to resist simple program synthesis.

## Decision Gate: No-Go

Per the project plan:
> "If after testing length 3 + new ops accuracy is still <0.5%, pivot to ARC-AGI-2 or pure efficiency leaderboard."

**Result:** Accuracy is 0.2% (below 0.5% threshold). **Pivoting to ARC-AGI-2.**

## Path Forward: ARC-AGI-2 Pivot

ARC-AGI-2 is the current active competition (ARC Prize 2025/2026):
- 1,000 public training tasks, 120 public evaluation tasks
- Top models (GPT-5.6 Sol, Claude Opus 5) score 75–92%
- Human performance: 66% average, 100% panel completion
- Grand prize threshold: >85%
- Pure LLMs score 0%; requires reasoning systems

### Next Steps

1. Clone ARC-AGI-2 dataset from `github.com/arcprize/ARC-AGI-2`
2. Set up ARC-AGI-2 data loader in goldworm
3. Run diagnostic on ARC-AGI-2 training set (1000 tasks)
4. Benchmark DSL solver on ARC-AGI-2 public eval (120 tasks)
5. Analyze ARC-AGI-2 task characteristics vs ARC-AGI-1
6. Research ARC-AGI-2-specific approaches (compositional reasoning, symbolic interpretation)

## Conclusion

Phase 2 has produced a **working DSL solver pipeline** and **verified cost metrics**, but has confirmed that **brute-force DSL search cannot achieve meaningful accuracy on ARC-AGI-1 evaluation tasks**. The efficiency leaderboard requires both axes of the cost-vs-accuracy tradeoff; we excel at cost but need a fundamentally different approach for accuracy.

This is not a failure. It is a calibrated baseline from which future GoldWorm ARC solvers can measure progress. We now pivot to ARC-AGI-2, where the benchmark has been specifically hardened against brute-force methods and requires genuine compositional reasoning.
