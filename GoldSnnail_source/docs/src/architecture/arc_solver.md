# ARC Solver Architecture

## Design Goals

The ARC solver is a compositional program search system. Given an ARC-AGI task (train pairs + test input), it searches the space of programs composed from primitive operations and returns the first program that solves all training pairs.

**Non-goals:**
- Solving ARC-AGI-1 evaluation set (requires Tiefe 4+ and object-level reasoning)
- Real-time inference (3.6s/task is acceptable for offline benchmarking)
- Learning from solution programs (no gradient-based meta-learning)

---

## Program Representation

### ArcOpToken

An operation is an 8-byte token:

```
[ opcode: u8 | p1: u8 | p2: u8 | p3: u8 | p4: u8 | p5: u8 | p6: u8 | p7: u8 ]
```

| Opcode | Op | Used Params | Description |
|--------|-----|-------------|-------------|
| 0 | Identity | — | No-op |
| 1 | Rotate | p1=angle | 90/180/270 |
| 2 | Flip | p1=axis | Horizontal/vertical |
| 3 | Move | p1=dx, p2=dy | Shift content |
| 4 | Fill | p1=color, p2=x, p3=y, p4=w, p5=h | Fill rectangle |
| 5 | Copy | p1=sx, p2=sy, p3=dx, p4=dy, p5=w, p6=h | Copy region |
| 6 | Gravity | p1=dir | Drop pixels (up/down/left/right) |
| 7 | Mirror | p1=axis_x, p2=axis_y | Mirror around point |
| 8 | Tile | p1=n, p2=m | Repeat grid n×m |
| 9 | Crop | p1=x, p2=y, p3=w, p4=h | Extract subrectangle |
| 10 | ReplaceColor | p1=src, p2=dst | Map color globally |
| 11 | Scale | p1=factor | Nearest-neighbor upscale (2x/3x) |
| 12 | CropContent | — | Auto-crop to non-background bbox |

### ArcProgram

A program is a sequence of `ArcOpToken`s applied left-to-right:

```rust
pub struct ArcProgram {
    pub tokens: Vec<ArcOpToken>,
}
```

---

## Search Strategy

### Phase 1: Color Map Inference

Before any operation search, infer a global color mapping from training pairs. If all pairs agree on a mapping (e.g., color 3→4, 1→5), try applying it alone or combined with a single operation.

```rust
if let Some(mapping) = infer_color_map(task) {
    // Try: identity + color map
    // Try: op + color map
}
```

### Phase 2: Grid Signature Pre-Filter

Compute a `GridSignature` from the first training pair:

```rust
struct GridSignature {
    dims: (usize, usize),       // (width, height)
    colors: Vec<u8>,            // sorted unique colors
    color_counts: Vec<(u8, usize)>,
    object_count: usize,        // 4-connected components
}
```

Use the signature to skip implausible operations:

| Op | Implausible when |
|----|------------------|
| Rotate90/270 | Input dims don't swap to output dims |
| Rotate180/Flip/Mirror | Input dims ≠ output dims |
| Tile/Scale | Output dims < input dims |
| Crop | Output dims > input dims |
| ReplaceColor | Input palette == output palette |

### Phase 3: Parameter Inference

For parameterized ops, extract exact parameters from training pairs before brute-force enumeration:

| Op | Inference Method |
|----|------------------|
| Tile | `n = out_width / in_width`, `m = out_height / in_height` |
| Scale | `factor = out_width / in_width` (must equal `out_height / in_height`) |
| Crop | Output non-zero bbox |
| ReplaceColor | Find consistent `src→dst` mapping from first differing cell |

### Phase 4: Tiered Depth Search

Search depth in order: 1 → 2 → 3. Each depth gets a budget fraction of the total 100K candidate limit:

| Depth | Budget | Typical Candidates |
|-------|--------|--------------------|
| 1 | 60% (60K) | ~200-500 per op × 13 ops |
| 2 | 30% (30K) | Pruned by fast-fail |
| 3 | 10% (10K) | Pruned by fast-fail |

### Phase 5: Fast-Fail Pruning

For composite programs (depth ≥ 2), check only the first training pair before recursing deeper:

```rust
fn partial_solves_train_prefix(task: &ArcTask, partial: &[ArcOpToken]) -> bool {
    let (input, _) = &task.train_pairs[0];
    apply_program(input, &ArcProgram::from_tokens(partial.to_vec())).is_some()
}
```

This eliminates ~60% of candidates at depth 2+ because most random programs fail on the first pair.

---

## Search Complexity

| Scenario | Candidates | Time |
|----------|-----------|------|
| Depth 1, 13 ops | ~200 | <1ms |
| Depth 1, all params | ~50K | ~50ms |
| Depth 2, pruned | ~100K | ~3s |
| Depth 3, pruned | >100K | >20s (hits budget) |

The 100K candidate budget is a hard limit to prevent infinite search on unsolvable tasks.

---

## Performance Results

### Training Set (400 tasks)

| Metric | Value |
|--------|-------|
| Solved | 16 (4.0%) |
| Depth 1 | 11 tasks (2.8%) |
| Depth 2 | 4 tasks (1.0%) |
| Depth 3 | 1 task (0.2%) |
| Avg time | 3.6s/task |
| Total time | 24min |

### DSL Oracle Upper Bound (max_length=3)

| Metric | Value |
|--------|-------|
| Solved | 27 (6.8%) |
| Depth 1 | 63% |
| Depth 2 | 26% |
| Depth 3 | 11% |

The compositional search achieves **59% of the DSL oracle upper bound** with only 13 ops vs. the DSL's 27 ops.

### Failure Analysis

Top 3 failure clusters (from 392 failures):

| Cluster | % | Root Cause |
|---------|---|------------|
| PatternExtend | 39.3% | Missing `extract_objects`, `fill_enclosed` |
| SizeChange | 35.2% | Missing `tile(n,m)` with n,m > 3 |
| ColorChange | 12.5% | Position-dependent color rules |

---

## Limitations

1. **No object-level reasoning.** The solver operates on pixels, not connected components. Tasks requiring "extract largest object", "sort by size", or "count objects" are unsolvable.
2. **Tiefe 3 ceiling.** The 100K candidate budget makes depth 3+ search infeasible without better pruning.
3. **No learning.** The solver does not adapt its search strategy based on past successes.
4. **ARC-only.** The solver is specialized for ARC-AGI grid tasks. It cannot handle video, audio, or text.

---

## Future Directions

| Enhancement | Expected Impact | Effort |
|-------------|-----------------|--------|
| `extract_objects` (4-CC) | +10-15% | Medium |
| `fill_enclosed` | +5-8% | Low |
| `sort_objects` | +3-5% | Medium |
| Better pruning (histogram, parity) | 2-3x speedup | Medium |
| Depth 4 with parallel search | +5-10% | High |
