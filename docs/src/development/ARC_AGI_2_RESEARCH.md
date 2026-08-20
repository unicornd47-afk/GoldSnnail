# ARC-AGI-2 Research Plan

## Why ARC-AGI-2?

Phase 1 demonstrated that GoldWorm's hyperbolic SNN cannot solve ARC-AGI-1 through pure embedding-based retrieval (0% exact match, Silhouette 0.189). However, ARC-AGI-2 introduces **interactive feedback loops**—the agent may ask questions, request examples, or execute actions.

This interactive paradigm aligns with GoldWorm's architectural strengths:
- **72 µs latency**: Real-time interaction without queueing
- **Online continual learning**: The SNN can update from feedback without full retraining
- **Multi-modal binding**: 83.3% semantic relevance enables cross-modal question answering

## Current Status

**Monitoring only.** No implementation until the ARC-AGI-2 specification is released.

## Trigger Conditions

Begin active research when **any** of the following are announced:
1. ARC-AGI-2 evaluation API or dataset release
2. Public benchmark server with interactive task interface
3. Official ARC Prize announcement with interactive rules

## Research Questions

### R1: Can hyperbolic routing reduce question space?
GoldWorm's task-family router (ratio 3.66) could pre-filter candidate solutions, reducing the number of questions needed to solve a task.

### R2: Does online learning from feedback exploit SNN substrate?
Traditional LLMs require full fine-tuning or in-context learning. GoldWorm's SNN can update synaptic weights from feedback in milliseconds.

### R3: Can multi-modal queries disambiguate tasks?
The 83.3% semantic relevance suggests that natural-language + grid queries could resolve ambiguities that pure grid inputs cannot.

## Experimental Design

### Phase 1: Specification Analysis (Trigger → 2 weeks)
- Parse ARC-AGI-2 task format and interaction grammar
- Identify latency constraints and question budgets
- Map interaction types to GoldWorm capabilities

### Phase 2: Prototype Router (2–4 weeks)
- Implement question-generation heuristics based on task-family clustering
- Simulate interaction on ARC-AGI-1 tasks (interpreting static tasks as single-turn interactions)
- Measure question reduction vs. baseline random guessing

### Phase 3: Integration (4–6 weeks)
- Connect SNN online learning to feedback loop
- Benchmark against static LLM approaches
- Submit to leaderboard if competitive

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Question reduction | >50% vs. random | Router must meaningfully constrain search space |
| Feedback convergence | <10 turns | SNN must learn from feedback faster than LLM context window |
| Cost per task | <$0.01 | Must compete on efficiency leaderboard |

## No-Go Gate

If after 4 weeks of implementation:
- Question reduction <20%, OR
- Feedback convergence >20 turns, OR
- Cost per task >$0.10

Then pivot to **pure efficiency leaderboard** and archive ARC-AGI-2 as a research curiosity only.

## Monitoring Channels

- [ARC Prize Discord / Twitter](https://github.com/fchollet/ARC)
- [François Chollet's blog](https://fchollet.com/)
- [Kaggle ARC-AGI competitions](https://www.kaggle.com/competitions)

## References

- Phase 1 Report: `docs/GOLDWORM_REPORT.md`
- Architecture: `docs/src/architecture/`
- Identity Baseline: `examples/arc_identity_baseline.rs`
