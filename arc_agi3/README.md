# arc_agi3 — ARC-AGI-3 Interactive Agent

Self-contained, offline-capable ARC-AGI-3 agent that learns game rules through
interaction and plans action sequences to win. Built per the rebuild plan
(`1787783492405-arc-agi-3-rebuild-plan.md`) which replaces the old ARC-AGI-2
strategy-prediction approach (score 0.16) with a true interactive agent.

## Architecture

| Module | Responsibility |
|--------|----------------|
| `types.py` | Toolkit-agnostic data model (`FrameData`, `Action`, `GameAction`, `SceneGraph`, `Transition`, `Hypothesis`, `Plan`). |
| `sim_env.py` | Simulated ARC-AGI-3 games (`sim_nav`, `sim_cm`, `sim_click`) implementing the same protocol as the real toolkit, so the pipeline is testable offline. |
| `env.py` | `Arcade` / `Environment` adapter. Uses the **real `arc-agi` toolkit when installed**, otherwise falls back to simulations. Handles scorecards + JSONL recordings. Implements `competition_score` (levels squared, weighted by level index — see plan "Scoring"). |
| `rust_bridge.py` | Optional bridge to the Rust `goldsnnail` compositional solver for pure grid-transform transitions; pure-Python apply fallback for 13 ops. |
| `perception.py` | Frame → connected-component `SceneGraph`; diff (motion / paint / add / remove); state `signature`. |
| `memory.py` | Append-only, searchable JSONL store of transitions, hypotheses, events; resumes across level resets. |
| `world_model.py` | Learns **parametric** transition rules (object motion + cell paint) from transitions so it can *simulate* unseen states; Rust synthesis fallback. |
| `verifier.py` | Predict → act → compare loop; marks the world model stale and records contradictions on mismatch. |
| `planner.py` | Phase 1 directed exploration (Thompson/epsilon-greedy bandit) + Phase 3 BFS goal search using the world model as simulator; perception-driven targeting for complex (click) actions. |
| `agent.py` | Orchestrator: observe → plan → act → verify → update, with per-level action budget and re-exploration on staleness. |
| `run.py` | CLI harness: `python -m arc_agi_3.run --all`. |

## Run (offline, simulated)

```bash
cd <repo>
pip install -r requirements.txt
python -m arc_agi3.run --all --verbose
```

All three simulated games are winnable; the agent reports `won=True` with
`steps_used <= budget * 2.0`.

## Plug in the real ARC-AGI-3 toolkit

1. `pip install arc-agi==0.9.9` (Python ≥3.12).
2. Provide game definitions in `environment_files/` (local `metadata.json` files)
   or use `OperationMode.ONLINE`/`COMPETITION` with an API key.
3. Call `Arcade(OperationMode.OFFLINE).make("<real_game_id>", seed=0)`.
   `env.py` auto-detects the toolkit and translates `GameAction`/`FrameData`
   to/from the internal types — **no agent code changes required**.

The Rust bridge is optional; set `use_rust=False` (or `GOLDSNNAIL_RUST=0`) to
skip it. Real-time planning currently prefers the fast Python rule learner.

## Kaggle / competition packaging — TODO (user action required)

Per plan Open Question #1, confirm the **submission format** (Docker image vs.
`requirements.txt` + entrypoint script vs. `arc-agi` server mode) before Phase 5.
All code here is offline-capable and dependency-light (`numpy` only; `arc-agi`
optional) to satisfy the "fully local" competition constraint. The agent must
make a single `arc.make()` call per environment and use level-resets only under
`OperationMode.COMPETITION`.
