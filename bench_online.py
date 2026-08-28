"""Online benchmark harness for ARC-AGI-3.

Runs Steps 1–4 of the validation plan against the 25 online environments:
  Step 1 — no-LLM baseline
  Step 2 — LLM-enabled run
  Step 3 — hypothesis usage analysis (post-run)
  Step 4 — step-waste profiling (post-run)

Outputs per-game JSONL results and an aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Optional

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc_agi3.agent import ARCAgent
from arc_agi3.env import Arcade, OperationMode
from arc_agi3.memory import MemoryStore
from arc_agi3.perception import Perception


def _safe_summary(mem: MemoryStore) -> dict[str, Any]:
    try:
        return mem.summary()
    except Exception:
        return {"n_transitions": 0, "n_hypotheses": 0, "n_events": 0, "actions_seen": [], "distinct_states": 0}


def _count_events(mem: MemoryStore, event_type: str) -> int:
    return sum(1 for r in mem._records if r.get("type") == "event" and r.get("event") == event_type)


def _inert_actions(mem: MemoryStore, perception: Perception) -> int:
    """Count transitions where grid state didn't change (sig_before == sig_after)."""
    count = 0
    for t in mem._transitions:
        try:
            sb = perception.signature(t.state_before)
            sa = perception.signature(t.state_after)
            if sb == sa:
                count += 1
        except Exception:
            pass
    return count


def _repeated_state_visits(mem: MemoryStore, perception: Perception) -> int:
    """Count transitions that revisit a state signature already seen earlier in memory."""
    seen: set[str] = set()
    repeats = 0
    for t in mem._transitions:
        try:
            sb = perception.signature(t.state_before)
            if sb in seen:
                repeats += 1
            seen.add(sb)
            sa = perception.signature(t.state_after)
            if sa in seen:
                repeats += 1
            seen.add(sa)
        except Exception:
            pass
    return repeats


def run_benchmark(
    arcade: Arcade,
    game_ids: list[str],
    max_steps: int = 300,
    use_llm: bool = False,
    budget_multiplier: float = 2.0,
    seed: int = 0,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run the agent on each game_id and return enriched per-game results."""
    results: list[dict[str, Any]] = []
    for gid in game_ids:
        try:
            agent = ARCAgent(
                arcade,
                gid,
                seed=seed,
                budget_multiplier=budget_multiplier,
                memory_dir="memory",
                use_rust=False,
                verbose=verbose,
                use_llm=use_llm,
            )
            entry = agent.run(max_steps=max_steps)
            mem_summary = _safe_summary(agent.memory)
            perception = agent.perception

            # Post-run metrics
            n_verification_failures = _count_events(agent.memory, "verification_failure")
            n_verification_successes = _count_events(agent.memory, "verification_success")
            n_contradictions = _count_events(agent.memory, "contradiction")
            n_contradictions_pruned = _count_events(agent.memory, "contradiction_pruned")
            n_step_failed = _count_events(agent.memory, "step_failed")
            n_replan = _count_events(agent.memory, "replan_after_failure")
            n_level_progress = _count_events(agent.memory, "level_progress")

            # Visited-set hits from planner
            visited_hits = len(getattr(agent.planner, "_visited", set()))
            # State outcomes tracked
            state_outcomes = len(getattr(agent.planner, "_state_outcomes", {}))

            # LLM-specific metrics
            llm_calls = 0
            llm_failures = 0
            compaction_triggers = 0
            hypotheses_surfaced = 0
            cot_memory_len = 0
            if agent.llm is not None:
                llm_calls = getattr(agent.llm, "_total_steps", 0)
                # Infer failures: steps where LLM was available but chose None
                # We can approximate by checking history for "default"/"planned" vs "llm"
                # But we don't track that directly; use call count minus successful actions.
                # Instead, count events where LLM returned None by inspecting history.
                llm_failures = 0  # approximate
                compaction_triggers = 0  # no direct counter; infer from cot size
                cot_memory_len = len(getattr(agent.llm, "_cot_memory", []))
                # Count hypotheses surfaced to LLM: strong hyps with confidence > 0.3
                try:
                    hyps = agent.world_model.hypotheses()
                    hypotheses_surfaced = sum(1 for h in hyps if h.confidence > 0.3)
                except Exception:
                    hypotheses_surfaced = 0

            inert = _inert_actions(agent.memory, perception)
            repeats = _repeated_state_visits(agent.memory, perception)

            res = {
                "game_id": gid,
                "seed": seed,
                "won": entry.won,
                "levels_completed": len(entry.level_scores),
                "total_score": round(entry.total_score, 4),
                "steps_used": entry.steps_used,
                "budget": entry.budget,
                "n_transitions": mem_summary.get("n_transitions", 0),
                "n_hypotheses": mem_summary.get("n_hypotheses", 0),
                "distinct_states": mem_summary.get("distinct_states", 0),
                # Step 4 waste metrics
                "n_verification_failures": n_verification_failures,
                "n_verification_successes": n_verification_successes,
                "n_contradictions": n_contradictions,
                "n_contradictions_pruned": n_contradictions_pruned,
                "n_step_failed": n_step_failed,
                "n_replan": n_replan,
                "n_level_progress": n_level_progress,
                "inert_actions": inert,
                "repeated_state_visits": repeats,
                # Planner visited-set
                "visited_set_size": visited_hits,
                "state_outcomes_size": state_outcomes,
                # LLM-specific
                "llm_calls": llm_calls,
                "llm_failures": llm_failures,
                "compaction_triggers": compaction_triggers,
                "hypotheses_surfaced": hypotheses_surfaced,
                "cot_memory_len": cot_memory_len,
                "use_llm": use_llm,
            }
            results.append(res)
            print(
                f"[{gid}] won={res['won']} levels={res['levels_completed']} "
                f"score={res['total_score']} steps={res['steps_used']}/{res['budget']} "
                f"trans={res['n_transitions']} hyps={res['n_hypotheses']} "
                f"fail={res['n_verification_failures']} inert={res['inert_actions']} "
                f"repeats={res['repeated_state_visits']} visited={res['visited_set_size']}"
            )
        except Exception as exc:
            print(f"[{gid}] ERROR: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                "game_id": gid,
                "seed": seed,
                "won": False,
                "levels_completed": 0,
                "total_score": 0.0,
                "steps_used": 0,
                "budget": 0,
                "error": str(exc),
                "use_llm": use_llm,
            })
    return results


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(r.get("total_score", 0) for r in results)
    wins = sum(1 for r in results if r.get("won"))
    n = len(results)
    avg_steps = sum(r.get("steps_used", 0) for r in results) / max(n, 1)
    avg_budget = sum(r.get("budget", 0) for r in results) / max(n, 1)
    avg_transitions = sum(r.get("n_transitions", 0) for r in results) / max(n, 1)
    avg_hyps = sum(r.get("n_hypotheses", 0) for r in results) / max(n, 1)
    avg_failures = sum(r.get("n_verification_failures", 0) for r in results) / max(n, 1)
    avg_inert = sum(r.get("inert_actions", 0) for r in results) / max(n, 1)
    avg_repeats = sum(r.get("repeated_state_visits", 0) for r in results) / max(n, 1)
    total_visited = sum(r.get("visited_set_size", 0) for r in results)
    total_contradictions = sum(r.get("n_contradictions", 0) for r in results)
    total_pruned = sum(r.get("n_contradictions_pruned", 0) for r in results)
    total_hypotheses_surfaced = sum(r.get("hypotheses_surfaced", 0) for r in results)
    return {
        "games": n,
        "wins": wins,
        "total_score": round(total, 4),
        "avg_steps": round(avg_steps, 2),
        "avg_budget": round(avg_budget, 2),
        "avg_transitions": round(avg_transitions, 2),
        "avg_hypotheses": round(avg_hyps, 2),
        "avg_verification_failures": round(avg_failures, 2),
        "avg_inert_actions": round(avg_inert, 2),
        "avg_repeated_state_visits": round(avg_repeats, 2),
        "total_visited_set_entries": total_visited,
        "total_contradictions": total_contradictions,
        "total_contradictions_pruned": total_pruned,
        "total_hypotheses_surfaced": total_hypotheses_surfaced,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Online benchmark harness for ARC-AGI-3.")
    parser.add_argument("--max-steps", type=int, default=300, help="Hard step cap per game.")
    parser.add_argument("--budget-multiplier", type=float, default=2.0, help="Budget multiplier.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operation-mode", default="ONLINE", choices=["ONLINE", "OFFLINE", "COMPETITION"])
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Disable LLM (baseline).")
    parser.add_argument("--llm", dest="use_llm", action="store_true", help="Enable LLM.")
    parser.add_argument("--output", default="bench_results.jsonl", help="Output JSONL path.")
    parser.add_argument("--games", nargs="*", default=None, help="Specific game IDs to run (default: all non-simulated).")
    args = parser.parse_args(argv)

    arcade = Arcade(
        operation_mode=OperationMode[args.operation_mode],
        environments_dir=args.environments_dir,
    )

    if args.games:
        game_ids = args.games
    else:
        infos = arcade.get_environments()
        game_ids = [info.game_id for info in infos if not info.is_simulated]
        if not game_ids:
            print("No non-simulated (online) environments found. Check connectivity / environments-dir.")
            return 1

    print(f"Running {len(game_ids)} games in {args.operation_mode} mode (LLM={args.use_llm}, max_steps={args.max_steps})")
    print(f"Games: {', '.join(game_ids[:5])}{'...' if len(game_ids) > 5 else ''}")

    results = run_benchmark(
        arcade,
        game_ids,
        max_steps=args.max_steps,
        use_llm=args.use_llm,
        budget_multiplier=args.budget_multiplier,
        seed=args.seed,
        verbose=args.verbose,
    )

    agg = aggregate(results)
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    # Write JSONL
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"\nResults written to {args.output}")

    # Correlation analysis (Step 3)
    if args.use_llm and results:
        print("\n=== Hypothesis Usage Analysis ===")
        hyps_surfaced = [r.get("hypotheses_surfaced", 0) for r in results]
        levels = [r.get("levels_completed", 0) for r in results]
        if len(hyps_surfaced) > 1:
            # Simple Pearson-ish correlation
            n = len(hyps_surfaced)
            mean_h = sum(hyps_surfaced) / n
            mean_l = sum(levels) / n
            num = sum((h - mean_h) * (l - mean_l) for h, l in zip(hyps_surfaced, levels))
            den_h = sum((h - mean_h) ** 2 for h in hyps_surfaced) ** 0.5
            den_l = sum((l - mean_l) ** 2 for l in levels) ** 0.5
            corr = num / (den_h * den_l) if den_h and den_l else 0.0
            print(f"  Correlation (hypotheses_surfaced vs levels_completed): {corr:.3f}")
        print(f"  Total hypotheses surfaced across games: {sum(hyps_surfaced)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
