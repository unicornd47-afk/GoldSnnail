"""Command-line harness to run the ARC-AGI-3 agent on simulated (or real) games.

Examples
--------
Run a single simulated game:
    python -m arc_agi3.run --game sim_nav --seed 1 --verbose

Run all simulated games and print an aggregate scorecard:
    python -m arc_agi3.run --all

When the real ``arc-agi`` toolkit + ``environment_files`` are present, pass a
real game id (e.g. ``ls20``) and the same code path drives it via the toolkit.
"""

from __future__ import annotations

import argparse
import sys

from arc_agi3.agent import ARCAgent
from arc_agi3.env import Arcade, OperationMode


def _run_one(arcade: Arcade, game_id: str, args) -> dict:
    agent = ARCAgent(
        arcade,
        game_id,
        seed=args.seed,
        budget_multiplier=args.budget_multiplier,
        save_recording=args.save_recording,
        memory_dir=args.memory_dir,
        use_rust=not args.no_rust,
        verbose=args.verbose,
        budget_override=args.budget,
        use_llm=not args.no_llm,
    )
    entry = agent.run(max_steps=args.max_steps)
    return {
        "game_id": entry.game_id,
        "seed": entry.seed,
        "won": entry.won,
        "levels_completed": len(entry.level_scores),
        "total_score": round(entry.total_score, 4),
        "steps_used": entry.steps_used,
        "budget": entry.budget,
        "mem": agent.memory.summary(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the ARC-AGI-3 interactive agent.")
    parser.add_argument("--game", default="sim_nav", help="Game id (sim_* or real toolkit id).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Run every simulated game.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--budget-multiplier", type=float, default=2.0)
    parser.add_argument("--budget", type=int, default=None, help="Absolute action budget override.")
    parser.add_argument("--no-rust", action="store_true", help="Disable the Rust solver bridge.")
    parser.add_argument("--save-recording", action="store_true")
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument("--memory-dir", default="memory")
    parser.add_argument("--operation-mode", default="OFFLINE", choices=[m.name for m in OperationMode])
    parser.add_argument("--environments-dir", default="environment_files", help="Directory scanned for local game metadata.json files.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-llm", action="store_true", help="Disable the optional LLM planner (force rule-based).")
    args = parser.parse_args(argv)

    arcade = Arcade(
        operation_mode=OperationMode[args.operation_mode],
        recordings_dir=args.recordings_dir,
        environments_dir=args.environments_dir,
    )

    if args.all:
        games = [info.game_id for info in arcade.get_environments() if info.is_simulated]
    else:
        games = [args.game]

    results = []
    for gid in games:
        try:
            res = _run_one(arcade, gid, args)
            results.append(res)
            print(
                f"[{res['game_id']}] won={res['won']} levels={res['levels_completed']} "
                f"score={res['total_score']} steps={res['steps_used']}/{res['budget']} "
                f"transitions={res['mem']['n_transitions']} hyps={res['mem']['n_hypotheses']}"
            )
        except Exception as exc:  # keep going through the suite
            print(f"[{gid}] ERROR: {exc}")
            import traceback
            traceback.print_exc()

    total = sum(r["total_score"] for r in results)
    wins = sum(1 for r in results if r["won"])
    print(f"\nAggregate: games={len(results)} wins={wins} total_score={round(total, 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
