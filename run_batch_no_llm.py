"""Batch runner for online benchmark with incremental saves."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_online import run_benchmark, aggregate
from arc_agi3.env import Arcade, OperationMode

BATCH_SIZE = 3
OUTPUT = "bench_no_llm.jsonl"
MAX_STEPS = 300
USE_LLM = False
SEED = 0

arcade = Arcade(operation_mode=OperationMode.ONLINE, environments_dir='environment_files')
all_games = [info.game_id for info in arcade.get_environments() if not info.is_simulated]
print(f"Total games: {len(all_games)}")

# Load existing results
results = []
done_ids = set()
if os.path.exists(OUTPUT):
    with open(OUTPUT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            results.append(r)
            done_ids.add(r["game_id"])
    print(f"Loaded {len(results)} existing results")

remaining = [g for g in all_games if g not in done_ids]
print(f"Remaining: {len(remaining)}")

for i in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[i:i+BATCH_SIZE]
    print(f"\n--- Batch {i//BATCH_SIZE + 1}: {batch} ---")
    try:
        batch_results = run_benchmark(
            arcade, batch,
            max_steps=MAX_STEPS,
            use_llm=USE_LLM,
            budget_multiplier=2.0,
            seed=SEED,
            verbose=False,
        )
        results.extend(batch_results)
        # Append to file
        with open(OUTPUT, "a") as f:
            for r in batch_results:
                f.write(json.dumps(r) + "\n")
        print(f"  Batch done. Total results: {len(results)}")
    except Exception as exc:
        print(f"  BATCH ERROR: {exc}")
        import traceback
        traceback.print_exc()
        time.sleep(2)

# Final aggregate
agg = aggregate(results)
print("\n=== FINAL AGGREGATE (no-LLM baseline) ===")
for k, v in agg.items():
    print(f"  {k}: {v}")

with open("bench_no_llm_summary.json", "w") as f:
    json.dump({"results": results, "aggregate": agg}, f, indent=2)
print("\nSaved to bench_no_llm.jsonl and bench_no_llm_summary.json")
