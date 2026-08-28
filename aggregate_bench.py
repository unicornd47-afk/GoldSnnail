import json, glob, sys, os
sys.path.insert(0, r'C:\Users\Student\Documents\Goldsnnail\Goldsnnail')

all_results = []
for f in sorted(glob.glob('bench_[a-i].json')):
    with open(f) as h:
        data = json.load(h)
        all_results.extend(data['results'])

from bench_online import aggregate
combined = aggregate(all_results)
with open('bench_summary.json', 'w') as out:
    json.dump({"aggregate": combined, "per_game": all_results}, out, indent=2)

print(json.dumps({"aggregate": combined}, indent=2))
print(f"\nPer-game count: {len(all_results)}")
print(f"Unique games: {len(set(r['game_id'] for r in all_results))}")
print(f"Wins: {combined.get('wins', 0)}")
print(f"Games with levels: {sum(1 for r in all_results if r.get('levels_completed', 0) > 0)}")
