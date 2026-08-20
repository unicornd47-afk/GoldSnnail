# goldworm-bench

`goldworm-bench` is a CLI tool for running, tracking, and packaging benchmark results for the GoldWorm spiking neural network (SNN) project. It wraps evaluation examples, records scores with hardware/software context, maintains a leaderboard tracker, and produces submission-ready artifacts.

## Installation

```bash
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- <command>
```

The tool resolves the GoldWorm repo root relative to the runner by default (`../..`). Output artifacts are written to `tools/benchmark_runner/benchmark_artifacts/` by default.

## Commands

### `scan`

Discovers which benchmarks have matching evaluation examples in the GoldWorm repo.

```bash
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- scan
```

Example output:

```
🔍 Scanne GoldWorm-Repo...
5 Kandidaten gefunden:

  • arc-prize — ARC-AGI Efficiency Leaderboard (Prio: 1)
    Metric: accuracy_per_dollar | Prize: $1,000,000+
  • n-mnist — N-MNIST 10-Digit Classification (Prio: 2)
    Metric: accuracy | Prize: Prestige
  ...
```

### `eval <name>`

Runs a single benchmark locally by invoking the corresponding GoldWorm example (`cargo run --release --example <eval_script>`), captures hardware/Rust version, computes the model SHA-256, and writes a timestamped result.

```bash
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- eval n-mnist
```

Output:

```
▶️ Running Benchmark: n-mnist
✅ Ergebnis:
EvalResult {
    benchmark: "n-mnist",
    timestamp: "2026-08-13T07:30:32.123Z",
    score: 0.982,
    model_size_mb: 0.92,
    latency_us: 72.0,
    model_sha256: "a1b2c3d4e5...",
    rust_version: "rustc 1.75.0",
    hardware: "Intel Core i7 / 8 cores",
    log: "Accuracy: 98.2%\n...",
}
```

### `run-all`

Iterates over all registered benchmarks and runs them sequentially. Prints a pass/fail summary.

```bash
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- run-all
```

Output:

```
🚀 Starte alle Benchmarks...

──────────────────────────────────────────────────────────
✓ arc-prize: 0.0
✓ n-mnist: 0.982
✓ shd: 0.0
✓ efficiency-baseline: 0.0
✓ shd-trained: 0.0
```

### `package <name>`

Creates a submission package directory containing a README, the latest `result.json`, and model metadata. The package is placed under `benchmark_artifacts/packages/`.

```bash
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- package n-mnist
```

Output:

```
📦 Paket erstellt: tools/benchmark_runner/benchmark_artifacts/packages/submission_n-mnist_20260813_073032
```

The directory contains:

```
submission_n-mnist_20260813_073032/
├── README.md
├── result.json
└── model_metadata.json
```

### `status`

Displays the current leaderboard tracker state from `benchmark_artifacts/leaderboard_tracker.json`.

```bash
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- status
```

Output:

```
📊 Leaderboard Tracker

Letztes Update: 2026-08-13T07:30:32.123Z

  arc-prize 0.00% (Baseline)
     Runs: 1 | Best: 2026-08-13T07:30:32.123Z
     Prize: $1,000,000+ | Next: Baseline etabliert — Reasoning-Modul benötigt
     Notes: Model: a1b2c3d4e5... | Latency: 72µs
```

### `init`

Initializes or resets the `leaderboard_tracker.json` file.

```bash
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- init
```

Output:

```
📝 Tracker initialisiert.
```

## Benchmark Registry

The registry contains five built-in benchmarks. Each entry specifies the evaluation example, output format, and scoring metric.

| Name | Description | Metric | Prize Pool | Output Format | Needs Model |
|---|---|---|---|---|---|
| `arc-prize` | ARC-AGI Efficiency Leaderboard | `accuracy_per_dollar` | $1,000,000+ | `JsonGrid` | true |
| `n-mnist` | N-MNIST 10-Digit Classification | `accuracy` | Prestige | `CsvLabels` | true |
| `shd` | Spiking Heidelberg Digits (Audio) | `accuracy` | Prestige | `CsvLabels` | true |
| `shd-trained` | SHD with trained Hyperbolic Encoder | `accuracy` | Prestige | `CsvLabels` | true |
| `efficiency-baseline` | GoldWorm internal metrics (Size, Latency) | `size_mb / latency_us` | None | `TextLog` | false |

### Registry behavior

- `discover(repo)` scans `examples/` in the repo. A benchmark is included only if its `eval_script` is empty or a matching `.rs` file exists in that directory.
- `get(name)` returns a single benchmark definition by name.
- `all()` returns all benchmarks sorted by `priority`.

## Artifacts

All output is rooted at `benchmark_artifacts/` (configurable with `-o`).

```
benchmark_artifacts/
├── runs/
│   └── <benchmark>_<timestamp>/
│       └── result.json
├── leaderboard_tracker.json
└── packages/
    └── submission_<benchmark>_<timestamp>/
        ├── README.md
        ├── result.json
        └── model_metadata.json
```

- **`runs/`** — Per-run result JSON files. The directory name encodes the benchmark name and an ISO-ish timestamp.
- **`leaderboard_tracker.json`** — Aggregated best scores, run counts, milestones, and notes. Updated automatically after every successful `eval` or `run-all`.
- **`packages/`** — Submission bundles produced by `package`. Designed for manual upload or further CI processing.

## Example Workflow

```bash
# 1. Initialize the tracker
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- init

# 2. Run all available benchmarks
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- run-all

# 3. Check status and best scores
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- status

# 4. Package the n-mnist submission
cargo run --manifest-path tools/benchmark_runner/Cargo.toml -- package n-mnist

# 5. Inspect the package
ls benchmark_artifacts/packages/submission_n-mnist_*
```

## Extending

To add a new benchmark to the registry, edit `src/registry.rs`:

```rust
defs.insert(
    "my-benchmark".into(),
    BenchmarkDef {
        name: "my-benchmark".into(),
        description: "My custom benchmark description".into(),
        metric: "accuracy".into(),
        prize_pool: "Prestige".into(),
        priority: 2,
        eval_script: "eval_my_benchmark".into(),
        output_format: OutputFormat::CsvLabels,
        needs_model: true,
    },
);
```

Place the corresponding example at `examples/eval_my_benchmark.rs`. The tool expects the example to print a line containing `Accuracy: <value>` or `Score: <value>` so the evaluator can parse the score automatically.

If the benchmark does not require a model or external example (like `efficiency-baseline`), set `eval_script` to `""` and `needs_model` to `false`. The evaluator will skip the run step and return a score of `0.0` with an internal-evaluation log entry.
