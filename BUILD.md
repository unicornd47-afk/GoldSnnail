# Build fixtures & reproducibility

How to reproduce every number this repo publishes.

## Toolchain

Pinned via [`rust-toolchain.toml`](rust-toolchain.toml) — `rustup` installs the exact toolchain (1.96.0) on first build. No toolchain lottery.

## Determinism

- Default features are CPU-only and offline: fixed seeds, no GPU, no network in the hot path. `vulkan` / `cuda` are opt-in features, never on the default path.
- `cargo test` and `cargo bench` run without extra setup beyond the pinned toolchain.

## Benches

Six benches ship as fixtures (`harness = false`, criterion):

```
cargo bench --bench dod_substrate
cargo bench --bench hyperbolic_geometry
cargo bench --bench neuron_dynamics
cargo bench --bench agi_pipeline
cargo bench --bench plasticity
cargo bench --bench benchmark_arc_compositional
```

Same toolchain + same machine → bit-identical numbers across reruns (verified across 3 consecutive reruns; see the evolution log in the README results section).

## Provenance

The README results table separates **verified metrics** from **negative results**, with per-number provenance. Numbers without a fixture in this file are not claimed.
