# Telemetry Layer

The telemetry layer is a **passive observer** that records avalanche metrics and criticality estimates without mutating simulation state or blocking computation.

## Core Types

### AvalancheMetrics

Metrics collected for a single avalanche window:

```rust
pub struct AvalancheMetrics {
    pub total_spikes: u64,      // Total spikes in window
    pub mean_activity: f32,     // Mean activity across neurons
    pub criticality_index: f32, // Branching ratio proxy
    pub entropy: f32,           // Shannon entropy of spike distribution
}
```

### TelemetryObserver

Passive observer that records avalanche metrics over time:

```rust
pub struct TelemetryObserver {
    pub history: Vec<AvalancheMetrics>,
    pub max_history: usize,
}
```

Key methods:
- `new(max_history)` — creates observer with bounded history
- `record(spike_count)` — records metrics for a window
- `snapshot()` — returns latest metrics

### PowerLawObserver

Fits power-law distributions to avalanche size data. Used to estimate criticality exponent τ.

```rust
pub struct PowerLawObserver { /* ... */ }
pub struct PowerLawFit { /* ... */ }
```

## Passive Observation Pattern

Telemetry is strictly passive. It observes spike counts and records metrics without mutating simulation state:

```rust
// Wrong: active gatekeeping
fn step(swarm: &mut Swarm) -> Result<(), StepError> {
    validate_ranges(&swarm)?;
    // ...
}

// Right: passive observation
fn step(swarm: &mut Swarm, observer: &mut TelemetryObserver) {
    // ... compute step ...
    observer.record_membrane_distribution(&swarm.state.membrane);
}
```

If the system diverges, we see it in the dashboard, not in a panic log.

## Bounded History

The observer maintains a bounded ring buffer of metrics. When `history.len() >= max_history`, the oldest entry is removed before pushing the new one. This prevents unbounded memory growth during long simulations.

**Performance note:** `Vec::remove(0)` is O(n). For high-frequency recording, consider replacing with `VecDeque` or a ring buffer for O(1) insertion/removal.

## Criticality Measurement

GoldWorm measures criticality via avalanche distributions. A branching ratio τ ≈ -1.92 indicates the network operates at the edge of chaos — the regime where information processing capacity is maximized.

The `PowerLawObserver` fits a power law `P(s) ∝ s^(-τ)` to avalanche sizes. Values of τ near -1.5 to -2.0 are associated with critical dynamics in cortical tissue.

## Integration Points

| Consumer | How it uses Telemetry |
|----------|----------------------|
| `Swarm` | Calls `record()` after each timestep |
| `Chat` | Uses avalanche metrics for response selection via `AvalancheGuidedSelector` |
| `Examples` | `simulate_avalanche` generates synthetic distributions for testing |

## Anti-Patterns

```rust
// FORBIDDEN: Active validation blocking computation
fn step() -> Result<(), ValidationError> { /* ... */ }

// FORBIDDEN: Unbounded telemetry history
pub history: Vec<AvalancheMetrics>, // No max_history cap

// CORRECT: Passive, bounded observation
pub struct TelemetryObserver {
    pub history: Vec<AvalancheMetrics>,
    pub max_history: usize,
}
```
