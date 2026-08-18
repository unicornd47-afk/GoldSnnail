# Swarm Layer

The swarm layer owns temporal evolution. It reads from `StateArena`, applies QLIF dynamics with noise, and writes spikes into `SpikeBuffer`.

## Core Types

### SwarmConfig

Configuration for QLIF swarm dynamics:

```rust
pub struct SwarmConfig {
    pub decay: f32,              // Membrane potential decay per timestep
    pub resting_potential: f32,  // Resting membrane potential (mV)
    pub noise_std: f32,          // Gaussian noise standard deviation
}
```

Default values: `decay = 0.95`, `resting_potential = -70.0`, `noise_std = 0.1`

### Swarm

QLIF swarm operating over flat state arenas:

```rust
pub struct Swarm {
    pub arena: StateArena,
    pub weights: WeightMatrix,
    pub spike_buffer: SpikeBuffer,
    pub config: SwarmConfig,
}
```

Construction pre-allocates `capacity` neurons, a `capacity x capacity` weight matrix, and a spike buffer of the same capacity.

### QLIFNeuron

Single QLIF neuron with persistent state. Intentionally small and stack-friendly. For population-level simulation, use `Swarm` + `StateArena`.

```rust
pub struct QLIFNeuron {
    pub v_m: f32,      // Membrane potential
    pub phase: f32,    // Phase tracker
    pub adapt: f32,    // Adaptation variable
    pub refract: u16,  // Refractory countdown
    pub quat: [f32; 4], // Quaternion state for attention coupling
}
```

## QLIF Dynamics

The `step` method advances a single neuron by one timestep:

1. If refractory > 0, decrement and return `None`
2. Integrate input current: `v_m += (current - v_m - adapt) * 0.1`
3. Elastic clamp: `v_m = v_m.clamp(-1.0, 1.0)`
4. If `v_m >= 1.0`: fire spike, reset `v_m = 0.0`, increment adaptation, set refractory = 5
5. Return `Some(())` on spike, `None` otherwise

### Noise Injection

Noise is not a hack to escape local minima. It is a structural element of QLIF dynamics. Controlled Gaussian noise (`noise_std`) is injected every timestep to maintain edge-of-chaos dynamics. Without noise, the system settles into rigid equilibrium.

```rust
let noise = rand_normal(0.0, noise_std);
membrane += input + noise - decay * recovery;
```

## Spike Propagation

Spikes are recorded in `SpikeBuffer` during the `step` call. The buffer is cleared at the start of each timestep. The `Swarm::step` method currently accepts `&[u32]` input spikes but the implementation is a stub pending full QLIF integration.

## Current Status

- `Swarm::step` is a stub (TODO: implement full QLIF step with noise injection)
- `QLIFNeuron::step` is fully implemented and tested
- `SwarmConfig` defaults are tuned for edge-of-chaos criticality (τ ≈ -1.92)

## Integration Points

| Consumer | How it uses Swarm |
|----------|-------------------|
| `WorldModel` | Reads `arena` state for predictive coding |
| `Chat` | Uses `QLIFNeuron` for spike token encoding |
| `Telemetry` | Observes `spike_buffer` counts for avalanche metrics |
| `Vision` | Feeds `ArcGrid` spikes into swarm for ARC evaluation |
