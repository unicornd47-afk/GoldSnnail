# Geometry Layer

The geometry layer provides elastic mathematical primitives for hyperbolic (Poincaré-ball) and quaternion operations. All operations use asymptotic functions (e.g., `tanh`) softly scaled and pressed against curvature boundaries. No hard panics on domain violations.

## Core Types

### HyperbolicPoint

A point in the Poincaré ball, validated to lie inside the unit ball.

```rust
pub struct HyperbolicPoint {
    pub coords: Vec<f64>,
}
```

Construction validates `||x|| < 1.0` and returns `Err(LabError::InvalidState)` if violated. However, all internal operations use elastic clamping to prevent boundary violations before they reach construction.

### PoincareBall

Poincaré ball model with configurable curvature.

```rust
pub struct PoincareBall {
    pub curvature: f64,
}
```

Key operations:

| Method | Description |
|--------|-------------|
| `exp_map_origin` | Exponential map at origin: lifts tangent vector to disc via `tanh(||v||) * (v / ||v||)` |
| `exp_map` | Exponential map at base point (elastic approximation) |
| `distance` | Distance metric (currently Euclidean proxy — see [Math Specs](math/poincare.md)) |

### PoincareDisk

Simplified 1-D Poincaré disc model for per-neuron scalar operations. Uses `f32` for hot-path performance.

```rust
pub struct PoincareDisk {
    pub curvature: f32, // typically -1.0
}
```

Key operations:

| Method | Description |
|--------|-------------|
| `soft_clamp` | Asymptotic clamp to `(-limit, limit)` via `tanh` |
| `mobius_addition` | 1-D Möbius addition: `(x + y) / (1 + x*y)` with elastic denominator protection |

### Quaternion

Unit quaternion for elastic rotations in the attention mechanism.

```rust
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
```

Operations: `conjugate`, `mul` (Hamilton product), `normalize` (elastic, falls back to identity on degenerate input), `Add`, `Mul<f32>`.

## Elastic Boundary Philosophy

Every numerical boundary in GoldSnnail is soft. Where conventional code would `panic!` or return `Err`, GoldSnnail uses asymptotic clamping:

```rust
// Wrong: abort on boundary violation
fn membrane_step(v: f32) -> f32 {
    if v > THRESHOLD {
        panic!("membrane overflow");
    }
    v
}

// Right: bend, do not break
fn membrane_step(v: f32) -> f32 {
    soft_clamp(v, THRESHOLD)
}
```

The `project_radius` family enforces the manifold boundary:

```rust
pub fn project_radius(r: f32, limit: f32) -> f32 {
    let r = if r.is_finite() { r } else { 0.0 };
    let abs_r = r.abs().min(limit * (1.0 - f32::EPSILON));
    let stretched = abs_r.atanh();
    let compressed = stretched.tanh() * limit;
    compressed.copysign(r)
}
```

This ensures `|r| < SAFE_LIMIT` after every write, preventing `atanh(1.0) = ∞` and similar domain violations.

## Floating-Point Consistency

The codebase currently uses mixed precision:

- `StateArena`, `PoincareDisk`, `Quaternion`: `f32`
- `WeightMatrix`, `HyperbolicPoint`, `WorldModel`, `GridEncoder`: `f64`

This causes implicit casts in cross-layer boundaries. The `StateArena` uses `f32` for performance and cache density, while geometric computations use `f64` for numerical stability in the hyperbolic manifold. Future work should standardize on `f32` with compensated arithmetic for hot paths, reserving `f64` for offline training and verification.

## Known Issues

- `PoincareBall::distance` returns Euclidean distance, not true hyperbolic distance. This is a known inconsistency documented in the codebase.
- `exp_map` at a base point is a linear projection with elastic rescaling, not the true Riemannian exponential map.
- `ndarray::Array1` is used in `HyperbolicPoint` but the rest of the codebase uses flat `Vec<f64>`. This violates the DoD mandate.

See [Elastic Poincaré Geometry](math/poincare.md) for the mathematical specification.
