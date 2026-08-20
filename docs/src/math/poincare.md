# Elastic Poincaré Geometry

This document specifies the mathematical foundations of GoldSnnail's hyperbolic geometry layer. All operations use elastic boundary functions that asymptotically compress values into the valid domain.

## The Poincaré Ball Model

The Poincaré ball model `B^n_c` (curvature `c < 0`) maps hyperbolic space onto the open unit ball:

```
B^n_c = { x ∈ ℝ^n : ||x|| < 1/√|c| }
```

For GoldSnnail, we use the unit disc (`c = -1`) with `f32` scalars for hot-path performance and `f64` for offline computations.

## Elastic Boundary Functions

### SAFE_LIMIT

```rust
pub const SAFE_LIMIT: f32 = 0.9999_f32;
```

All radial coordinates are kept below `SAFE_LIMIT` to prevent `atanh(1.0) = ∞` and similar domain violations.

### elastic_clamp

```rust
pub fn elastic_clamp(x: f32) -> f32 {
    x.tanh()
}
```

Asymptotically squashes any real `x` into `(-1, 1)`. Values well inside the interval pass through nearly linearly; values near or beyond the boundary are smoothly compressed.

### project_radius

```rust
pub fn project_radius(r: f32, limit: f32) -> f32 {
    let r = if r.is_finite() { r } else { 0.0 };
    let abs_r = r.abs().min(limit * (1.0 - f32::EPSILON));
    let stretched = abs_r.atanh();
    let compressed = stretched.tanh() * limit;
    compressed.copysign(r)
}
```

Projects a possibly-out-of-bounds radius back into the safe disc. The `atanh` stretch magnifies small violations, then `tanh` compresses them back within the limit. NaN/Inf are absorbed to `0.0`.

## Hyperbolic Operations

### Möbius Addition (1-D)

```rust
pub fn mobius_add(x: f32, y: f32) -> f32 {
    let denom = 1.0 + x * y;
    let denom_safe = if denom.abs() < 1e-7 { 1e-7_f32.copysign(denom) } else { denom };
    let result = (x + y) / denom_safe;
    project_radius(result, SAFE_LIMIT)
}
```

Formula: `x ⊕ y = (x + y) / (1 + x·y)`

The denominator is protected against near-zero collapse. The result is re-projected to `SAFE_LIMIT` if it wanders outside the disc.

### Hyperbolic Distance (1-D)

```rust
pub fn hyperbolic_distance(x: f32, y: f32) -> f32 {
    let diff = mobius_add(x, -y).abs().min(SAFE_LIMIT);
    2.0 * diff.atanh()
}
```

Formula: `d(x, y) = 2 · atanh(|x ⊕ (-y)|)`

Never panics: the Möbius addition ensures the argument of `atanh` stays within `(-1, 1)`.

### Exponential Map (Origin)

```rust
pub fn exp_map_origin(v: f32) -> f32 {
    if v.abs() < 1e-8 {
        return 0.0;
    }
    let norm = v.abs();
    v.tanh() / norm * norm.tanh()
}
```

Formula: `exp_0(v) = tanh(||v||) · (v / ||v||)`

Elastic: near-zero `v` returns `0.0` (no division hazard).

### Logarithmic Map (Origin)

```rust
pub fn log_map_origin(x: f32) -> f32 {
    let x = project_radius(x, SAFE_LIMIT);
    if x.abs() < 1e-8 {
        return 0.0;
    }
    let norm = x.abs().min(SAFE_LIMIT);
    x.signum() * norm.atanh()
}
```

Formula: `log_0(x) = atanh(||x||) · (x / ||x||)`

Elastic: input is re-projected to `SAFE_LIMIT` before `atanh` to prevent `∞`.

## Quaternion Operations

Unit quaternions represent rotations in 3D space. GoldSnnail uses them for the Twistor attention mechanism.

### Hamilton Product

```rust
pub fn mul(self, other: Self) -> Self {
    Self {
        w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
        y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
        z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * self.w,
    }
}
```

### Elastic Normalization

```rust
pub fn normalize(&self) -> Self {
    let mag_sq = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
    if mag_sq < 1e-12_f32.powi(2) {
        return Self::IDENTITY;
    }
    let inv_mag = mag_sq.sqrt().recip();
    Self { /* scaled components */ }
}
```

If the magnitude is near zero or NaN, returns the identity quaternion. No panics.

## Known Limitations

1. **`PoincareBall::distance` is Euclidean.** The `distance` method in `src/geometry/mod.rs` computes pure Euclidean distance, not true hyperbolic distance. This is a known inconsistency. Use `poincare::hyperbolic_distance` for 1-D hyperbolic distance.
2. **`exp_map` is an elastic approximation.** The multi-dimensional `exp_map` at a base point is a linear projection with elastic rescaling, not the true Riemannian exponential map. This is acceptable for small tangent vectors but accumulates error for large steps.
3. **Mixed `f32`/`f64`.** `PoincareDisk` uses `f32` while `HyperbolicPoint` uses `f64`. Cross-layer boundaries require explicit casts. See [Geometry Layer](geometry.md) for details.

## Invariants

After every geometric operation, the following invariants hold:

1. `|r| < SAFE_LIMIT` for all radial coordinates
2. `quaternion.norm() ≈ 1.0` (elastic normalization)
3. `mobius_add(x, y)` stays in `(-1, 1)` for any `x, y ∈ (-1, 1)`
4. `hyperbolic_distance(x, y) >= 0` for all valid inputs
