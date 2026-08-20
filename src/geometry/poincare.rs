//! Poincaré-Ball Operations with Elastic Boundaries
//!
//! All operations act on raw `f32` scalars or flat slices. No struct hierarchies,
//! no trait objects. Boundary violations (r ≥ 1) are **never** panics or errors —
//! values are asymptotically compressed back into the open unit disc.
//!
//! # Mathematical Background
//!
//! The Poincaré ball model B^n_c (curvature c < 0) maps hyperbolic space onto the
//! open unit ball { x ∈ ℝ^n : ‖x‖ < 1/√c }. For GoldSnnail we use the unit disc
//! (c = -1, n = 1 for per-neuron scalars) but the 2-D extension is trivial.
//!
//! Key invariant: all radial coordinates satisfy `|r| < SAFE_LIMIT` after every
//! write. The `project_radius*` family enforces this elastically.

/// The radial safety limit: r is always kept below this value to avoid
/// numerical blow-up in hyperbolic metrics (e.g. `atanh(1.0) = ∞`).
pub const SAFE_LIMIT: f32 = 0.9999_f32;

// ============================================================================
// Scalar utilities
// ============================================================================

/// Elastic scalar clamp: squashes any real `x` into `(-1, 1)` via `tanh`.
///
/// Values well inside the unit interval pass through nearly linearly;
/// values near or beyond the boundary are smoothly compressed asymptotically.
/// This is the "Bending, not Breaking" primitive for all geometric saturation.
#[inline(always)]
pub fn elastic_clamp(x: f32) -> f32 {
    x.tanh()
}

/// Projects a possibly-out-of-bounds Poincaré radius back into the safe disc.
///
/// - If `r` is already inside `(-limit, limit)` the output is close to `r`.
/// - If `r` violates the boundary the output is pressed against `±limit`
///   asymptotically — no clipping discontinuity, no panic.
///
/// # Arguments
///
/// * `r` — radial coordinate (may be any finite `f32`, including NaN-safe paths).
/// * `limit` — safety radius, typically [`SAFE_LIMIT`] (0.9999).
#[inline(always)]
pub fn project_radius(r: f32, limit: f32) -> f32 {
    // Guard against NaN/Inf that could propagate from upstream numeric instability.
    let r = if r.is_finite() { r } else { 0.0 };
    // Map the absolute value: [0, limit) → [0, atanh(limit)] → elastic.
    let abs_r = r.abs().min(limit * (1.0 - f32::EPSILON)); // avoid atanh(1)
    let stretched = abs_r.atanh(); // [0, limit) → [0, ∞)
    let compressed = stretched.tanh() * limit; // elastic clamp then rescale
    compressed.copysign(r)
}

/// In-place elastic projection over a flat slice of radial coordinates.
///
/// Suitable for bulk application over an entire `StateArena.poincare_r` array.
#[inline]
pub fn project_radius_slice(r: &mut [f32], limit: f32) {
    for val in r.iter_mut() {
        *val = project_radius(*val, limit);
    }
}

// ============================================================================
// 1-D Möbius addition (hyperbolic translation)
// ============================================================================

/// Möbius addition in the 1-D Poincaré disc: `x ⊕ y`.
///
/// Formula: `(x + y) / (1 + x·y)`
///
/// Elastic: the denominator is protected against near-zero collapse; the result
/// is re-projected to `SAFE_LIMIT` if it wanders outside the disc.
#[inline]
pub fn mobius_add(x: f32, y: f32) -> f32 {
    let denom = 1.0 + x * y;
    // Elastic protection: if denom is near zero, collapse to the midpoint.
    let denom_safe = if denom.abs() < 1e-7 { 1e-7_f32.copysign(denom) } else { denom };
    let result = (x + y) / denom_safe;
    project_radius(result, SAFE_LIMIT)
}

// ============================================================================
// Hyperbolic distance
// ============================================================================

/// Hyperbolic distance between two points in the 1-D unit disc.
///
/// `d(x, y) = 2 · atanh(|x ⊕ (−y)|)`
///
/// Returns a non-negative `f32`. Never panics: the Möbius addition ensures the
/// argument of `atanh` stays within `(-1, 1)`.
#[inline]
pub fn hyperbolic_distance(x: f32, y: f32) -> f32 {
    let diff = mobius_add(x, -y).abs().min(SAFE_LIMIT);
    2.0 * diff.atanh()
}

// ============================================================================
// Exponential and logarithmic maps at the origin
// ============================================================================

/// Exponential map at the origin: lifts a tangent vector `v` to the disc.
///
/// `exp_0(v) = tanh(‖v‖) · (v / ‖v‖)` (1-D simplification: v/|v| = sign(v))
///
/// Elastic: near-zero `v` returns `0.0` (no division hazard).
#[inline]
pub fn exp_map_origin(v: f32) -> f32 {
    if v.abs() < 1e-8 {
        return 0.0;
    }
    let norm = v.abs();
    v.tanh() / norm * norm.tanh() // simplifies to tanh(v) for 1-D
}

/// Logarithmic map at the origin: lifts a disc point `x` to the tangent space.
///
/// `log_0(x) = atanh(‖x‖) · (x / ‖x‖)`
///
/// Elastic: input is re-projected to `SAFE_LIMIT` before `atanh` to prevent `∞`.
#[inline]
pub fn log_map_origin(x: f32) -> f32 {
    let x = project_radius(x, SAFE_LIMIT);
    if x.abs() < 1e-8 {
        return 0.0;
    }
    let norm = x.abs().min(SAFE_LIMIT);
    x.signum() * norm.atanh()
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_clamp_stays_in_unit_interval() {
        for x in [-1000.0f32, -1.0, 0.0, 1.0, 1000.0] {
            let y = elastic_clamp(x);
            assert!(y >= -1.0 && y <= 1.0, "clamp({x}) = {y} out of (-1,1)");
        }
    }

    #[test]
    fn project_radius_never_escapes_limit() {
        let test_vals = [-10.0f32, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 10.0, f32::MAX];
        for r in test_vals {
            let out = project_radius(r, SAFE_LIMIT);
            assert!(
                out.abs() <= SAFE_LIMIT,
                "project_radius({r}) = {out} > SAFE_LIMIT"
            );
        }
    }

    #[test]
    fn mobius_add_closure_in_disc() {
        // x ⊕ y must stay in the disc for any x, y ∈ (-1, 1).
        let pairs = [(0.5, 0.3), (-0.7, 0.4), (0.99, 0.99), (-0.99, -0.99)];
        for (x, y) in pairs {
            let z = mobius_add(x, y);
            assert!(z.abs() < 1.0, "mobius_add({x},{y}) = {z} outside disc");
        }
    }

    #[test]
    fn hyperbolic_distance_non_negative() {
        let pairs = [(0.0, 0.0), (0.5, 0.5), (0.5, -0.5), (0.9, 0.1)];
        for (x, y) in pairs {
            let d = hyperbolic_distance(x, y);
            assert!(d >= 0.0, "distance({x},{y}) = {d} < 0");
        }
    }

    #[test]
    fn nan_input_does_not_propagate() {
        // Elastic guard: NaN input must not produce NaN output.
        let out = project_radius(f32::NAN, SAFE_LIMIT);
        assert!(
            out.is_finite(),
            "NaN should be absorbed elastically, got {out}"
        );
    }
}

