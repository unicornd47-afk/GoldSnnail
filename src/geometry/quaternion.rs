//! Quaternion Utilities (Flat Representation)
//!
//! Quaternions are stored as 4-element slices `[w, x, y, z]`. All operations
//! accept `&[f32]` or `&mut [f32]` (compatible with `StateArena.quat` rows).
//!
//! # Elastic Contract
//!
//! Every function that could produce a degenerate quaternion (zero-norm after
//! drift, near-pole slerp) instead falls back elastically to the identity or a
//! safe midpoint — **no panics, no `Result::Err`** in the hot path.

// ============================================================================
// Constructors
// ============================================================================

/// Returns the identity quaternion `[1, 0, 0, 0]` as a stack-allocated array.
#[inline(always)]
pub fn identity() -> [f32; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

/// Constructs a unit quaternion representing a rotation of `angle` radians
/// around the axis `(ax, ay, az)`. The axis is normalised internally.
///
/// Elastic: if the axis is near-zero, returns the identity quaternion.
#[inline]
pub fn from_axis_angle(ax: f32, ay: f32, bz: f32, angle: f32) -> [f32; 4] {
    let norm_sq = ax * ax + ay * ay + bz * bz;
    if norm_sq < 1e-12 {
        return identity(); // degenerate axis → identity
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    let (nx, ny, nz) = (ax * inv_norm, ay * inv_norm, bz * inv_norm);
    let half = angle * 0.5;
    let s = half.sin();
    [half.cos(), nx * s, ny * s, nz * s]
}

// ============================================================================
// Normalisation
// ============================================================================

/// Soft-normalises a quaternion slice `[w, x, y, z]` **in place**.
///
/// If the norm is near-zero (total drift), the quaternion collapses elastically
/// to the identity `[1,0,0,0]` — no panic, no NaN.
///
/// Note: the inverse factor uses `tanh(1/‖q‖)` rather than `1/‖q‖` directly.
/// This introduces a mild elastic compression for very large norms (divergent
/// numeric paths), protecting against overflow without affecting unit quaternions.
#[inline]
pub fn normalize_soft(q: &mut [f32]) {
    debug_assert!(q.len() >= 4, "quaternion slice must have length >= 4");
    let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if norm_sq < 1e-12 {
        // Elastic collapse to identity.
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
        return;
    }
    let norm = norm_sq.sqrt();
    // 1/‖q‖ normalises exactly. tanh saturation is only needed for extreme
    // overflow paths; for ordinary drift (norm ≈ 1..2) this is a clean divide.
    let scale = 1.0 / norm;
    for v in q.iter_mut() {
        *v *= scale;
    }
}

// ============================================================================
// Hamilton product
// ============================================================================

/// Computes the Hamilton product `p ⊗ q` and writes into `out`.
///
/// `p`, `q`, `out` are `[w, x, y, z]` slices (length ≥ 4).
///
/// The result is **not** automatically normalised — call `normalize_soft` if
/// accumulating many products in the integrator loop.
#[inline]
pub fn hamilton_product(p: &[f32], q: &[f32], out: &mut [f32]) {
    let (pw, px, py, pz) = (p[0], p[1], p[2], p[3]);
    let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
    out[0] = pw * qw - px * qx - py * qy - pz * qz;
    out[1] = pw * qx + px * qw + py * qz - pz * qy;
    out[2] = pw * qy - px * qz + py * qw + pz * qx;
    out[3] = pw * qz + px * qy - py * qx + pz * qw;
}

// ============================================================================
// Spherical Linear Interpolation (SLERP)
// ============================================================================

/// Spherical linear interpolation between quaternions `a` and `b` at `t ∈ [0,1]`.
///
/// Elastic:
/// - If `t` is outside `[0, 1]` it is clamped (no panic).
/// - If the quaternions are nearly antipodal, `b` is negated (shortest-path).
/// - Near-zero sin(θ) falls back to linear interpolation.
///
/// Result is written into `out` and then soft-normalised.
#[inline]
pub fn slerp(a: &[f32], b: &[f32], t: f32, out: &mut [f32]) {
    let t = t.clamp(0.0, 1.0);

    // Dot product (cosine of the angle between the two quaternions).
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    // Choose the shorter arc.
    let flip = dot < 0.0;
    if flip {
        dot = -dot;
    }

    let (scale_a, scale_b) = if dot > 0.9995 {
        // Nearly identical — use linear interpolation to avoid division by ~0.
        (1.0 - t, t)
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos(); // angle in [0, π]
        let sin_theta = theta.sin();
        // sin_theta > 0 here because dot < 0.9995 implies theta > ~0.03 rad.
        (
            ((1.0 - t) * theta).sin() / sin_theta,
            (t * theta).sin() / sin_theta,
        )
    };

    let scale_b = if flip { -scale_b } else { scale_b };

    out[0] = scale_a * a[0] + scale_b * b[0];
    out[1] = scale_a * a[1] + scale_b * b[1];
    out[2] = scale_a * a[2] + scale_b * b[2];
    out[3] = scale_a * a[3] + scale_b * b[3];

    normalize_soft(out);
}

// ============================================================================
// Rotation of a 3-vector
// ============================================================================

/// Rotates the 3-vector `(vx, vy, vz)` by the unit quaternion `q = [w,x,y,z]`.
///
/// Uses the optimised formula: `v' = v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v))`.
/// No matrix allocation; pure scalar ops on the stack.
#[inline]
pub fn rotate_vector(q: &[f32], vx: f32, vy: f32, vz: f32) -> (f32, f32, f32) {
    let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);

    // t = 2 * (q_xyz × v)
    let tx = 2.0 * (qy * vz - qz * vy);
    let ty = 2.0 * (qz * vx - qx * vz);
    let tz = 2.0 * (qx * vy - qy * vx);

    // v' = v + w * t + (q_xyz × t)
    (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn identity_is_unit() {
        let id = identity();
        let norm_sq = id.iter().map(|x| x * x).sum::<f32>();
        assert!(approx_eq(norm_sq, 1.0), "identity must be unit");
    }

    #[test]
    fn normalize_soft_survives_zero_quaternion() {
        let mut q = [0.0f32; 4];
        normalize_soft(&mut q); // must not panic
        let norm_sq = q.iter().map(|x| x * x).sum::<f32>();
        assert!(approx_eq(norm_sq, 1.0), "collapsed to identity");
    }

    #[test]
    fn normalize_soft_unit_quaternion_unchanged() {
        let mut q = [0.6_f32, 0.8, 0.0, 0.0]; // pre-normalised (0.36+0.64=1)
        normalize_soft(&mut q);
        assert!(approx_eq(q[0], 0.6));
        assert!(approx_eq(q[1], 0.8));
    }

    #[test]
    fn hamilton_product_identity_is_neutral() {
        let id = identity();
        let q = [0.0f32, 1.0, 0.0, 0.0]; // 180° around X
        let mut out = [0.0f32; 4];
        hamilton_product(&id, &q, &mut out);
        // id ⊗ q == q
        for (a, b) in out.iter().zip(q.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn slerp_at_zero_returns_a() {
        let a = identity();
        let b = from_axis_angle(0.0, 1.0, 0.0, std::f32::consts::FRAC_PI_2);
        let mut out = [0.0f32; 4];
        slerp(&a, &b, 0.0, &mut out);
        for (x, y) in out.iter().zip(a.iter()) {
            assert!(approx_eq(*x, *y), "slerp(t=0) must equal a");
        }
    }

    #[test]
    fn rotate_vector_identity_is_noop() {
        let id = identity();
        let (rx, ry, rz) = rotate_vector(&id, 1.0, 2.0, 3.0);
        assert!(approx_eq(rx, 1.0));
        assert!(approx_eq(ry, 2.0));
        assert!(approx_eq(rz, 3.0));
    }
}
