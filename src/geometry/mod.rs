//! Geometry — Elastic Math for Poincaré Manifolds & Quaternions
//!
//! All operations use asymptotic functions (e.g. `tanh`) softly scaled and
//! pressed against the curvature boundary. No hard panics on domain violations.

pub mod quaternion;
pub mod poincare;

use ndarray::Array1;

/// Hyperbolic point in the Poincaré ball.
#[derive(Debug, Clone)]
pub struct HyperbolicPoint {
    pub coords: Vec<f64>,
}

impl HyperbolicPoint {
    /// Creates a new point, validating that it lies inside the unit ball.
    pub fn new(coords: Array1<f64>) -> Result<Self, crate::LabError> {
        let norm = coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            return Err(crate::LabError::InvalidState);
        }
        Ok(Self { coords: coords.to_vec() })
    }

    /// Euclidean norm of the coordinates.
    pub fn euclidean_norm(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// Poincaré ball model with a given curvature.
#[derive(Debug, Clone, Copy)]
pub struct PoincareBall {
    pub curvature: f64,
}

impl PoincareBall {
    pub fn new(curvature: f64) -> Self {
        Self { curvature }
    }

    /// Exponential map at the origin: lifts a tangent vector to the disc.
    pub fn exp_map_origin(&self, v: &Array1<f64>) -> Result<HyperbolicPoint, crate::LabError> {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return HyperbolicPoint::new(Array1::zeros(v.len()));
        }
        let scale = norm.tanh() / norm;
        let coords: Vec<f64> = v.iter().map(|x| x * scale).collect();
        HyperbolicPoint::new(Array1::from(coords))
    }

    /// Exponential map at a base point (elastic approximation).
    pub fn exp_map(&self, base: &HyperbolicPoint, v: &Array1<f64>) -> Result<HyperbolicPoint, crate::LabError> {
        let mut coords = vec![0.0; base.coords.len()];
        for i in 0..base.coords.len() {
            coords[i] = base.coords[i] + v[i];
        }
        let norm = coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm;
            for x in &mut coords {
                *x *= scale;
            }
        }
        HyperbolicPoint::new(Array1::from(coords))
    }

    /// Hyperbolic distance between two points (Euclidean proxy for now).
    pub fn distance(&self, a: &HyperbolicPoint, b: &HyperbolicPoint) -> Result<f64, crate::LabError> {
        let dist = a.coords.iter().zip(&b.coords)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>().sqrt();
        Ok(dist)
    }
}


/// Poincaré disk model for hyperbolic geometry operations.
///
/// Values are softly clamped so they never escape the disc boundary.
#[derive(Debug, Clone, Copy)]
pub struct PoincareDisk {
    /// Curvature of the disk (typically -1.0).
    pub curvature: f32,
}

impl PoincareDisk {
    /// Creates a new Poincaré disk with the default curvature of -1.0.
    pub fn new() -> Self {
        Self { curvature: -1.0 }
    }

    /// Softly clamps `value` to the interval `[-limit, limit]`.
    ///
    /// Returns `0.0` for NaN. Uses `tanh` so the mapping is smooth everywhere.
    pub fn soft_clamp(&self, value: f32, limit: f32) -> f32 {
        if value.is_nan() {
            return 0.0;
        }
        limit * value.tanh()
    }

    /// Simplified 1D Poincaré addition (Möbius addition).
    ///
    /// Returns a value softly bounded by the hyperbolic geometry.
    pub fn mobius_addition(&self, a: f32, b: f32) -> f32 {
        if a.is_nan() || b.is_nan() {
            return 0.0;
        }
        let a = a.clamp(-0.9999, 0.9999);
        let b = b.clamp(-0.9999, 0.9999);
        let num = a.atanh() + b.atanh();
        let denom = 1.0 + a * b;
        if denom.abs() < 1e-6 {
            return num.tanh();
        }
        num.tanh() / denom
    }
}

impl Default for PoincareDisk {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Unit quaternion for elastic rotations.
///
/// Normalization never panics; falls back to identity on degenerate input.
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quaternion {
    /// Creates a new quaternion.
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Returns the conjugate of the quaternion.
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Hamilton product (quaternion multiplication).
    pub fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Returns the Euclidean norm (magnitude) of the quaternion.
    pub fn norm(self) -> f32 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Softly normalises the quaternion.
    ///
    /// If the magnitude is near zero or NaN, returns the identity quaternion.
    pub fn normalize(&self) -> Self {
        let mag_sq = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        if mag_sq < 1e-12_f32.powi(2) {
            return Self::IDENTITY;
        }
        let inv_mag = mag_sq.sqrt().recip();
        Self {
            w: self.w * inv_mag,
            x: self.x * inv_mag,
            y: self.y * inv_mag,
            z: self.z * inv_mag,
        }
    }
}

impl std::ops::Add for Quaternion {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            w: self.w + other.w,
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::AddAssign for Quaternion {
    fn add_assign(&mut self, other: Self) {
        self.w += other.w;
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl std::ops::Mul<f32> for Quaternion {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            w: self.w * scalar,
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Quaternion {
    /// Identity quaternion (no rotation).
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_clamp_nan_returns_zero() {
        let disk = PoincareDisk::new();
        assert_eq!(disk.soft_clamp(f32::NAN, 1.0), 0.0);
    }

    #[test]
    fn soft_clamp_bounds() {
        let disk = PoincareDisk::new();
        let v = disk.soft_clamp(1e6, 1.0);
        assert!(v <= 1.0);
        assert!(v >= -1.0);
    }

    #[test]
    fn quaternion_normalize_degenerate() {
        let q = Quaternion::new(0.0, 0.0, 0.0, 0.0);
        let n = q.normalize();
        assert_eq!(n, Quaternion::IDENTITY);
    }

    #[test]
    fn quaternion_normalize_valid() {
        let q = Quaternion::new(3.0, 4.0, 0.0, 0.0);
        let n = q.normalize();
        assert!((n.w * n.w + n.x * n.x + n.y * n.y + n.z * n.z - 1.0).abs() < 1e-5);
    }
}


