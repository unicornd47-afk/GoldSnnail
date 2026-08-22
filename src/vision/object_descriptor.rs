//! Object-level descriptors for ARC grids — Transform-Codec Stage 0.
//!
//! Feeds transform extraction (Stage 1) and committee voting (Stage 3) with
//! per-object shape/color/position descriptors. No learned model: everything
//! is exact, O(n) per object, and deterministic.
//!
//! # Design principles
//! - **DOD-compliant:** flat arrays, no pointer chasing in hot loops.
//! - **Elastic failure:** empty grids yield empty descriptor sets, never panic.
//! - **Invariant matching:** Hu moments give rotation/translation/scale
//!   invariance; the canonicalized border histogram adds 90°-rotation
//!   invariant contour information (ARC symmetries are D4).

use std::collections::HashSet;

use crate::arc_parser::{extract_components_with, Component};
use crate::vision::ArcGrid;

/// Rotation/translation/scale-invariant descriptor of one ARC object.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectDescriptor {
    /// The object's color.
    pub color: u8,
    /// Number of pixels.
    pub size: usize,
    /// Bounding box (min_row, max_row, min_col, max_col).
    pub bbox: (usize, usize, usize, usize),
    /// Centroid in grid coordinates.
    pub centroid: (f64, f64),
    /// Log-scaled Hu moments φ1..φ7 (invariant under rotation/translation/scale).
    pub hu: [f64; 7],
    /// Canonicalized 8-bin border direction histogram (90°-rotation invariant).
    pub border: [f64; 8],
    /// Compactness: perimeter² / (4π·area). 1.0 ≈ disc; larger = spiky.
    pub compactness: f64,
}

impl ObjectDescriptor {
    /// Builds a descriptor from an extracted connected component.
    pub fn from_component(comp: &Component) -> Self {
        let (hu, _) = hu_moments(&comp.pixels);
        let (border, perimeter) = border_histogram(&comp.pixels);
        let area = comp.pixels.len() as f64;
        let centroid = comp
            .pixels
            .iter()
            .fold((0.0f64, 0.0f64), |(sr, sc), &(r, c)| {
                (sr + r as f64, sc + c as f64)
            });
        Self {
            color: comp.color,
            size: comp.size(),
            bbox: comp.bbox,
            centroid: (centroid.0 / area, centroid.1 / area),
            hu,
            border,
            compactness: if area > 0.0 {
                (perimeter * perimeter) / (4.0 * std::f64::consts::PI * area)
            } else {
                f64::INFINITY
            },
        }
    }

    /// Extracts descriptors for all objects in a grid.
    ///
    /// `diag = true` enables 8-connectivity (recommended for ARC object
    /// extraction); `false` is 4-connectivity (color-region semantics).
    pub fn describe_grid(grid: &ArcGrid, diag: bool) -> Vec<ObjectDescriptor> {
        extract_components_with(grid, diag)
            .iter()
            .map(ObjectDescriptor::from_component)
            .collect()
    }

    /// Shape-only distance (log-scaled Hu moments), invariant under
    /// rotation/translation/scale. Use for object matching across grids.
    pub fn hu_distance(&self, other: &ObjectDescriptor) -> f64 {
        self.hu
            .iter()
            .zip(&other.hu)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ─── Raw moments ─────────────────────────────────────────────────────────────

/// Raw moments up to order 3 of a binary mask:
/// m00, m10, m01, m20, m11, m02, m30, m21, m12, m03.
fn raw_moments(pixels: &[(usize, usize)]) -> [f64; 10] {
    let mut m = [0.0f64; 10];
    for &(r, c) in pixels {
        let r = r as f64;
        let c = c as f64;
        m[0] += 1.0;
        m[1] += c;
        m[2] += r;
        m[3] += c * c;
        m[4] += c * r;
        m[5] += r * r;
        m[6] += c * c * c;
        m[7] += c * c * r;
        m[8] += c * r * r;
        m[9] += r * r * r;
    }
    m
}

/// The seven Hu moments of a binary mask, log-scaled:
/// `h_i = sign(φ_i) · ln(1 + |φ_i|)`.
///
/// # Returns
/// `(hu, raw_φ)` — the log-scaled values (used for matching) and the raw
/// values (useful for debugging/ablation).
pub fn hu_moments(pixels: &[(usize, usize)]) -> ([f64; 7], [f64; 7]) {
    let m = raw_moments(pixels);
    if m[0] == 0.0 {
        return ([0.0; 7], [0.0; 7]);
    }
    let (cx, cy) = (m[1] / m[0], m[2] / m[0]);
    // Central moments.
    let mu20 = m[3] - m[0] * cx * cx;
    let mu11 = m[4] - m[0] * cx * cy;
    let mu02 = m[5] - m[0] * cy * cy;
    let mu30 = m[6] - 3.0 * m[3] * cx + 2.0 * m[0] * cx * cx * cx;
    let mu21 = m[7] - m[3] * cy - 2.0 * m[4] * cx + 2.0 * m[0] * cx * cx * cy;
    let mu12 = m[8] - m[5] * cx - 2.0 * m[4] * cy + 2.0 * m[0] * cx * cy * cy;
    let mu03 = m[9] - 3.0 * m[5] * cy + 2.0 * m[0] * cy * cy * cy;
    // Scale normalization: η_pq = μ_pq / μ00^(1 + (p+q)/2).
    let m00 = m[0];
    let m00_2 = m00 * m00;
    let m00_25 = m00_2 * m00.sqrt();
    let n20 = mu20 / m00_2;
    let n11 = mu11 / m00_2;
    let n02 = mu02 / m00_2;
    let n30 = mu30 / m00_25;
    let n21 = mu21 / m00_25;
    let n12 = mu12 / m00_25;
    let n03 = mu03 / m00_25;

    let p1 = n20 + n02;
    let p2 = (n20 - n02).powi(2) + 4.0 * n11 * n11;
    let p3 = (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2);
    let p4 = (n30 + n12).powi(2) + (n21 + n03).powi(2);
    let p5 = (n30 - 3.0 * n12) * (n30 + n12)
        * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        + (3.0 * n21 - n03) * (n21 + n03)
            * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));
    let p6 = (n20 - n02) * ((n30 + n12).powi(2) - (n21 + n03).powi(2))
        + 4.0 * n11 * (n30 + n12) * (n21 + n03);
    let p7 = (3.0 * n21 - n03) * (n30 + n12)
        * ((n30 + n12).powi(2) - 3.0 * (n21 + n03).powi(2))
        - (n30 - 3.0 * n12) * (n21 + n03)
            * (3.0 * (n30 + n12).powi(2) - (n21 + n03).powi(2));
    let raw = [p1, p2, p3, p4, p5, p6, p7];
    let log_scale = |p: f64| p.signum() * (1.0 + p.abs()).ln();
    (
        [
            log_scale(p1),
            log_scale(p2),
            log_scale(p3),
            log_scale(p4),
            log_scale(p5),
            log_scale(p6),
            log_scale(p7),
        ],
        raw,
    )
}

// ─── Border signature ────────────────────────────────────────────────────────

/// 8-direction border histogram + perimeter length.
///
/// The histogram counts, for each object cell, the 8-neighbors outside the
/// object. It is canonicalized by taking the lexicographic minimum over 90°
/// rotations (shifts by 2 bins), making it invariant under the D4 symmetry
/// group of ARC grids.
fn border_histogram(pixels: &[(usize, usize)]) -> ([f64; 8], f64) {
    let set: HashSet<(usize, usize)> = pixels.iter().copied().collect();
    let mut histo = [0.0f64; 8];
    const DIRS: [(isize, isize); 8] = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ];
    for &(r, c) in pixels {
        for (i, &(dr, dc)) in DIRS.iter().enumerate() {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            let inside = nr >= 0
                && nc >= 0
                && set.contains(&(nr as usize, nc as usize));
            if !inside {
                histo[i] += 1.0;
            }
        }
    }
    let perimeter = histo.iter().sum::<f64>();
    let shift = |h: &[f64; 8], s: usize| -> [f64; 8] {
        let mut out = [0.0f64; 8];
        for (i, item) in out.iter_mut().enumerate() {
            *item = h[(i + s) % 8];
        }
        out
    };
    let mut best = histo;
    for s in [2usize, 4, 6] {
        let cand = shift(&histo, s);
        if cand < best {
            best = cand;
        }
    }
    (best, perimeter)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn grid(pixels: &[(usize, usize)], w: usize, h: usize) -> ArcGrid {
        let mut data = vec![vec![0u8; w]; h];
        for &(r, c) in pixels {
            data[r][c] = 1;
        }
        ArcGrid {
            data,
            width: w,
            height: h,
        }
    }

    /// L-tetromino shape, used across invariance tests.
    fn l_shape() -> Vec<(usize, usize)> {
        vec![(0, 0), (0, 1), (0, 2), (1, 0)]
    }

    /// Rotates pixel coordinates 90° clockwise inside an S×S container.
    fn rot90(pixels: &[(usize, usize)], s: usize) -> Vec<(usize, usize)> {
        pixels.iter().map(|&(r, c)| (c, s - 1 - r)).collect()
    }

    /// Scales pixel coordinates by factor f (integer grid).
    fn scale(pixels: &[(usize, usize)], f: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for &(r, c) in pixels {
            for dr in 0..f {
                for dc in 0..f {
                    out.push((r * f + dr, c * f + dc));
                }
            }
        }
        out
    }

    #[test]
    fn hu_invariant_under_90_rotation() {
        let a = hu_moments(&l_shape()).0;
        for _ in 0..3 {
            let b = hu_moments(&rot90(&l_shape(), 8)).0;
            for (x, y) in a.iter().zip(&b) {
                assert_abs_diff_eq!(x, y, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn hu_invariant_under_scaling() {
        let a = hu_moments(&l_shape()).0;
        let b = hu_moments(&scale(&l_shape(), 2)).0;
        for (x, y) in a.iter().zip(&b) {
            assert_abs_diff_eq!(x, y, epsilon = 0.05);
        }
    }

    #[test]
    fn border_invariant_under_90_rotation() {
        let a = ObjectDescriptor::from_component(&Component {
            color: 1,
            pixels: l_shape(),
            bbox: (0, 1, 0, 2),
        });
        let b = ObjectDescriptor::from_component(&Component {
            color: 1,
            pixels: rot90(&l_shape(), 8),
            bbox: (0, 2, 5, 7),
        });
        for (x, y) in a.border.iter().zip(&b.border) {
            assert_abs_diff_eq!(x, y, epsilon = 1e-9);
        }
    }

    #[test]
    fn describe_grid_diagonal_connectivity() {
        // Diagonal line: 3 objects under 4-CC, 1 object under 8-CC.
        let g = grid(&[(0, 0), (1, 1), (2, 2)], 4, 4);
        assert_eq!(ObjectDescriptor::describe_grid(&g, false).len(), 3);
        assert_eq!(ObjectDescriptor::describe_grid(&g, true).len(), 1);
    }

    #[test]
    fn hu_distance_same_shape_zero() {
        let a = ObjectDescriptor::from_component(&Component {
            color: 2,
            pixels: l_shape(),
            bbox: (0, 1, 0, 2),
        });
        let b = ObjectDescriptor::from_component(&Component {
            color: 3,
            pixels: rot90(&l_shape(), 8),
            bbox: (0, 2, 5, 7),
        });
        assert!(a.hu_distance(&b) < 1e-6);
    }
}
