//! Transform-Codec Stage 1 — transform extraction between ARC grids.
//!
//! The codec searches for the *transform* that maps the input grid's objects
//! onto the output — not for the output image itself. Extraction order:
//!
//! 1. **Dihedral (D4)** — exact whole-grid rotation/reflection check (8
//!    elements, O(n) each, µs).
//! 2. **Color map** — consistent color renaming (dims preserved).
//! 3. **Tiling** — output = input repeated on a lattice (self-similarity;
//!    targets ARC-1 PatternExtend/SizeChange and ARC-2 self-sim).
//! 4. **Similarity (ℂ)** — 2D similarity transform `z ↦ a·z + b` (or
//!    `a·z̄ + b` for reflections) fitted between matched objects. In the
//!    complex plane the least-squares fit is a single division (Horn's
//!    method for 2D); quaternions/SU(2) are deliberately NOT used — for 2D
//!    grids ℂ is the correct algebra, D4 its discrete subgroup.
//!
//! # Design principles
//! - Exact-first: deterministic rules beat fitted transforms whenever one
//!   explains the pair exactly.
//! - DOD-compliant: flat data, no heap churn in hot loops.
//! - Elastic failure: unknown/unmatchable pairs return `TransformKind::Unknown`.

use crate::arc_parser::extract_components_with;
use crate::vision::{ArcGrid, ObjectDescriptor};

/// Checks if a grid is exactly 10x10.
pub fn is_10x10(grid: &ArcGrid) -> bool {
    grid.width == 10 && grid.height == 10
}

/// The kind of transform found between two grids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformKind {
    /// One of the 8 dihedral symmetries (D4).
    Dihedral,
    /// 2D similarity `z ↦ a·z + b` (rotation+scale+translation, optional reflection).
    Similarity,
    /// Output is the input repeated on an n×m lattice.
    Tiling,
    /// Consistent color renaming.
    ColorMap,
    /// No single transform explains the pair (→ committee / program search).
    Unknown,
}

/// Parameters of a found transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformParams {
    /// `d4_index` into [`apply_d4`] (0..=7).
    Dihedral { d4_index: u8 },
    /// `a = a_re + i·a_im` (rotation+scale), `b = b_c + i·b_r` (translation).
    Similarity {
        a_re: f64,
        a_im: f64,
        b_r: f64,
        b_c: f64,
        flip: bool,
    },
    /// Tile count (columns n, rows m).
    Tiling { n: usize, m: usize },
    /// `mapping[src_color] = dst_color` (255 = unused).
    ColorMap { mapping: [u8; 10] },
    None,
}

/// A extracted transform with its residual error (0.0 = exact).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformCode {
    pub kind: TransformKind,
    pub params: TransformParams,
    /// RMS pixel error of the fit (0.0 for exact rules).
    pub residual: f64,
}

/// Result of the ℂ similarity fit (Horn's method in 2D).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityFit {
    pub a_re: f64,
    pub a_im: f64,
    pub b_r: f64,
    pub b_c: f64,
    /// True when the reflected model `z ↦ a·z̄ + b` fit better.
    pub flip: bool,
    /// RMS error after the fit, in pixels.
    pub residual: f64,
}

// ─── Dihedral group D4 ───────────────────────────────────────────────────────

/// Applies the `idx`-th dihedral transform (0..=7) to a grid.
///
/// Indices: 0=identity, 1=rot90, 2=rot180, 3=rot270, 4=flip horizontal,
/// 5=flip vertical, 6=transpose (main diagonal), 7=anti-diagonal reflection.
pub fn apply_d4(grid: &ArcGrid, idx: u8) -> ArcGrid {
    let (h, w) = (grid.height, grid.width);
    match idx {
        0 => ArcGrid {
            data: grid.data.clone(),
            width: w,
            height: h,
        },
        1 => {
            // rot90 CW: (r, c) -> (c, h-1-r); dims swap
            let mut data = vec![vec![0u8; h]; w];
            for r in 0..h {
                for c in 0..w {
                    data[c][h - 1 - r] = grid.data[r][c];
                }
            }
            ArcGrid {
                data,
                width: h,
                height: w,
            }
        }
        2 => {
            let mut data = vec![vec![0u8; w]; h];
            for r in 0..h {
                for c in 0..w {
                    data[h - 1 - r][w - 1 - c] = grid.data[r][c];
                }
            }
            ArcGrid {
                data,
                width: w,
                height: h,
            }
        }
        3 => {
            // rot270 CW (90 CCW): (r, c) -> (w-1-c, r); dims swap
            let mut data = vec![vec![0u8; h]; w];
            for r in 0..h {
                for c in 0..w {
                    data[w - 1 - c][r] = grid.data[r][c];
                }
            }
            ArcGrid {
                data,
                width: h,
                height: w,
            }
        }
        4 => {
            let mut data = vec![vec![0u8; w]; h];
            for r in 0..h {
                for c in 0..w {
                    data[r][w - 1 - c] = grid.data[r][c];
                }
            }
            ArcGrid {
                data,
                width: w,
                height: h,
            }
        }
        5 => {
            let mut data = vec![vec![0u8; w]; h];
            for r in 0..h {
                for c in 0..w {
                    data[h - 1 - r][c] = grid.data[r][c];
                }
            }
            ArcGrid {
                data,
                width: w,
                height: h,
            }
        }
        6 => {
            let mut data = vec![vec![0u8; h]; w];
            for r in 0..h {
                for c in 0..w {
                    data[c][r] = grid.data[r][c];
                }
            }
            ArcGrid {
                data,
                width: h,
                height: w,
            }
        }
        7 => {
            let mut data = vec![vec![0u8; h]; w];
            for r in 0..h {
                for c in 0..w {
                    data[w - 1 - c][h - 1 - r] = grid.data[r][c];
                }
            }
            ArcGrid {
                data,
                width: h,
                height: w,
            }
        }
        _ => ArcGrid {
            data: grid.data.clone(),
            width: w,
            height: h,
        },
    }
}

/// Returns the D4 index mapping `input` onto `output` exactly, if any.
pub fn find_d4(input: &ArcGrid, output: &ArcGrid) -> Option<u8> {
    (0..8u8).find(|&idx| {
        let t = apply_d4(input, idx);
        t.width == output.width && t.height == output.height && t.data == output.data
    })
}

// ─── Color map ───────────────────────────────────────────────────────────────

/// Consistent color renaming `input → output` (dims preserved). 255 = unused.
pub fn find_color_map(input: &ArcGrid, output: &ArcGrid) -> Option<[u8; 10]> {
    if input.width != output.width || input.height != output.height {
        return None;
    }
    let mut map = [255u8; 10];
    for r in 0..input.height {
        for c in 0..input.width {
            let a = input.data[r][c];
            let b = output.data[r][c];
            if a == 0 && b != 0 {
                return None; // background must stay background
            }
            if map[a as usize] == 255 {
                map[a as usize] = b;
            } else if map[a as usize] != b {
                return None;
            }
        }
    }
    Some(map)
}

// ─── Tiling / self-similarity ────────────────────────────────────────────────

/// Detects `output = input` repeated on an n×m lattice (axis-aligned).
///
/// This is the self-similarity detector: ARC-1 PatternExtend/SizeChange and
/// the ARC-2 self-sim slice all live here.
pub fn find_tiling(input: &ArcGrid, output: &ArcGrid) -> Option<(usize, usize)> {
    if input.width == 0 || input.height == 0 {
        return None;
    }
    if output.width % input.width != 0 || output.height % input.height != 0 {
        return None;
    }
    let n = output.width / input.width;
    let m = output.height / input.height;
    if n <= 1 && m <= 1 {
        return None;
    }
    for tr in 0..m {
        for tc in 0..n {
            for r in 0..input.height {
                for c in 0..input.width {
                    if output.data[tr * input.height + r][tc * input.width + c]
                        != input.data[r][c]
                    {
                        return None;
                    }
                }
            }
        }
    }
    Some((n, m))
}

// ─── ℂ similarity fit ────────────────────────────────────────────────────────

type C = (f64, f64);

#[inline]
fn cadd(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}
#[inline]
fn csub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}
#[inline]
fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
#[inline]
fn cconj(a: C) -> C {
    (a.0, -a.1)
}
#[inline]
fn cdiv(a: C, b: C) -> C {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
#[inline]
fn cscale(a: C, s: f64) -> C {
    (a.0 * s, a.1 * s)
}
#[inline]
fn cnorm2(a: C) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

/// Fits the 2D similarity transform between two matched point sets.
///
/// Model: `z' = a·z + b` (orientation-preserving) or `z' = a·z̄ + b`
/// (reflected), with `z = col + i·row`. Both variants are fitted in closed
/// form; the one with the lower residual wins.
///
/// # Arguments
/// * `a_pts` — source points (matched 1:1 with `b_pts`)
/// * `b_pts` — target points
///
/// # Returns
/// The fit, or `None` for empty/unmatched input.
pub fn similarity_fit(a_pts: &[(usize, usize)], b_pts: &[(usize, usize)]) -> Option<SimilarityFit> {
    if a_pts.is_empty() || a_pts.len() != b_pts.len() {
        return None;
    }
    let n = a_pts.len() as f64;
    let z = |p: (usize, usize)| (p.1 as f64, p.0 as f64); // (row, col) -> col + i·row
    let mut ma = (0.0, 0.0);
    let mut mb = (0.0, 0.0);
    for (&pa, &pb) in a_pts.iter().zip(b_pts) {
        ma = cadd(ma, z(pa));
        mb = cadd(mb, z(pb));
    }
    ma = cscale(ma, 1.0 / n);
    mb = cscale(mb, 1.0 / n);

    let mut num_d = (0.0, 0.0); // Σ z'·z̄  (direct)
    let mut num_f = (0.0, 0.0); // Σ z'·z   (reflected)
    let mut den = 0.0;          // Σ |z|²
    for (&pa, &pb) in a_pts.iter().zip(b_pts) {
        let za = csub(z(pa), ma);
        let zb = csub(z(pb), mb);
        num_d = cadd(num_d, cmul(zb, cconj(za)));
        num_f = cadd(num_f, cmul(zb, za));
        den += cnorm2(za);
    }
    if den < 1e-12 {
        return None;
    }
    let a_direct = cdiv(num_d, (den, 0.0));
    let a_flip = cdiv(num_f, (den, 0.0));
    let b_direct = csub(mb, cmul(a_direct, ma));
    // reflected model: z' = a·z̄ + b → b = mean' − a·conj(mean)
    let b_flip = csub(mb, cmul(a_flip, cconj(ma)));

    let res = |a: C, b: C, flip: bool| -> f64 {
        let mut s = 0.0;
        for (&pa, &pb) in a_pts.iter().zip(b_pts) {
            let za = z(pa);
            let pred = if flip {
                cadd(cmul(a, cconj(za)), b)
            } else {
                cadd(cmul(a, za), b)
            };
            s += cnorm2(csub(z(pb), pred));
        }
        s / n
    };

    let (a, b, flip, mse) = if res(a_direct, b_direct, false) <= res(a_flip, b_flip, true) {
        (a_direct, b_direct, false, res(a_direct, b_direct, false))
    } else {
        (a_flip, b_flip, true, res(a_flip, b_flip, true))
    };

    Some(SimilarityFit {
        a_re: a.0,
        a_im: a.1,
        b_r: b.1, // imag = row
        b_c: b.0, // real = col
        flip,
        residual: mse.sqrt(),
    })
}

// ─── Full extraction ─────────────────────────────────────────────────────────

/// Extracts the best single transform between two grids.
///
/// Exact rules first (D4 → color map → tiling), then the object-level
/// similarity fit, else `Unknown`.
pub fn extract_transform(input: &ArcGrid, output: &ArcGrid) -> TransformCode {
    if let Some(idx) = find_d4(input, output) {
        return TransformCode {
            kind: TransformKind::Dihedral,
            params: TransformParams::Dihedral { d4_index: idx },
            residual: 0.0,
        };
    }
    if input.width == output.width && input.height == output.height {
        if let Some(map) = find_color_map(input, output) {
            let is_identity = (0..10).all(|i| map[i] == 255 || map[i] == i as u8);
            if !is_identity {
                return TransformCode {
                    kind: TransformKind::ColorMap,
                    params: TransformParams::ColorMap { mapping: map },
                    residual: 0.0,
                };
            }
        }
    }
    if let Some((n, m)) = find_tiling(input, output) {
        return TransformCode {
            kind: TransformKind::Tiling,
            params: TransformParams::Tiling { n, m },
            residual: 0.0,
        };
    }

    // Object-level similarity: match descriptors by shape, fit the transform.
    let comps_a = extract_components_with(input, true);
    let comps_b = extract_components_with(output, true);
    let descs_a: Vec<ObjectDescriptor> = comps_a.iter().map(ObjectDescriptor::from_component).collect();
    let descs_b: Vec<ObjectDescriptor> = comps_b.iter().map(ObjectDescriptor::from_component).collect();

    let mut best: Option<(f64, SimilarityFit)> = None;
    // (hu_distance, centroid_a, centroid_b, scale) for size-changing matches
    let mut best_scale: Option<(f64, (f64, f64), (f64, f64), f64)> = None;
    for (i, da) in descs_a.iter().enumerate() {
        for (j, db) in descs_b.iter().enumerate() {
            // NOTE: no size-equality gate — Hu moments are scale-invariant,
            // and ARC scale tasks change the object's pixel count.
            let d = da.hu_distance(db);
            if d > 1.0 {
                continue;
            }
            if da.size == db.size {
                if let Some(fit) = similarity_fit(&comps_a[i].pixels, &comps_b[j].pixels) {
                    if fit.residual < 2.0 && best.map_or(true, |(r, _)| fit.residual < r) {
                        best = Some((fit.residual, fit));
                    }
                }
            } else if d < 0.5 && da.size >= 4 && db.size >= 4 {
                // Same shape up to scale: the pixel sets have no 1:1
                // correspondence, so the point-fit does not apply. Emit a
                // uniform-scale transform with residual = shape distance.
                // (Min size 4: 1-2 cell objects carry no shape information.)
                let scale = (db.size as f64 / da.size as f64).sqrt();
                if best_scale.map_or(true, |(bd, _, _, _)| d < bd) {
                    best_scale = Some((d, da.centroid, db.centroid, scale));
                }
            }
        }
    }
    if let Some((residual, fit)) = best {
        return TransformCode {
            kind: TransformKind::Similarity,
            params: TransformParams::Similarity {
                a_re: fit.a_re,
                a_im: fit.a_im,
                b_r: fit.b_r,
                b_c: fit.b_c,
                flip: fit.flip,
            },
            residual,
        };
    }
    if let Some((hu_d, cent_a, cent_b, scale)) = best_scale {
        // z' = a·z + b with a = scale (rotation 0), b from the centroids.
        let (a_re, a_im) = (scale, 0.0);
        let b_c = cent_b.1 - (a_re * cent_a.1 - a_im * cent_a.0);
        let b_r = cent_b.0 - (a_re * cent_a.0 + a_im * cent_a.1);
        return TransformCode {
            kind: TransformKind::Similarity,
            params: TransformParams::Similarity {
                a_re,
                a_im,
                b_r,
                b_c,
                flip: false,
            },
            residual: hu_d,
        };
    }

    TransformCode {
        kind: TransformKind::Unknown,
        params: TransformParams::None,
        residual: f64::INFINITY,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn grid(rows: &[&[u8]]) -> ArcGrid {
        let data: Vec<Vec<u8>> = rows.iter().map(|r| r.to_vec()).collect();
        ArcGrid {
            width: rows[0].len(),
            height: rows.len(),
            data,
        }
    }

    /// Asymmetric 3×4 pattern so no D4 index coincides with another.
    fn asym() -> ArcGrid {
        grid(&[
            &[1, 0, 0, 2],
            &[0, 1, 0, 0],
            &[0, 0, 3, 3],
        ])
    }

    #[test]
    fn d4_recovers_all_eight() {
        let g = asym();
        for idx in 0..8u8 {
            let t = apply_d4(&g, idx);
            assert_eq!(find_d4(&g, &t), Some(idx), "idx {idx} not recovered");
        }
    }

    #[test]
    fn d4_rotation_task_detected() {
        let input = asym();
        let output = apply_d4(&input, 1);
        let code = extract_transform(&input, &output);
        assert_eq!(code.kind, TransformKind::Dihedral);
        assert_eq!(code.params, TransformParams::Dihedral { d4_index: 1 });
        assert_eq!(code.residual, 0.0);
    }

    #[test]
    fn color_map_detected() {
        let input = grid(&[&[1, 1, 2], &[2, 1, 0]]);
        let output = grid(&[&[4, 4, 7], &[7, 4, 0]]);
        let code = extract_transform(&input, &output);
        assert_eq!(code.kind, TransformKind::ColorMap);
    }

    #[test]
    fn tiling_detected() {
        let input = grid(&[&[1, 2], &[2, 1]]);
        let mut output = vec![vec![0u8; 6]; 6];
        for r in 0..6 {
            for c in 0..6 {
                output[r][c] = input.data[r % 2][c % 2];
            }
        }
        let output = ArcGrid {
            data: output,
            width: 6,
            height: 6,
        };
        assert_eq!(find_tiling(&input, &output), Some((3, 3)));
        let code = extract_transform(&input, &output);
        assert_eq!(code.kind, TransformKind::Tiling);
    }

    #[test]
    fn tiling_rejects_mismatch() {
        let input = grid(&[&[1, 2], &[2, 1]]);
        let output = grid(&[&[1, 2, 1, 2], &[2, 1, 2, 1], &[1, 2, 1, 2], &[2, 1, 2, 2]]);
        assert_eq!(find_tiling(&input, &output), None);
    }

    #[test]
    fn similarity_recovers_known_transform() {
        // L-shape points; apply z' = a·z + b with a = 2i (rot90 + scale 2), b = (3, 5).
        let pts: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (0, 2), (1, 0)];
        let (a_re, a_im, b_c, b_r) = (0.0, 2.0, 3.0, 5.0);
        let target: Vec<(usize, usize)> = pts
            .iter()
            .map(|&(r, c)| {
                // z = col + i·row; z' = a·z + b
                let (zr, zc) = (r as f64, c as f64);
                let (pr, pc) = (a_re * zc - a_im * zr + b_c, a_re * zr + a_im * zc + b_r);
                (pc.round() as usize, pr.round() as usize)
            })
            .collect();
        let fit = similarity_fit(&pts, &target).unwrap();
        assert!(!fit.flip);
        assert_abs_diff_eq!(fit.a_re, a_re, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.a_im, a_im, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.b_c, b_c, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.b_r, b_r, epsilon = 1e-9);
        assert!(fit.residual < 1e-9);
    }

    #[test]
    fn similarity_detects_reflection() {
        // Reflect across the vertical axis x=5: z' = -conj(z) + 10 → a = -1, flip.
        let pts: Vec<(usize, usize)> = vec![(1, 2), (1, 5), (3, 4)];
        let target: Vec<(usize, usize)> = pts.iter().map(|&(r, c)| (r, 10 - c)).collect();
        let fit = similarity_fit(&pts, &target).unwrap();
        assert!(fit.flip);
        assert_abs_diff_eq!(fit.a_re, -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.a_im, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.b_c, 10.0, epsilon = 1e-9);
        assert_abs_diff_eq!(fit.b_r, 0.0, epsilon = 1e-9);
        assert!(fit.residual < 1e-9);
    }

    #[test]
    fn extract_scale_task_via_objects() {
        // Single object, scaled ×2 with the same color.
        let input = grid(&[&[0, 0, 0], &[0, 1, 1], &[0, 1, 1]]);
        let mut data = vec![vec![0u8; 6]; 6];
        for r in 0..6 {
            for c in 0..6 {
                data[r][c] = input.data[(r / 2).min(2)][(c / 2).min(2)];
            }
        }
        let output = ArcGrid {
            data,
            width: 6,
            height: 6,
        };
        let code = extract_transform(&input, &output);
        assert_eq!(code.kind, TransformKind::Similarity);
        if let TransformParams::Similarity { a_re, a_im, flip, .. } = code.params {
            assert!(!flip);
            assert_abs_diff_eq!(a_re, 2.0, epsilon = 1e-6);
            assert_abs_diff_eq!(a_im, 0.0, epsilon = 1e-6);
        } else {
            panic!("expected Similarity params");
        }
    }

    #[test]
    fn unrelated_grids_unknown() {
        let a = grid(&[&[1, 0, 2], &[0, 3, 0]]);
        let b = grid(&[&[1, 1, 1], &[1, 1, 1]]);
        let code = extract_transform(&a, &b);
        assert_eq!(code.kind, TransformKind::Unknown);
    }
}

/// Applies a TransformCode to a grid.
pub fn apply_transform(grid: &ArcGrid, code: &TransformCode) -> Option<ArcGrid> {
    match code.kind {
        TransformKind::Dihedral => {
            if let TransformParams::Dihedral { d4_index } = code.params {
                Some(apply_d4(grid, d4_index))
            } else {
                None
            }
        }
        TransformKind::Similarity => {
            if let TransformParams::Similarity { a_re, a_im, b_r, b_c, flip } = code.params {
                let mut out_data = vec![vec![0u8; grid.width]; grid.height];
                
                for r in 0..grid.height {
                    for c in 0..grid.width {
                        // Inverse transform: z = (z' - b) / a
                        // z' = c + i*r (target coordinates)
                        let zc = c as f64;
                        let zr = r as f64;
                        
                        if flip {
                            // z' = a * conj(z) + b  =>  conj(z) = (z' - b) / a  =>  z = conj((z' - b) / a)
                            let zc_b = zc - b_c;
                            let zr_b = zr - b_r;
                            let denom = a_re * a_re + a_im * a_im;
                            if denom.abs() < 1e-12 {
                                return None;
                            }
                            let z_re = (a_re * zc_b + a_im * zr_b) / denom;
                            let z_im = (a_re * zr_b - a_im * zc_b) / denom;
                            // conj: (re, -im) -> (re, im)
                            let src_c = z_re;
                            let src_r = -z_im;
                            
                            let src_r_i = src_r.round() as isize;
                            let src_c_i = src_c.round() as isize;
                            if src_r_i >= 0 && src_r_i < grid.height as isize &&
                               src_c_i >= 0 && src_c_i < grid.width as isize {
                                out_data[r][c] = grid.data[src_r_i as usize][src_c_i as usize];
                            }
                        } else {
                            // z' = a * z + b  =>  z = (z' - b) / a
                            let zc_b = zc - b_c;
                            let zr_b = zr - b_r;
                            let denom = a_re * a_re + a_im * a_im;
                            if denom.abs() < 1e-12 {
                                return None;
                            }
                            let z_re = (a_re * zc_b + a_im * zr_b) / denom;
                            let z_im = (a_re * zr_b - a_im * zc_b) / denom;
                            let src_c = z_re;
                            let src_r = z_im;
                            
                            let src_r_i = src_r.round() as isize;
                            let src_c_i = src_c.round() as isize;
                            if src_r_i >= 0 && src_r_i < grid.height as isize &&
                               src_c_i >= 0 && src_c_i < grid.width as isize {
                                out_data[r][c] = grid.data[src_r_i as usize][src_c_i as usize];
                            }
                        }
                    }
                }
                
                Some(ArcGrid { data: out_data, width: grid.width, height: grid.height })
            } else {
                None
            }
        }
        TransformKind::Tiling => {
            // Tiling application not implemented yet
            None
        }
        TransformKind::ColorMap => {
            // Color map application
            if let TransformParams::ColorMap { mapping } = code.params {
                let mut out_data = grid.data.clone();
                for r in 0..grid.height {
                    for c in 0..grid.width {
                        let src = grid.data[r][c];
                        if src < 10 && mapping[src as usize] != 255 {
                            out_data[r][c] = mapping[src as usize];
                        }
                    }
                }
                Some(ArcGrid { data: out_data, width: grid.width, height: grid.height })
            } else {
                None
            }
        }
        TransformKind::Unknown => None,
    }
}
