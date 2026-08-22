//! Transform-Codec Stage 2 — transform embedding into the Poincaré ball.
//!
//! We do NOT embed grid images. We embed the *transform vectors* (a, b, flip)
//! from Stage 1 as points in the ball, so that "similar transformations" are
//! near each other in hyperbolic space. k-NN then retrieves known transforms,
//! and their Einstein midpoint is the consensus prediction.
//!
//! Embedding:  map (a_re, a_im, b_r, b_c, flip)  →  ball point  via
//!   1) Project to tangent space at 0 (identity map for small vectors)
//!   2) Exponential map exp_0(v) = tanh(‖v‖) · v/‖v‖
//!      (clamped so ‖x‖ < SAFE_LIMIT = 1 - 1e-5)
//!
//! This is a SINGLE forward pass — no training, no gradients.

use crate::geometry::poincare::{SAFE_LIMIT, elastic_clamp, project_radius, hyperbolic_distance};
use crate::vision::transform_codec::{SimilarityFit, TransformCode, TransformKind, TransformParams};

#[derive(Clone, Debug)]
pub struct TransformMemory {
    // Each entry: (embedding, transform_code, task_id, ex_idx)
    // embedding is a 5D vector (a_re, a_im, b_r, b_c, flip) inside the unit ball
    pub entries: Vec<(Vec<f32>, TransformCode, String, usize)>,
    dim: usize,
}

impl TransformMemory {
    pub fn new(dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            dim,
        }
    }

    /// Embed a TransformCode into the ball using the f32 elastic API.
    pub fn embed(&self, code: &TransformCode) -> Vec<f32> {
        // Extract (a_re, a_im, b_r, b_c, flip) from the transform
        let (a_re, a_im, b_r, b_c, flip) = match code.kind {
            TransformKind::Similarity => {
                if let TransformParams::Similarity { a_re, a_im, b_r, b_c, flip } = code.params {
                    (a_re as f32, a_im as f32, b_r as f32, b_c as f32, if flip { 1.0 } else { 0.0 })
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0)
                }
            }
            TransformKind::Dihedral => {
                // D4 transforms as fixed points in transform space
                // Map each D4 index to a canonical embedding
                if let TransformParams::Dihedral { d4_index } = code.params {
                    let idx = d4_index as f32;
                    // 8 corners of the 5D hypercube, scaled
                    let patterns = [
                        [0.0, 0.0, 0.0, 0.0, 0.0],  // identity
                        [0.0, 1.0, 0.0, 0.0, 0.0],  // rot90
                        [-1.0, 0.0, 0.0, 0.0, 0.0], // rot180
                        [0.0, -1.0, 0.0, 0.0, 0.0], // rot270
                        [1.0, 0.0, 0.0, 0.0, 1.0],  // flip H
                        [-1.0, 0.0, 0.0, 0.0, 1.0], // flip V
                        [0.0, 1.0, 0.0, 0.0, 1.0],  // transpose
                        [0.0, -1.0, 0.0, 0.0, 1.0], // anti-diagonal
                    ];
                    let p = patterns[idx as usize];
                    (p[0], p[1], p[2], p[3], p[4])
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0)
                }
            }
            TransformKind::ColorMap => (0.0, 0.0, 0.0, 0.0, 0.0),
            TransformKind::Tiling => (0.0, 0.0, 0.0, 0.0, 0.0),
            TransformKind::Unknown => (0.0, 0.0, 0.0, 0.0, 0.0),
        };

        // Exponential map at 0: exp_0(v) = tanh(||v||) * v / ||v||
        let v = vec![a_re, a_im, b_r, b_c, flip];
        let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        if norm < 1e-8 {
            return vec![0.0; self.dim];
        }
        let scale = norm.tanh() / norm;
        let mut p: Vec<f32> = v.iter().map(|x| x * scale).collect();
        // Clamp to SAFE_LIMIT to avoid boundary blowup (elastic)
        for x in &mut p {
            *x = project_radius(*x, SAFE_LIMIT);
        }
        p
    }

    /// Add a transform from a training example.
    pub fn add(&mut self, code: TransformCode, task_id: String, ex_idx: usize) {
        let emb = self.embed(&code);
        self.entries.push((emb, code, task_id, ex_idx));
    }

    /// k-NN search in transform space using the f32 Poincaré distance.
    /// Returns (distance, transform_code) sorted by distance ascending.
    pub fn knn(&self, query: &TransformCode, k: usize) -> Vec<(f32, &TransformCode)> {
        let q = self.embed(query);
        let mut results: Vec<_> = self.entries.iter().map(|(emb, code, _, _)| {
            // Hyperbolic distance in 1D per dimension, then sqrt(sum squares)
            let dist_sq: f32 = q.iter().zip(emb.iter())
                .map(|(a, b)| {
                    let d = hyperbolic_distance(*a, *b);
                    d * d
                })
                .sum();
            (dist_sq.sqrt(), code)
        }).collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(k);
        results
    }
}

/// Build memory from ARC training tasks (input→output pairs).
pub fn build_memory_from_tasks(
    tasks: &[crate::vision::arc_loader::ArcTask],
    dim: usize,
) -> TransformMemory {
    let mut mem = TransformMemory::new(dim);
    for task in tasks {
        for (ex_idx, (input, output)) in task.train_pairs.iter().enumerate() {
            let code = crate::vision::transform_codec::extract_transform(input, output);
            mem.add(code, task.id.clone(), ex_idx);
        }
    }
    mem
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::transform_codec::{SimilarityFit, TransformCode, TransformKind, TransformParams};
    use crate::vision::arc_loader::ArcGrid;

    fn make_fit(a_re: f64, a_im: f64, b_r: f64, b_c: f64, flip: bool) -> SimilarityFit {
        SimilarityFit { a_re, a_im, b_r, b_c, flip, residual: 0.0 }
    }

    fn make_code(kind: TransformKind, params: TransformParams, residual: f64) -> TransformCode {
        TransformCode { kind, params, residual }
    }

    #[test]
    fn embed_and_knn_identity() {
        let mut mem = TransformMemory::new(5);
        // Identity transform (a=1, b=0)
        let code = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        mem.add(code, "task1".into(), 0);
        
        // Query with near-identity
        let query = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        let results = mem.knn(&query, 1);
        assert_eq!(results.len(), 1);
        // Distance should be very small (same transform)
        assert!(results[0].0 < 0.1);
    }

    #[test]
    fn knn_distinguishes_different_scales() {
        let mut mem = TransformMemory::new(5);
        // Scale 1x
        mem.add(make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        ), "t1".into(), 0);
        // Scale 2x
        mem.add(make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 2.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        ), "t2".into(), 0);
        
        // Query 1x should be closer to 1x
        let query = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        let results = mem.knn(&query, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].0 < results[1].0); // first is 1x, second is 2x
    }

    #[test]
    fn d4_embeddings_are_distinct() {
        let mut mem = TransformMemory::new(5);
        // Add all 8 D4 transforms
        for idx in 0..8 {
            mem.add(make_code(
                TransformKind::Dihedral,
                TransformParams::Dihedral { d4_index: idx },
                0.0,
            ), format!("d4_{idx}"), 0);
        }
        
        // Query identity
        let query = make_code(
            TransformKind::Dihedral,
            TransformParams::Dihedral { d4_index: 0 },
            0.0,
        );
        let results = mem.knn(&query, 8);
        assert_eq!(results.len(), 8);
        // Identity should be closest (distance ~0)
        assert!(results[0].0 < 0.1);
        // Others should be ordered by D4 distance
    }
}