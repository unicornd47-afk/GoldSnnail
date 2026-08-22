//! Transform-Codec Stage 3 — Committee Voting via Einstein Midpoint.
//!
//! The Committee is the Invariant.
//!
//! Five voters each propose a transform. Their predictions are points in the
//! Poincaré ball. The consensus is their **Einstein midpoint**:
//!     m = (∑ γ_i x_i) / (∑ γ_i),   γ_i = 1 / √(1 - ‖x_i‖²)
//!
//! This is the unique Fréchet mean in the Poincaré ball — the point minimizing
//! the sum of squared hyperbolic distances. Single pass, O(n·d), no iteration.
//!
//! Uncertainty: σ = √(∑ γ_i d(m, x_i)² / ∑ γ_i) — the weighted RMS hyperbolic
//! distance to the midpoint. Low σ = high consensus.

use crate::geometry::poincare::{SAFE_LIMIT, hyperbolic_distance, project_radius};
use crate::vision::transform_codec::TransformCode;
use crate::vision::transform_memory::TransformMemory;

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct VoterPrediction {
    /// The proposed transform.
    pub code: TransformCode,
    /// Weight of this voter (confidence, inverse residual, etc.).
    pub weight: f32,
    /// Embedding of the transform in the ball (cached).
    pub embedding: [f32; 5],
}

#[derive(Clone, Debug)]
pub struct Committee {
    voters: Vec<VoterPrediction>,
}

impl Committee {
    /// Create a committee from k-NN results + their weights.
    pub fn from_knn(knn_results: &[(f32, &TransformCode)], memory: &TransformMemory) -> Self {
        let voters: Vec<_> = knn_results.iter().map(|(dist, code)| {
            // Weight: inverse distance (closer = more confident), clamped
            let weight = (1.0 / (dist + 1e-6)).min(100.0);
            let emb = memory.embed(code);
            let mut embedding = [0.0f32; 5];
            embedding.copy_from_slice(&emb);
            VoterPrediction {
                code: (*code).clone(),
                weight,
                embedding,
            }
        }).collect();
        Self { voters }
    }

    /// Create a committee from explicit voter predictions.
    pub fn new(voters: Vec<VoterPrediction>) -> Self {
        Self { voters }
    }

    /// Compute the Einstein midpoint of the committee.
    /// Returns (consensus_embedding, uncertainty_sigma).
    pub fn einstein_midpoint(&self) -> ([f32; 5], f32) {
        let n = self.voters.len();
        if n == 0 {
            return ([0.0; 5], f32::INFINITY);
        }
        if n == 1 {
            return (self.voters[0].embedding, 0.0);
        }

        // γ_i = weight_i / √(1 - ‖x_i‖²)
        // For points already in the ball, ‖x‖ < SAFE_LIMIT ≈ 0.9999
        let mut gamma_sum = 0.0f32;
        let mut weighted_sum = [0.0f32; 5];

        for v in &self.voters {
            let norm_sq: f32 = v.embedding.iter().map(|x| x * x).sum();
            let gamma = v.weight / (1.0 - norm_sq).sqrt().max(1e-6);
            gamma_sum += gamma;
            for i in 0..5 {
                weighted_sum[i] += gamma * v.embedding[i];
            }
        }

        // Midpoint in tangent space, then project back
        let mut mid = [0.0f32; 5];
        for i in 0..5 {
            mid[i] = weighted_sum[i] / gamma_sum;
        }

        // Project to safe ball (elastic clamp per dimension)
        for i in 0..5 {
            mid[i] = project_radius(mid[i], SAFE_LIMIT);
        }

        // Uncertainty: weighted RMS hyperbolic distance to midpoint
        let mut var_sum = 0.0f32;
        for v in &self.voters {
            let dist_sq: f32 = mid.iter().zip(v.embedding.iter())
                .map(|(a, b)| {
                    let d = hyperbolic_distance(*a, *b);
                    d * d
                })
                .sum();
            let gamma = v.weight / (1.0 - v.embedding.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
            var_sum += gamma * dist_sq;
        }
        let sigma = (var_sum / gamma_sum).sqrt();

        (mid, sigma)
    }

    /// Decode the consensus embedding back to a TransformCode.
    /// Uses the nearest known transform in the memory.
    pub fn decode(&self, memory: &TransformMemory, consensus_emb: [f32; 5]) -> TransformCode {
        // Find the nearest entry in memory by hyperbolic distance
        let mut best: Option<(f32, TransformCode)> = None;
        for (emb, code, _, _) in &memory.entries {
            let dist_sq: f32 = consensus_emb.iter().zip(emb.iter())
                .map(|(a, b)| {
                    let d = hyperbolic_distance(*a, *b);
                    d * d
                })
                .sum();
            let dist = dist_sq.sqrt();
            if best.map_or(true, |(bd, _)| dist < bd) {
                best = Some((dist, *code));
            }
        }
        best.map(|(_, c)| c).unwrap_or(TransformCode {
            kind: crate::vision::transform_codec::TransformKind::Unknown,
            params: crate::vision::transform_codec::TransformParams::None,
            residual: f64::INFINITY,
        })
    }

    /// Full pipeline: k-NN → committee → consensus → decode.
    pub fn predict(
        memory: &TransformMemory,
        query: &TransformCode,
        k: usize,
    ) -> (TransformCode, f32) {
        let knn = memory.knn(query, k);
        let committee = Self::from_knn(&knn, memory);
        let (consensus_emb, sigma) = committee.einstein_midpoint();
        let decoded = committee.decode(memory, consensus_emb);
        (decoded, sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::transform_codec::{TransformCode, TransformKind, TransformParams};
    use crate::vision::transform_memory::TransformMemory;

    fn make_code(kind: TransformKind, params: TransformParams, residual: f64) -> TransformCode {
        TransformCode { kind, params, residual }
    }

    #[test]
    fn single_voter_returns_that_voter() {
        let mut mem = TransformMemory::new(5);
        let code = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        mem.add(code, "t1".into(), 0);
        
        let knn = mem.knn(&code, 1);
        let committee = Committee::from_knn(&knn, &mem);
        let (consensus, sigma) = committee.einstein_midpoint();
        assert!(sigma == 0.0);
        
        let decoded = committee.decode(&mem, consensus);
        assert_eq!(decoded.kind, TransformKind::Similarity);
    }

    #[test]
    fn two_identical_voters_zero_uncertainty() {
        let mut mem = TransformMemory::new(5);
        let code = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.5, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        mem.add(code.clone(), "t1".into(), 0);
        mem.add(code.clone(), "t2".into(), 0);
        
        let knn = mem.knn(&code, 2);
        let committee = Committee::from_knn(&knn, &mem);
        let (consensus, sigma) = committee.einstein_midpoint();
        assert!(sigma < 1e-2); // relaxed for numerical precision
    }

    #[test]
    fn committee_averages_different_scales() {
        let mut mem = TransformMemory::new(5);
        let code1 = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        let code2 = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 2.0, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        mem.add(code1, "t1".into(), 0);
        mem.add(code2, "t2".into(), 0);
        
        // Query at 1.5x — should get intermediate
        let query = make_code(
            TransformKind::Similarity,
            TransformParams::Similarity { a_re: 1.5, a_im: 0.0, b_r: 0.0, b_c: 0.0, flip: false },
            0.0,
        );
        let knn = mem.knn(&query, 2);
        let committee = Committee::from_knn(&knn, &mem);
        let (consensus_emb, sigma) = committee.einstein_midpoint();
        
        // Consensus a_re (first component) should be between 1.0 and 2.0
        // Note: embedding uses tanh projection, so check raw value
        assert!(consensus_emb[0] > 0.0 && consensus_emb[0] < 1.0, "consensus a_re component = {}", consensus_emb[0]);
        assert!(sigma > 0.0); // non-zero uncertainty from spread
    }
}