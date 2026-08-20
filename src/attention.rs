//! Quaternion Attention (Twistor Mechanism)
//!
//! `attention(Q, K, V) = softmax(||Q ⊗ K*||) ⊗ V`
//!
//! Operates on flat `Vec<Quaternion>`. All intermediate allocations are
//! pre-sized; no `Vec` growth in the hot path.

use crate::geometry::Quaternion;

/// Quaternion Attention head.
///
/// Computes self-attention over quaternion-valued features using the Twistor
/// mechanism: similarity is the norm of the Hamilton product `Q ⊗ K*`.
#[derive(Debug, Clone, Copy)]
pub struct QuaternionAttention;

impl QuaternionAttention {
    pub fn new() -> Self {
        Self
    }

    /// Single-head forward pass.
    ///
    /// Returns a `Vec<Quaternion>` of the same length as `queries`.
    pub fn forward(
        &self,
        queries: &[Quaternion],
        keys: &[Quaternion],
        values: &[Quaternion],
    ) -> Vec<Quaternion> {
        assert_eq!(keys.len(), values.len(), "keys and values must have the same length");

        let mut output = Vec::with_capacity(queries.len());

        for q in queries {
            // Scores: ||Q ⊗ K*|| for each key.
            let mut scores = Vec::with_capacity(keys.len());
            for k in keys {
                let score = q.mul(k.conjugate()).norm();
                scores.push(score);
            }

            // Numerically stable softmax over scores.
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            let mut weights = Vec::with_capacity(scores.len());
            for s in &scores {
                let w = (s - max_score).exp();
                weights.push(w);
                sum_exp += w;
            }
            if sum_exp < 1e-12 {
                sum_exp = 1e-12;
            }
            for w in &mut weights {
                *w /= sum_exp;
            }

            // Weighted sum of values.
            let mut out = Quaternion::new(0.0, 0.0, 0.0, 0.0);
            for (v, w) in values.iter().zip(weights.iter()) {
                out = out + (*v) * *w;
            }
            output.push(out);
        }

        output
    }

    /// In-place variant: writes into a pre-allocated `output` buffer.
    ///
    /// # Panics
    ///
    /// Panics if `output.len() != queries.len()`.
    pub fn forward_in_place(
        &self,
        queries: &[Quaternion],
        keys: &[Quaternion],
        values: &[Quaternion],
        output: &mut [Quaternion],
    ) {
        assert_eq!(keys.len(), values.len(), "keys and values must have the same length");
        assert_eq!(output.len(), queries.len(), "output buffer must match queries length");

        for (i, q) in queries.iter().enumerate() {
            let mut scores = Vec::with_capacity(keys.len());
            for k in keys {
                scores.push(q.mul(k.conjugate()).norm());
            }

            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            let mut weights = Vec::with_capacity(scores.len());
            for s in &scores {
                let w = (s - max_score).exp();
                weights.push(w);
                sum_exp += w;
            }
            if sum_exp < 1e-12 {
                sum_exp = 1e-12;
            }
            for w in &mut weights {
                *w /= sum_exp;
            }

            let mut out = Quaternion::new(0.0, 0.0, 0.0, 0.0);
            for (v, w) in values.iter().zip(weights.iter()) {
                out = out + (*v) * *w;
            }
            output[i] = out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn self_attention_preserves_norm() {
        let attn = QuaternionAttention::new();
        let q = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0).normalize()];
        let k = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0).normalize()];
        let v = vec![Quaternion::new(0.0, 1.0, 0.0, 0.0).normalize()];

        let out = attn.forward(&q, &k, &v);
        assert_relative_eq!(out[0].norm(), v[0].norm(), epsilon = 1e-6);
    }

    #[test]
    fn attention_in_place_matches_return() {
        let attn = QuaternionAttention::new();
        let q = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0)];
        let k = vec![Quaternion::new(0.5, 0.5, 0.0, 0.0), Quaternion::new(0.0, 1.0, 0.0, 0.0)];
        let v = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0), Quaternion::new(0.0, 1.0, 0.0, 0.0)];

        let out_vec = attn.forward(&q, &k, &v);
        let mut out_buf = vec![Quaternion::new(0.0, 0.0, 0.0, 0.0); 1];
        attn.forward_in_place(&q, &k, &v, &mut out_buf);

        assert_relative_eq!(out_vec[0].w, out_buf[0].w, epsilon = 1e-10);
        assert_relative_eq!(out_vec[0].x, out_buf[0].x, epsilon = 1e-10);
        assert_relative_eq!(out_vec[0].y, out_buf[0].y, epsilon = 1e-10);
        assert_relative_eq!(out_vec[0].z, out_buf[0].z, epsilon = 1e-10);
    }
}
