//! Semantic Encoder — Token → Quaternion → HyperbolicPoint
//!
//! Maps discrete tokens (words, symbols) to hyperbolic quaternion embeddings.
//! The 4D rotation encodes semantic features; the hyperbolic norm encodes
//! hierarchy (abstract = near boundary, concrete = near center).

use crate::geometry::{HyperbolicPoint, Quaternion};
use crate::LabError;
use ndarray::Array1;
use std::collections::HashMap;

/// Maps discrete tokens (words, symbols) to hyperbolic quaternion embeddings.
/// DOD: flat weight matrix [vocab_size × 4], no Vec<Quaternion>.
#[derive(Clone)]
pub struct SemanticEncoder {
    pub vocab: HashMap<String, usize>,
    pub embeddings: Vec<f64>, // flat: [id][w,x,y,z]
    pub latent_proj: Vec<f64>, // [latent_dim × 4]: Quaternion → HyperbolicPoint
    pub latent_dim: usize,
}

impl SemanticEncoder {
    pub fn new(vocab: Vec<String>, latent_dim: usize) -> Self {
        let vocab_size = vocab.len();
        let mut map = HashMap::with_capacity(vocab_size);
        for (i, word) in vocab.into_iter().enumerate() {
            map.insert(word, i);
        }

        // Deterministic init: sinus features (no rand needed)
        let embeddings: Vec<f64> = (0..vocab_size * 4)
            .map(|i| (i as f64 * 0.618).sin() * 0.3)
            .collect();

        let latent_proj: Vec<f64> = (0..latent_dim * 4)
            .map(|i| (i as f64 * 0.317).cos() * 0.2)
            .collect();

        Self {
            vocab: map,
            embeddings,
            latent_proj,
            latent_dim,
        }
    }

    /// Token → Quaternion (raw embedding)
    pub fn encode_token(&self, token: &str) -> Option<Quaternion> {
        let id = self.vocab.get(token)?;
        let base = id * 4;
        Some(Quaternion::new(
            self.embeddings[base] as f32,
            self.embeddings[base + 1] as f32,
            self.embeddings[base + 2] as f32,
            self.embeddings[base + 3] as f32,
        ))
    }

    /// Quaternion → HyperbolicPoint (semantic latent space)
    pub fn to_hyperbolic(&self, q: &Quaternion) -> Result<HyperbolicPoint, LabError> {
        let mut latent = vec![0.0f64; self.latent_dim];
        let comps = [q.w as f64, q.x as f64, q.y as f64, q.z as f64];
        for i in 0..self.latent_dim {
            let mut acc = 0.0;
            for j in 0..4 {
                acc += self.latent_proj[i * 4 + j] * comps[j];
            }
            latent[i] = acc.tanh() * 0.95; // safe inside ball
        }
        HyperbolicPoint::new(Array1::from(latent))
    }

    /// Full pipeline: String → HyperbolicPoint
    pub fn encode(&self, token: &str) -> Option<Result<HyperbolicPoint, LabError>> {
        let q = self.encode_token(token)?;
        Some(self.to_hyperbolic(&q))
    }

    /// Sequence → spike train (for Working Memory)
    /// Each token fires at a specific timepoint with its quaternion phase
    pub fn encode_sequence(
        &self,
        tokens: &[String],
        dt_ms: f64,
    ) -> Vec<(f64, Quaternion)> {
        tokens.iter().enumerate().filter_map(|(i, t)| {
            let q = self.encode_token(t)?;
            Some((i as f64 * dt_ms, q))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_known_token() {
        let enc = SemanticEncoder::new(
            vec!["katze".into(), "hund".into(), "tier".into()],
            2,
        );
        let q = enc.encode_token("katze").unwrap();
        assert!(q.norm() > 0.0);
    }

    #[test]
    fn hyperbolic_inside_ball() {
        let enc = SemanticEncoder::new(vec!["a".into()], 2);
        let q = enc.encode_token("a").unwrap();
        let h = enc.to_hyperbolic(&q).unwrap();
        assert!(h.euclidean_norm() < 1.0);
    }

    #[test]
    fn sequence_has_temporal_structure() {
        let enc = SemanticEncoder::new(vec!["x".into(), "y".into()], 2);
        let seq = enc.encode_sequence(
            &["x".into(), "y".into()],
            10.0,
        );
        assert_eq!(seq.len(), 2);
        assert_eq!(seq[1].0, 10.0); // time offset
    }
}
