//! Hyperbolic Contrastive Learning
//!
//! The heart of semantic self-supervised learning:
//! - Similar meanings stay close
//! - Dissimilar meanings stay far apart
//! Uses the existing `PoincareBall` distance metric.

use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::LabError;
use ndarray::Array1;

/// Hyperbolic contrastive learning
/// Positive pairs (e.g., "dog" and "canine") → distance ↓
/// Negative pairs (e.g., "dog" and "table") → distance ↑
pub struct HyperbolicContrastive {
    pub ball: PoincareBall,
    pub margin: f64,
    pub temperature: f64,
}

impl HyperbolicContrastive {
    pub fn new(curvature: f64, margin: f64, temperature: f64) -> Self {
        Self {
            ball: PoincareBall::new(curvature),
            margin,
            temperature,
        }
    }

    /// Loss for a positive pair (a, b) and a negative pair (a, c)
    /// L = max(0, d(a,b)² - d(a,c)² + margin)
    pub fn triplet_loss(
        &self,
        anchor: &HyperbolicPoint,
        positive: &HyperbolicPoint,
        negative: &HyperbolicPoint,
    ) -> Result<f64, LabError> {
        let d_pos = self.ball.distance(anchor, positive)?;
        let d_neg = self.ball.distance(anchor, negative)?;

        let loss = (d_pos.powi(2) - d_neg.powi(2) + self.margin).max(0.0);
        Ok(loss)
    }

    /// Softmax-like contrastive over a batch
    /// L = -log( exp(-d(a,p)/T) / Σ exp(-d(a,n_i)/T) )
    pub fn info_nce_loss(
        &self,
        anchor: &HyperbolicPoint,
        positive: &HyperbolicPoint,
        negatives: &[HyperbolicPoint],
    ) -> Result<f64, LabError> {
        let d_pos = self.ball.distance(anchor, positive)? / self.temperature;
        let pos_score = (-d_pos).exp();

        let mut sum_neg = pos_score;
        for neg in negatives {
            let d_neg = self.ball.distance(anchor, neg)? / self.temperature;
            sum_neg += (-d_neg).exp();
        }

        Ok(-(pos_score / sum_neg).ln())
    }

    /// Gradient for anchor update (manual, no autograd)
    /// Returns direction the anchor should move
    pub fn gradient_step(
        &self,
        anchor: &HyperbolicPoint,
        positive: &HyperbolicPoint,
        negative: &HyperbolicPoint,
        lr: f64,
    ) -> Result<HyperbolicPoint, LabError> {
        let d_pos = self.ball.distance(anchor, positive)?;
        let d_neg = self.ball.distance(anchor, negative)?;

        // Direction: away from negative, toward positive
        let mut new_coords = anchor.coords.clone();
        if d_pos > d_neg {
            for i in 0..new_coords.len() {
                let toward_pos = positive.coords[i] - anchor.coords[i];
                let away_neg = anchor.coords[i] - negative.coords[i];
                new_coords[i] += lr * (toward_pos + away_neg);
            }
        }

        // Project back into ball
        let norm = new_coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            for x in &mut new_coords {
                *x *= 0.99 / norm;
            }
        }

        HyperbolicPoint::new(Array1::from(new_coords))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn triplet_loss_is_positive_when_ordering_wrong() {
        let hc = HyperbolicContrastive::new(1.0, 0.5, 1.0);
        let a = HyperbolicPoint::new(array![0.0, 0.0]).unwrap();
        let p = HyperbolicPoint::new(array![0.8, 0.0]).unwrap();
        let n = HyperbolicPoint::new(array![0.05, 0.0]).unwrap();

        let loss = hc.triplet_loss(&a, &p, &n).unwrap();
        assert!(loss > 0.0, "Loss should be positive when positive is farther than negative");
    }

    #[test]
    fn triplet_loss_is_zero_when_ordering_correct() {
        let hc = HyperbolicContrastive::new(1.0, 0.1, 1.0);
        let a = HyperbolicPoint::new(array![0.0, 0.0]).unwrap();
        let p = HyperbolicPoint::new(array![0.05, 0.0]).unwrap();
        let n = HyperbolicPoint::new(array![0.8, 0.0]).unwrap();

        let loss = hc.triplet_loss(&a, &p, &n).unwrap();
        assert_eq!(loss, 0.0, "Loss should be zero when positive is closer than negative + margin");
    }

    #[test]
    fn info_nce_decreases_with_similarity() {
        let hc = HyperbolicContrastive::new(1.0, 0.1, 0.5);
        let a = HyperbolicPoint::new(array![0.0, 0.0]).unwrap();
        let p_near = HyperbolicPoint::new(array![0.01, 0.0]).unwrap();
        let p_far = HyperbolicPoint::new(array![0.5, 0.0]).unwrap();
        let negatives = vec![
            HyperbolicPoint::new(array![0.9, 0.0]).unwrap(),
        ];

        let loss_near = hc.info_nce_loss(&a, &p_near, &negatives).unwrap();
        let loss_far = hc.info_nce_loss(&a, &p_far, &negatives).unwrap();

        assert!(loss_near < loss_far, "Nearer positive = lower loss");
    }
}
