//! World Chat — World Model integration for predictive conversation
//!
//! Uses the hyperbolic World Model to predict next states from conversation
//! context, enabling forward-thinking response generation.

use crate::chat::config::WorldGeometry;
use crate::chat::{ConversationBuffer};
use crate::semantics::{SemanticTrainer, LexiconToken};
use crate::chat::spike_token_bridge::TokenSpikeEncoder;
use crate::world_model::WorldModel;
use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::LabError;
use ndarray::Array1;

/// Integrates WorldModel with chat for predictive response generation.
pub struct WorldChat {
    pub world_model: WorldModel,
    pub ball: PoincareBall,
}

impl WorldChat {
    pub fn new(latent_dim: usize, hidden_dim: usize, curvature: f64) -> Self {
        let world_model = WorldModel::new(latent_dim, hidden_dim, curvature);
        let ball = PoincareBall::new(curvature);
        Self { world_model, ball }
    }

    pub fn from_config(geom: WorldGeometry) -> Self {
        Self::new(geom.latent_dim, geom.hidden_dim, geom.curvature as f64)
    }

    /// Encode a sentence into a single hyperbolic state by averaging token embeddings.
    pub fn encode_sentence_state(&self, trainer: &SemanticTrainer, sentence: &[String]) -> HyperbolicPoint {
        if sentence.is_empty() {
            return HyperbolicPoint::new(Array1::from(vec![0.0; trainer.lexicon.tokens[0].hyperbolic.coords.len()])).unwrap();
        }
        
        let dim = trainer.lexicon.tokens[0].hyperbolic.coords.len();
        let mut sum = vec![0.0f64; dim];
        let mut count = 0usize;
        
        for word in sentence {
            if let Some(token) = trainer.lexicon.get(word) {
                for (i, &v) in token.hyperbolic.coords.iter().enumerate() {
                    if i < dim {
                        sum[i] += v;
                    }
                }
                count += 1;
            }
        }
        
        if count == 0 {
            return HyperbolicPoint::new(Array1::from(vec![0.0; dim])).unwrap();
        }
        
        let avg: Vec<f64> = sum.iter().map(|&v| v / count as f64).collect();
        HyperbolicPoint::new(Array1::from(avg)).unwrap_or_else(|_| HyperbolicPoint::new(Array1::from(vec![0.0; dim])).unwrap())
    }

    /// Predict the next conversational state from current input.
    pub fn predict_next(&mut self, current: &HyperbolicPoint) -> Result<HyperbolicPoint, LabError> {
        self.world_model.predict(current)
    }

    /// Find the lexicon token closest to a predicted state.
    pub fn closest_token<'a>(&self, trainer: &'a SemanticTrainer, predicted: &HyperbolicPoint) -> Option<&'a LexiconToken> {
        let mut best_token = None;
        let mut best_dist = f64::INFINITY;
        
        for token in &trainer.lexicon.tokens {
            if let Ok(dist) = self.ball.distance(predicted, &token.hyperbolic) {
                if dist < best_dist {
                    best_dist = dist;
                    best_token = Some(token);
                }
            }
        }
        
        best_token
    }

    /// Generate a prediction-based response word from conversation context.
    pub fn predict_response_word(&mut self, trainer: &SemanticTrainer, input: &[String]) -> Option<String> {
        let current = self.encode_sentence_state(trainer, input);
        let predicted = self.predict_next(&current).ok()?;
        let token = self.closest_token(trainer, &predicted)?;
        Some(token.surface.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn new_creates_world_chat() {
        let wc = WorldChat::new(4, 8, 1.0);
        assert_eq!(wc.world_model.latent_dim, 4);
    }

    #[test]
    fn encode_sentence_state_averages_tokens() {
        let trainer = SemanticTrainer::new(1.0);
        
        let wc = WorldChat::new(2, 4, 1.0);
        let state = wc.encode_sentence_state(&trainer, &["hund".to_string(), "katze".to_string()]);
        assert_eq!(state.coords.len(), 2);
    }

    #[test]
    fn predict_next_returns_hyperbolic_point() {
        let mut wc = WorldChat::new(2, 4, 1.0);
        let current = HyperbolicPoint::new(array![0.1, 0.2]).unwrap();
        let predicted = wc.predict_next(&current).unwrap();
        assert!(predicted.euclidean_norm() < 1.0, "Predicted state must stay inside ball");
    }

    #[test]
    fn closest_token_finds_nearest() {
        let trainer = SemanticTrainer::new(1.0);
        
        let wc = WorldChat::new(2, 4, 1.0);
        let hund_token = trainer.lexicon.get("hund").unwrap();
        let token = wc.closest_token(&trainer, &hund_token.hyperbolic).unwrap();
        assert_eq!(token.surface, "hund");
    }
}
