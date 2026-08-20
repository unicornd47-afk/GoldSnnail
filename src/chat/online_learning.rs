//! Online Learning — Dynamic vocabulary expansion during conversation
//!
//! This module allows GoldSnnail to learn new words on-the-fly:
//! - Detect unknown words in user input
//! - Infer meaning from conversation context and concept graph
//! - Add new words to the lexicon dynamically
//!
//! DOD-compliant: uses usize indices into ChatArena, no raw pointers, no unsafe.

use crate::chat::ConversationBuffer;
use crate::substrate::ChatArena;
use crate::HyperbolicPoint;
use ndarray::Array1;
use rand::Rng;

/// Errors that can occur during online learning.
#[derive(Debug, Clone)]
pub enum LearnerError {
    InvalidTrainerIndex,
    InvalidEncoderIndex,
    InvalidDecoderIndex,
}

impl std::fmt::Display for LearnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LearnerError::InvalidTrainerIndex => write!(f, "invalid trainer index"),
            LearnerError::InvalidEncoderIndex => write!(f, "invalid encoder index"),
            LearnerError::InvalidDecoderIndex => write!(f, "invalid decoder index"),
        }
    }
}

impl std::error::Error for LearnerError {}

/// Manages dynamic vocabulary expansion during conversation.
///
/// Uses `usize` indices into a `ChatArena` instead of raw pointers.
/// This enables SIMD-parallelization and eliminates unsafe code.
pub struct OnlineLearner {
    pub trainer_idx: usize,
    pub encoder_idx: usize,
    pub decoder_idx: usize,
    pub min_confidence: f64,
}

impl OnlineLearner {
    /// Creates a new learner pointing to objects in a ChatArena.
    pub fn new(trainer_idx: usize, encoder_idx: usize, decoder_idx: usize) -> Self {
        Self {
            trainer_idx,
            encoder_idx,
            decoder_idx,
            min_confidence: 0.6,
        }
    }

    /// Attempt to learn unknown words from conversation context.
    /// Returns the list of successfully learned words.
    pub fn process_unknown(
        &self,
        arena: &mut ChatArena,
        unknown_words: &[String],
        context: &ConversationBuffer,
    ) -> Result<Vec<String>, LearnerError> {
        let mut learned = Vec::new();

        if unknown_words.is_empty() {
            return Ok(learned);
        }

        let last_turn = match context.last_user_turn() {
            Some(t) => t,
            None => return Ok(learned),
        };

        let context_tokens: Vec<String> = last_turn.text.split_whitespace()
            .map(|s| s.to_string())
            .collect();

        if context_tokens.is_empty() {
            return Ok(learned);
        }

        // Phase 1: Read-only access to trainer for concept resolution
        let trainer = arena.trainer_mut(self.trainer_idx)
            .ok_or(LearnerError::InvalidTrainerIndex)?;
        let resolved: Vec<_> = trainer.composer.resolve(&context_tokens);
        if resolved.is_empty() {
            return Ok(learned);
        }

        let seed_token = &resolved[0];
        let neighbors = match trainer.concept_graph.nearest_neighbors(&seed_token.hyperbolic, 5) {
            Ok(n) => n,
            Err(_) => return Ok(learned),
        };

        if neighbors.is_empty() {
            return Ok(learned);
        }

        let (best_id, best_dist) = neighbors.iter()
            .find(|(id, _)| {
                trainer.concept_graph.nodes.get(*id)
                    .map(|n| n.label != seed_token.surface)
                    .unwrap_or(true)
            })
            .copied()
            .unwrap_or(neighbors[0]);

        if best_dist > 1.0 - self.min_confidence {
            return Ok(learned);
        }

        let neighbor_node = match trainer.concept_graph.nodes.get(best_id) {
            Some(n) => n,
            None => return Ok(learned),
        };

        let neighbor_lex = match trainer.lexicon.get(&neighbor_node.label) {
            Some(l) => l,
            None => return Ok(learned),
        };

        let neighbor_class = neighbor_lex.class;
        let neighbor_embedding = neighbor_lex.embedding;
        let neighbor_coords = neighbor_lex.hyperbolic.coords.clone();

        let mut rng = rand::thread_rng();

        // Phase 2: Mutate lexicon (trainer) — scoped to release borrow before encoder/decoder
        let learned_pairs: Vec<(String, usize)> = {
            let trainer = arena.trainer_mut(self.trainer_idx)
                .ok_or(LearnerError::InvalidTrainerIndex)?;
            let mut pairs = Vec::new();
            for word in unknown_words {
                let noise_vec: Vec<f64> = (0..2).map(|_| (rng.r#gen::<f64>() - 0.5) * 0.05).collect();
                let mut new_coords = neighbor_coords.clone();
                for (i, n) in noise_vec.iter().enumerate() {
                    new_coords[i] += n;
                }
                let hp = HyperbolicPoint::new(Array1::from_vec(new_coords)).unwrap_or_else(|_| {
                    let clamped: Vec<f64> = neighbor_coords.iter().map(|&c| c * 0.99).collect();
                    HyperbolicPoint::new(Array1::from_vec(clamped)).unwrap()
                });

                let id = trainer.lexicon.tokens.len();
                let q = neighbor_embedding;
                trainer.lexicon.tokens.push(crate::LexiconToken {
                    id,
                    surface: word.clone(),
                    class: neighbor_class,
                    embedding: q,
                    hyperbolic: hp,
                    salience: 0.3,
                });
                trainer.lexicon.word_index.insert(word.clone(), id);
                trainer.lexicon.class_index.entry(neighbor_class).or_default().push(id);
                pairs.push((word.clone(), id));
            }
            pairs
        };

        // Phase 3: Register new words with encoder and decoder
        let encoder = &mut arena.encoders[self.encoder_idx];
        let decoder = &mut arena.decoders[self.decoder_idx];

        for (word, id) in learned_pairs {
            encoder.register_word(word.clone(), id);
            decoder.register_word(word.clone(), id);
            learned.push(word);
        }

        Ok(learned)
    }

    /// Suggest what an unknown word might mean based on context, without modifying the lexicon.
    pub fn suggest_meaning(&self, arena: &ChatArena, _word: &str, context: &ConversationBuffer) -> Option<String> {
        let trainer = arena.trainers.get(self.trainer_idx)?;
        let last_turn = context.last_user_turn()?;
        let context_tokens: Vec<String> = last_turn.text.split_whitespace()
            .map(|s| s.to_string())
            .collect();

        if context_tokens.is_empty() {
            return None;
        }

        let resolved: Vec<_> = trainer.composer.resolve(&context_tokens);
        if resolved.is_empty() {
            return None;
        }

        let seed_token = &resolved[0];
        let neighbors = match trainer.concept_graph.nearest_neighbors(&seed_token.hyperbolic, 5) {
            Ok(n) => n,
            Err(_) => return None,
        };

        if neighbors.is_empty() {
            return None;
        }

        let (best_id, best_dist) = neighbors.iter()
            .find(|(id, _)| {
                trainer.concept_graph.nodes.get(*id)
                    .map(|n| n.label != seed_token.surface)
                    .unwrap_or(true)
            })
            .copied()
            .unwrap_or(neighbors[0]);
        if best_dist > 1.0 - self.min_confidence {
            return None;
        }

        let neighbor_node = trainer.concept_graph.nodes.get(best_id)?;
        Some(neighbor_node.label.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ConversationTurn;
    use crate::semantics::SemanticTrainer;
    use crate::chat::spike_token_bridge::{TokenSpikeEncoder, SpikeTokenDecoder};

    #[test]
    fn new_creates_learner() {
        let learner = OnlineLearner::new(0, 0, 0);
        assert_eq!(learner.trainer_idx, 0);
    }

    #[test]
    fn suggest_meaning_returns_none_for_unknown_context() {
        let arena = ChatArena::new();
        let learner = OnlineLearner::new(0, 0, 0);
        let conv = ConversationBuffer::new(10);
        let result = learner.suggest_meaning(&arena, "unknown", &conv);
        assert!(result.is_none());
    }

    #[test]
    fn process_unknown_adds_words() {
        let mut trainer = SemanticTrainer::new(1.0);

        for token in &trainer.lexicon.tokens.clone() {
            trainer.concept_graph.add_concept(&token.surface, token.hyperbolic.clone());
        }

        let mut encoder = TokenSpikeEncoder::new(1.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);
        encoder.register_lexicon(&trainer.lexicon);
        decoder.register_lexicon(&trainer.lexicon);

        let mut arena = ChatArena::new();
        let trainer_idx = arena.push(trainer, encoder, decoder);
        let mut learner = OnlineLearner::new(trainer_idx, trainer_idx, trainer_idx);
        let mut conv = ConversationBuffer::new(10);
        conv.push(ConversationTurn::new_user("hund läuft".to_string()));

        let learned = learner.process_unknown(&mut arena, &["neuwort".to_string()], &conv).unwrap();
        assert_eq!(learned.len(), 1);
        assert_eq!(learned[0], "neuwort");
        assert!(arena.encoders[trainer_idx].neuron_for_word("neuwort").is_some());
    }

    #[test]
    fn process_unknown_respects_min_confidence() {
        let mut trainer = SemanticTrainer::new(1.0);

        for token in &trainer.lexicon.tokens.clone() {
            trainer.concept_graph.add_concept(&token.surface, token.hyperbolic.clone());
        }

        let mut encoder = TokenSpikeEncoder::new(1.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);
        encoder.register_lexicon(&trainer.lexicon);
        decoder.register_lexicon(&trainer.lexicon);

        let mut arena = ChatArena::new();
        let trainer_idx = arena.push(trainer, encoder, decoder);
        let mut learner = OnlineLearner::new(trainer_idx, trainer_idx, trainer_idx);
        learner.min_confidence = 0.99;
        let mut conv = ConversationBuffer::new(10);
        conv.push(ConversationTurn::new_user("hund läuft".to_string()));

        let learned = learner.process_unknown(&mut arena, &["neuwort".to_string()], &conv).unwrap();
        assert_eq!(learned.len(), 0);
    }
}
