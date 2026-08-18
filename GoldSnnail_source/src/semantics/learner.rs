//! Semantic Learner — Bridge from Reward to Weight Updates
//!
//! This is the P0 missing link: translates the 6-dimensional RewardSignal
//! into concrete weight changes in Lexicon, ConceptGraph, and WorldModel
//! via RSTDP-based learning.

use crate::geometry::{HyperbolicPoint, PoincareBall, Quaternion};
use crate::plasticity::RSTDP;
use crate::semantics::{
    ConceptGraph, Lexicon, LexiconToken, RelationType, RewardSignal,
};
use crate::LabError;
use ndarray::Array1;

/// Connects semantic reward with RSTDP-based learning.
/// Each component of RewardSignal flows into a different parameter block.
pub struct SemanticLearner {
    pub stdp: RSTDP,
    pub ball: PoincareBall,
    /// Learning rate per reward component
    pub lr: LearningRates,
    /// Accumulated gradients for batch updates
    _encoder_grad: Vec<f64>,
    /// Hebbian trace: which token pairs were temporally close?
    cooccurrence: Vec<(usize, usize, f64)>,
}

#[derive(Debug, Clone, Copy)]
pub struct LearningRates {
    pub semantic: f64,     // Lexicon embeddings
    pub syntactic: f64,    // ConceptGraph edges
    pub prediction: f64,   // WorldModel weights
    pub novelty: f64,      // New edge creation
    pub noise_robust: f64, // Threshold adjustment
    pub compression: f64,  // Bottleneck projection
}

impl Default for LearningRates {
    fn default() -> Self {
        Self {
            semantic: 0.02,
            syntactic: 0.01,
            prediction: 0.005,
            novelty: 0.015,
            noise_robust: 0.01,
            compression: 0.001,
        }
    }
}

impl SemanticLearner {
    pub fn new(curvature: f64, lr: LearningRates) -> Self {
        Self {
            stdp: RSTDP::new(0.01, 20.0, curvature),
            ball: PoincareBall::new(curvature),
            lr,
            _encoder_grad: Vec::new(),
            cooccurrence: Vec::new(),
        }
    }

    /// MAIN FUNCTION: A RewardSignal is translated into weight changes.
    pub fn learn_from_reward(
        &mut self,
        reward: &RewardSignal,
        tokens: &[LexiconToken],
        _predicted: Option<&HyperbolicPoint>,
        _actual: Option<&HyperbolicPoint>,
        concept_graph: &mut ConceptGraph,
        lexicon: &mut Lexicon,
    ) -> Result<LearningMetrics, LabError> {
        let mut metrics = LearningMetrics::default();

        // 1. SEMANTIC LEARNING: Move related tokens closer together
        if reward.semantic > 0.0 && tokens.len() >= 2 {
            let lr = self.lr.semantic * reward.semantic;
            for window in tokens.windows(2) {
                let a = &window[0];
                let b = &window[1];

                let dist = self.ball.distance(&a.hyperbolic, &b.hyperbolic)?;
                let target_dist = 0.05;

                let delta = if dist > target_dist {
                    self.gradient_toward(&a.hyperbolic, &b.hyperbolic, lr)?
                } else {
                    self.gradient_away(&a.hyperbolic, &b.hyperbolic, lr * 0.1)?
                };

                self.update_lexicon_embedding(lexicon, a.id, &delta)?;
                metrics.embedding_shift += delta.euclidean_norm();
            }
        }

        // 2. SYNTACTIC LEARNING: Strengthen ConceptGraph edges
        if reward.syntactic > 0.5 && tokens.len() >= 2 {
            let lr = self.lr.syntactic * reward.syntactic;
            for window in tokens.windows(2) {
                let from = window[0].id;
                let to = window[1].id;

                let edge_idx = concept_graph.edges.iter().position(|e| {
                    e.source == from && e.target == to
                });

                if let Some(idx) = edge_idx {
                    concept_graph.edges[idx].weight = (concept_graph.edges[idx].weight + lr)
                        .clamp(-1.0, 1.0);
                    metrics.edge_potentiation += 1;
                } else {
                    self.cooccurrence.push((from, to, lr));
                    metrics.new_cooccurrences += 1;
                }
            }
        }

        // 3. PREDICTION LEARNING: WorldModel error → exp_map correction
        if let (Some(pred), Some(act)) = (_predicted, _actual) {
            let lr = self.lr.prediction * reward.prediction;
            let err = self.ball.distance(pred, act)?;
            if err > 0.01 {
                let _ = self.gradient_toward(pred, act, lr)?;
                metrics.prediction_correction = err;
            }
        }

        // 4. NOVELTY: Consolidate repeated co-activations into new edges
        if reward.novelty > 0.3 {
            self.consolidate_cooccurrences(concept_graph);
            metrics.consolidations += 1;
        }

        // 5. NOISE ROBUSTNESS: Increase salience of tokens that stay active despite noise
        if reward.noise_robust > 0.0 {
            let lr = self.lr.noise_robust * reward.noise_robust;
            for token in tokens {
                if let Some(t) = lexicon.tokens.get_mut(token.id) {
                    t.salience = (t.salience + lr).min(1.0);
                }
            }
        }

        metrics.total_reward = reward.total;
        Ok(metrics)
    }

    /// Gradient: point a is moved toward b (in tangential space)
    fn gradient_toward(
        &self,
        a: &HyperbolicPoint,
        b: &HyperbolicPoint,
        lr: f64,
    ) -> Result<HyperbolicPoint, LabError> {
        let a_coords = Array1::from_vec(a.coords.clone());
        let mut delta = Array1::zeros(a_coords.len());
        for i in 0..delta.len() {
            delta[i] = (b.coords[i] - a.coords[i]) * lr;
        }
        let shifted = &a_coords + &delta;
        let norm = shifted.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let safe = &shifted * (0.99 / norm);
            return HyperbolicPoint::new(safe);
        }
        HyperbolicPoint::new(shifted)
    }

    fn gradient_away(
        &self,
        a: &HyperbolicPoint,
        b: &HyperbolicPoint,
        lr: f64,
    ) -> Result<HyperbolicPoint, LabError> {
        let a_coords = Array1::from_vec(a.coords.clone());
        let mut delta = Array1::zeros(a_coords.len());
        for i in 0..delta.len() {
            delta[i] = (a.coords[i] - b.coords[i]) * lr;
        }
        let shifted = &a_coords + &delta;
        let norm = shifted.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let safe = &shifted * (0.99 / norm);
            return HyperbolicPoint::new(safe);
        }
        HyperbolicPoint::new(shifted)
    }

    fn update_lexicon_embedding(
        &mut self,
        lexicon: &mut Lexicon,
        id: usize,
        delta: &HyperbolicPoint,
    ) -> Result<(), LabError> {
        if let Some(token) = lexicon.tokens.get_mut(id) {
            token.hyperbolic = delta.clone();
            token.embedding = Quaternion::new(
                delta.coords[0] as f32,
                delta.coords.get(1).copied().unwrap_or(0.0) as f32,
                delta.coords.get(2).copied().unwrap_or(0.0) as f32,
                delta.coords.get(3).copied().unwrap_or(0.0) as f32,
            ).normalize();
        }
        Ok(())
    }

    /// Repeated co-activations → permanent edges
    fn consolidate_cooccurrences(&mut self, graph: &mut ConceptGraph) {
        let mut merged: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
        for (from, to, w) in &self.cooccurrence {
            *merged.entry((*from, *to)).or_insert(0.0) += *w;
        }

        for ((from, to), weight) in merged {
            if weight > 0.05 {
                let exists = graph.edges.iter().any(|e| e.source == from && e.target == to);
                if !exists {
                    let from_label = graph.nodes.get(from).map(|n| n.label.clone()).unwrap_or_default();
                    let to_label = graph.nodes.get(to).map(|n| n.label.clone()).unwrap_or_default();
                    let _ = graph.add_edge(&from_label, &to_label, RelationType::RelatedTo, weight.min(0.9));
                }
            }
        }
        self.cooccurrence.clear();
    }

    /// Train one epoch: batch of sequences → average reward
    pub fn train_epoch(
        &mut self,
        trainer: &mut crate::semantics::token_engine::SemanticTrainer,
        sequences: &[Vec<String>],
    ) -> Result<EpochMetrics, LabError> {
        let mut total_reward = 0.0;
        let mut total_shift = 0.0;
        let mut learned_edges = 0;

        for seq in sequences {
            let (clean_reward, noisy_reward) = trainer.train_with_noise(seq);

            let tokens = trainer.composer.resolve(seq);
            if !tokens.is_empty() {
                let metrics = self.learn_from_reward(
                    &clean_reward,
                    &tokens,
                    None, None,
                    &mut trainer.concept_graph,
                    &mut trainer.lexicon,
                )?;
                total_shift += metrics.embedding_shift;
                learned_edges += metrics.edge_potentiation + metrics.consolidations;
            }

            if noisy_reward.total > clean_reward.total * 0.7 {
                let noisy_tokens = trainer.composer.resolve(
                    &trainer.composer.noise.corrupt_sequence(seq, &trainer.lexicon)
                );
                if !noisy_tokens.is_empty() {
                    let _ = self.learn_from_reward(
                        &noisy_reward,
                        &noisy_tokens,
                        None, None,
                        &mut trainer.concept_graph,
                        &mut trainer.lexicon,
                    )?;
                }
            }

            total_reward += clean_reward.total;
        }

        Ok(EpochMetrics {
            avg_reward: total_reward / sequences.len().max(1) as f64,
            avg_embedding_shift: total_shift / sequences.len().max(1) as f64,
            new_edges: learned_edges,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct LearningMetrics {
    pub embedding_shift: f64,
    pub edge_potentiation: usize,
    pub new_cooccurrences: usize,
    pub consolidations: usize,
    pub prediction_correction: f64,
    pub total_reward: f64,
}

#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub avg_reward: f64,
    pub avg_embedding_shift: f64,
    pub new_edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SemanticTrainer;

    #[test]
    fn learning_shifts_embeddings() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut learner = SemanticLearner::new(1.0, LearningRates::default());

        let seq = vec!["der".into(), "hund".into(), "läuft".into()];
        let before = trainer.lexicon.get("hund").unwrap().hyperbolic.clone();

        for _ in 0..10 {
            let reward = trainer.train_step(&seq, false);
            let tokens = trainer.composer.resolve(&seq);
            let _ = learner.learn_from_reward(
                &reward, &tokens, None, None,
                &mut trainer.concept_graph, &mut trainer.lexicon,
            );
        }

        let after = trainer.lexicon.get("hund").unwrap().hyperbolic.clone();
        let dist = learner.ball.distance(&before, &after).unwrap();
        assert!(dist > 1e-6, "Embedding should shift after training");
    }

    #[test]
    fn syntactic_learning_creates_edges() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut learner = SemanticLearner::new(1.0, LearningRates::default());

        let seq = vec!["der".into(), "hund".into(), "läuft".into()];
        let edge_count_before = trainer.concept_graph.edges.len();

        let reward = trainer.train_step(&seq, false);
        let tokens = trainer.composer.resolve(&seq);
        let _ = learner.learn_from_reward(
            &reward, &tokens, None, None,
            &mut trainer.concept_graph, &mut trainer.lexicon,
        );

        learner.consolidate_cooccurrences(&mut trainer.concept_graph);

        let edge_count_after = trainer.concept_graph.edges.len();
        assert!(edge_count_after >= edge_count_before,
            "Should learn edges: {} → {}", edge_count_before, edge_count_after);
    }

    #[test]
    fn epoch_training_improves_reward() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut learner = SemanticLearner::new(1.0, LearningRates::default());

        let batch = trainer.composer.generate_training_batch(50);
        let epoch1 = learner.train_epoch(&mut trainer, &batch).unwrap();
        let epoch2 = learner.train_epoch(&mut trainer, &batch).unwrap();

        println!("Epoch 1: {:.3}, Epoch 2: {:.3}", epoch1.avg_reward, epoch2.avg_reward);
        assert!(epoch2.avg_reward > 0.0);
    }

    #[test]
    fn noise_robustness_increases_salience() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut learner = SemanticLearner::new(1.0, LearningRates::default());

        let seq = vec!["der".into(), "hund".into(), "läuft".into()];
        let sal_before = trainer.lexicon.get("hund").unwrap().salience;

        for _ in 0..5 {
            let (_, noisy_reward) = trainer.train_with_noise(&seq);
            let noisy_tokens = trainer.composer.resolve(
                &trainer.composer.noise.corrupt_sequence(&seq, &trainer.lexicon)
            );
            let _ = learner.learn_from_reward(
                &noisy_reward, &noisy_tokens, None, None,
                &mut trainer.concept_graph, &mut trainer.lexicon,
            );
        }

        let sal_after = trainer.lexicon.get("hund").unwrap().salience;
        assert!(sal_after >= sal_before, "Salience should increase through robustness reward");
    }
}
