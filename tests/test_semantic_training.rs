//! Integration tests for the semantic training pipeline.
//!
//! Tests the full semantic stack: Lexicon → TokenComposer → SemanticRewardEngine → ConceptGraph.

use goldworm::semantics::RelationType;
use goldworm::{
    Lexicon, LexiconToken, TokenClass, NoiseInjector,
    SemanticRewardEngine, RewardWeights, RewardSignal,
    TokenComposer, SemanticTrainer,
    ConceptGraph, HyperbolicContrastive,
    WorkingMemory, WorldModel, RLAgent, StateVector,
};
use goldworm::semantics::curriculum::SemanticCurriculum;

#[test]
fn semantic_taxonomy_learning() {
    let graph = SemanticCurriculum::level1_taxonomy();
    let encoder = goldworm::SemanticEncoder::new(
        vec!["hund".into(), "katze".into(), "tier".into()],
        2,
    );
    let contrastive = HyperbolicContrastive::new(1.0, 0.1, 0.5);

    let hund = graph.nodes[graph.index["hund"]].embedding.clone();
    let katze = graph.nodes[graph.index["katze"]].embedding.clone();
    let vogel = graph.nodes[graph.index["vogel"]].embedding.clone();

    let d_hund_katze = graph.ball.distance(&hund, &katze).unwrap();
    let d_hund_vogel = graph.ball.distance(&hund, &vogel).unwrap();

    assert!(d_hund_katze < d_hund_vogel,
        "Hund and Katze (both mammals/pets) should be closer than Hund and Vogel");

    let loss = contrastive.triplet_loss(&hund, &katze, &vogel).unwrap();
    assert!(loss < 0.5, "Loss should be low for semantically related pairs");
}

#[test]
fn sentence_to_memory_pipeline() {
    let encoder = goldworm::SemanticEncoder::new(
        vec!["der".into(), "hund".into(), "läuft".into()],
        2,
    );

    let sentence = vec!["der".into(), "hund".into(), "läuft".into()];
    let spikes = encoder.encode_sequence(&sentence, 10.0);

    assert_eq!(spikes.len(), 3);
    assert_eq!(spikes[2].0, 20.0); // temporal structure preserved
}

#[test]
fn end_to_end_semantic_training() {
    let mut trainer = SemanticTrainer::new(1.0);
    let mut total_reward = 0.0;

    // Epoch 1: Clean Training
    let batch = trainer.composer.generate_training_batch(20);
    for seq in &batch {
        let r = trainer.train_step(seq, false);
        total_reward += r.total;
    }
    let avg_clean = total_reward / 20.0;
    println!("Avg clean reward: {:.3}", avg_clean);

    // Epoch 2: With noise (robustness)
    total_reward = 0.0;
    for seq in &batch {
        let (_, r_noisy) = trainer.train_with_noise(seq);
        total_reward += r_noisy.total;
    }
    let avg_noisy = total_reward / 20.0;
    println!("Avg noisy reward: {:.3}", avg_noisy);

    // System should learn to tolerate noise
    assert!(avg_noisy > 0.0);
}
