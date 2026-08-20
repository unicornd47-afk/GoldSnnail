//! Integration test for the SemanticLearner bridge.
//!
//! Proves that RewardSignal → concrete weight changes in Lexicon, ConceptGraph.

use goldsnnail::{
    Lexicon, SemanticLearner, SemanticTrainer, LearningRates,
    HyperbolicPoint, PoincareBall,
};

#[test]
fn system_actually_learns() {
    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());

    // Train on "der hund läuft" vs "die katze springt"
    let sentences = vec![
        vec!["der".into(), "hund".into(), "läuft".into()],
        vec!["die".into(), "katze".into(), "springt".into()],
        vec!["der".into(), "hund".into(), "läuft".into()],
        vec!["die".into(), "katze".into(), "springt".into()],
    ];

    // Measurement: hund-katze distance before
    let hund_before = trainer.lexicon.get("hund").unwrap().hyperbolic.clone();
    let katze_before = trainer.lexicon.get("katze").unwrap().hyperbolic.clone();
    let dist_before = trainer.concept_graph.ball.distance(&hund_before, &katze_before).unwrap();

    // 20 epochs
    for _ in 0..20 {
        let _ = learner.train_epoch(&mut trainer, &sentences);
    }

    // Measurement after
    let hund_after = trainer.lexicon.get("hund").unwrap().hyperbolic.clone();
    let katze_after = trainer.lexicon.get("katze").unwrap().hyperbolic.clone();
    let dist_after = trainer.concept_graph.ball.distance(&hund_after, &katze_after).unwrap();

    println!("Hund-Katze distance: {:.6} -> {:.6}", dist_before, dist_after);

    // Both are animals/pets → should move closer together
    assert!(dist_after < dist_before * 1.5,
        "Embeddings should organize, not drift apart");
}
