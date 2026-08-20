//! GoldWorm Semantic Training Run
//!
//! Long-term experiment with a generated corpus + generalization test.
//!
//! Usage:
//!   cargo run --example train_corpus

use goldworm::{
    PoincareBall, SemanticLearner, SemanticTrainer, LearningRates, EpochMetrics,
};
use std::fs;
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("=== GoldWorm Semantic Training Run ===\n");

    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());

    // Generate large corpus
    let corpus_size = 5_000;
    let batch_size = 100;
    let epochs = 50;

    println!("Generating corpus of {} sentences...", corpus_size);
    let corpus = trainer.composer.generate_training_batch(corpus_size);
    println!(
        "Corpus generated. Vocab size: {}\n",
        trainer.lexicon.tokens.len()
    );

    let mut history: Vec<EpochMetrics> = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let start = Instant::now();

        // One batch per epoch (stochastic)
        let batch_start = (epoch * batch_size) % corpus.len();
        let batch_end = (batch_start + batch_size).min(corpus.len());
        let batch: Vec<_> = corpus[batch_start..batch_end].to_vec();

        let metrics = learner.train_epoch(&mut trainer, &batch).unwrap();
        let duration = start.elapsed();

        history.push(metrics.clone());

        println!(
            "[Epoch {:>3}] avg_reward={:.4} | shift={:.6} | new_edges={:<3} | {:>6.2?}",
            epoch + 1,
            metrics.avg_reward,
            metrics.avg_embedding_shift,
            metrics.new_edges,
            duration
        );

        // Early stopping if reward collapsed (numerical instability)
        if metrics.avg_reward < 0.0 && epoch > 5 {
            println!("\n⚠️  Reward collapsed. Stopping early.");
            break;
        }
    }

    // === GENERALIZATION TEST ===
    println!("\n=== Generalization Test ===");

    // Test 1: Unseen combination (known words, new order)
    let novel = vec!["die".into(), "katze".into(), "sieht".into()];
    let reward_novel = trainer.train_step(&novel, false);
    println!(
        "Novel sequence 'die katze sieht': reward={:.4}",
        reward_novel.total
    );

    // Test 2: Invalid combination (syntactically wrong)
    let invalid = vec!["läuft".into(), "der".into(), "heiß".into()];
    let reward_invalid = trainer.train_step(&invalid, false);
    println!(
        "Invalid sequence 'läuft der heiß': reward={:.4}",
        reward_invalid.total
    );

    // Test 3: Semantic distance after training
    let hund = trainer.lexicon.get("hund").unwrap().hyperbolic.clone();
    let katze = trainer.lexicon.get("katze").unwrap().hyperbolic.clone();
    let tisch = trainer.lexicon.get("tisch").unwrap().hyperbolic.clone();
    let ball = PoincareBall::new(1.0);

    let d_hund_katze = ball.distance(&hund, &katze).unwrap();
    let d_hund_tisch = ball.distance(&hund, &tisch).unwrap();

    println!(
        "\nSemantic distances after training:\n  d(hund, katze) = {:.4}\n  d(hund, tisch) = {:.4}",
        d_hund_katze, d_hund_tisch
    );

    if d_hund_katze < d_hund_tisch {
        println!("✅ Generalization: Animals cluster closer than animal-object");
    } else {
        println!("❌ No clustering learned");
    }

    // Export for HTML demo
    export_state(&trainer, "training_state.json");
    println!("\nState exported to training_state.json");
}

fn export_state(trainer: &SemanticTrainer, path: &str) {
    let mut out = String::from("{\"tokens\":[\n");

    for (i, token) in trainer.lexicon.tokens.iter().enumerate() {
        let coords: Vec<String> = token
            .hyperbolic
            .coords
            .iter()
            .map(|c| format!("{:.4}", c))
            .collect();
        out.push_str(&format!(
            "  {{\"word\":\"{}\",\"class\":\"{:?}\",\"x\":{},\"y\":{},\"salience\":{:.2}}}",
            token.surface,
            token.class,
            coords.get(0).unwrap_or(&"0".to_string()),
            coords.get(1).unwrap_or(&"0".to_string()),
            token.salience
        ));
        if i < trainer.lexicon.tokens.len() - 1 {
            out.push_str(",\n");
        }
    }
    out.push_str("\n],\"edges\":[\n");

    for (i, edge) in trainer.concept_graph.edges.iter().enumerate() {
        let from = trainer
            .concept_graph
            .nodes
            .get(edge.source)
            .map(|n| n.label.clone())
            .unwrap_or_default();
        let to = trainer
            .concept_graph
            .nodes
            .get(edge.target)
            .map(|n| n.label.clone())
            .unwrap_or_default();
        out.push_str(&format!(
            "  {{\"from\":\"{}\",\"to\":\"{}\",\"weight\":{:.3}}}",
            from, to, edge.weight
        ));
        if i < trainer.concept_graph.edges.len() - 1 {
            out.push_str(",\n");
        }
    }
    out.push_str("\n]}");

    let mut file = fs::File::create(path).unwrap();
    file.write_all(out.as_bytes()).unwrap();
}
