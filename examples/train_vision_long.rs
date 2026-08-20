//! GoldSnnail Vision-Semantic Long-Run Training
//!
//! Trains a SemanticLearner on word sequences while measuring how visual
//! patches map to the evolving semantic lexicon. Exports JSON for the HTML demo.
//!
//! Usage:
//!   cargo run --example train_vision_long

use goldsnnail::{
    PatchEncoder, SemanticTrainer, SemanticLearner, LearningRates, EpochMetrics,
    SemanticEncoder, PoincareBall, generate_test_image,
};
use std::fs;
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("=== GoldSnnail Vision Long-Run v0.3.0 ===\n");

    // --- Setup ---
    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());

    // PatchEncoder with its own SemanticEncoder for visual binding
    let semantic_for_vision = SemanticEncoder::new(
        vec!["hund".into(), "katze".into(), "tisch".into(), "haus".into(),
             "vogel".into(), "fisch".into(), "stein".into(), "liebe".into()],
        2,
    );
    let encoder = PatchEncoder::new(4, 2, 1.0).with_semantic(semantic_for_vision);
    let ball = PoincareBall::new(1.0);

    // --- Dataset: (image_pattern, label) ---
    let dataset = [
        ("horizontal_stripes", "hund"),
        ("vertical_stripes", "katze"),
        ("checkerboard", "tisch"),
        ("gradient", "haus"),
        ("horizontal_stripes", "vogel"),
        ("vertical_stripes", "fisch"),
        ("checkerboard", "stein"),
        ("gradient", "liebe"),
    ];

    // --- Pre-training: measure visual-semantic distances ---
    println!("--- Pre-training distances ---");
    let mut pre_distances: Vec<(String, String, f64)> = Vec::new();
    for (pattern, label) in &dataset {
        let img = generate_test_image(pattern, 16, 16);
        let tokens = encoder.encode_image(&img, 16, 16);
        let center = &tokens[tokens.len() / 2].hyperbolic;
        let label_hp = trainer.lexicon.get(label).unwrap().hyperbolic.clone();
        let dist = ball.distance(center, &label_hp).unwrap();
        pre_distances.push((pattern.to_string(), label.to_string(), dist));
        println!("  {} → {}: {:.4}", pattern, label, dist);
    }

    // --- Generate training sequences for the semantic learner ---
    let mut sequences: Vec<Vec<String>> = Vec::new();
    for _ in 0..50 {
        for (_, label) in &dataset {
            sequences.push(vec!["der".into(), label.to_string(), "sieht".into()]);
        }
    }

    // --- Training loop ---
    let epochs = 50;
    let mut history: Vec<EpochMetrics> = Vec::with_capacity(epochs);

    println!("\n--- Training ---");
    for epoch in 0..epochs {
        let start = Instant::now();
        let mut total_reward = 0.0;

        // Manual loop so we can track per-sample reward
        for seq in &sequences {
            let reward = trainer.train_step(seq, false);
            let tokens = trainer.composer.resolve(seq);
            if !tokens.is_empty() {
                let _ = learner.learn_from_reward(
                    &reward,
                    &tokens,
                    None,
                    None,
                    &mut trainer.concept_graph,
                    &mut trainer.lexicon,
                );
            }
            total_reward += reward.total;
        }

        let metrics = EpochMetrics {
            avg_reward: total_reward / sequences.len().max(1) as f64,
            avg_embedding_shift: 0.0,
            new_edges: trainer.concept_graph.edges.len(),
        };
        history.push(metrics.clone());

        if epoch % 10 == 0 {
            println!(
                "[Epoch {:>3}] reward={:.4} | edges={:<4} | time={:.2?}",
                epoch, metrics.avg_reward, metrics.new_edges, start.elapsed()
            );
        }
    }

    // --- Post-training: measure visual-semantic distances ---
    println!("\n--- Post-training distances ---");
    let mut post_distances: Vec<(String, String, f64)> = Vec::new();
    for (pattern, label) in &dataset {
        let img = generate_test_image(pattern, 16, 16);
        let tokens = encoder.encode_image(&img, 16, 16);
        let center = &tokens[tokens.len() / 2].hyperbolic;
        let label_hp = trainer.lexicon.get(label).unwrap().hyperbolic.clone();
        let dist = ball.distance(center, &label_hp).unwrap();
        post_distances.push((pattern.to_string(), label.to_string(), dist));
        println!("  {} → {}: {:.4}", pattern, label, dist);
    }

    // --- Generalization test ---
    println!("\n=== Generalization Test ===");
    let test_patterns = [
        ("horizontal_stripes", "hund"),
        ("vertical_stripes", "katze"),
        ("checkerboard", "tisch"),
        ("gradient", "haus"),
    ];

    let mut correct = 0;
    for (pattern, expected) in &test_patterns {
        let img = generate_test_image(pattern, 16, 16);
        let tokens = encoder.encode_image(&img, 16, 16);
        let center = &tokens[tokens.len() / 2].hyperbolic;

        let mut best_label = "";
        let mut best_dist = f64::INFINITY;
        for (_, label) in &dataset {
            let label_hp = trainer.lexicon.get(label).unwrap().hyperbolic.clone();
            if let Ok(d) = ball.distance(center, &label_hp) {
                if d < best_dist {
                    best_dist = d;
                    best_label = label;
                }
            }
        }

        let ok = best_label == *expected;
        if ok { correct += 1; }
        println!("  {} → predicted: {} (expected: {}) {}",
            pattern, best_label, expected, if ok { "ok" } else { "FAIL" });
    }
    println!("\nAccuracy: {}/{} = {:.1}%",
        correct, test_patterns.len(),
        100.0 * correct as f64 / test_patterns.len() as f64);

    // --- Export for HTML demo ---
    export_for_demo(&trainer, &history, &pre_distances, &post_distances,
        "docs/src/development/training_state.json");
    println!("\nExported to docs/src/development/training_state.json");
    println!("Open docs/src/development/agi demo.html to view results.");
}

fn export_for_demo(
    trainer: &SemanticTrainer,
    history: &[EpochMetrics],
    pre: &[(String, String, f64)],
    post: &[(String, String, f64)],
    path: &str,
) {
    let mut json = String::from("{\n");

    // Epoch history
    json.push_str("  \"epoch_history\": [\n");
    for (i, m) in history.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"epoch\": {}, \"reward\": {:.6}, \"edges\": {}}}",
            i, m.avg_reward, m.new_edges
        ));
        if i < history.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    // Tokens (from lexicon)
    json.push_str("  \"tokens\": [\n");
    for (i, token) in trainer.lexicon.tokens.iter().enumerate() {
        let x = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
        let y = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        let class_str = format!("{:?}", token.class).replace("TokenClass::", "");

        json.push_str(&format!(
            "    {{\"word\": \"{}\", \"class\": \"{}\", \"x\": {:.6}, \"y\": {:.6}, \"salience\": {:.4}}}",
            token.surface, class_str, x, y, token.salience
        ));
        if i < trainer.lexicon.tokens.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    // Edges
    json.push_str("  \"edges\": [\n");
    for (i, edge) in trainer.concept_graph.edges.iter().enumerate() {
        let from = trainer.concept_graph.nodes.get(edge.source)
            .map(|n| n.label.clone()).unwrap_or_default();
        let to = trainer.concept_graph.nodes.get(edge.target)
            .map(|n| n.label.clone()).unwrap_or_default();

        json.push_str(&format!(
            "    {{\"from\": \"{}\", \"to\": \"{}\", \"weight\": {:.4}, \"rel\": \"{:?}\"}}",
            from, to, edge.weight, edge.rel
        ));
        if i < trainer.concept_graph.edges.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    // Visual-semantic distances (pre vs post)
    json.push_str("  \"distances\": [\n");
    for (i, ((p1, l1, d1), (_, _, d2))) in pre.iter().zip(post.iter()).enumerate() {
        json.push_str(&format!(
            "    {{\"pattern\": \"{}\", \"label\": \"{}\", \"pre\": {:.6}, \"post\": {:.6}}}",
            p1, l1, d1, d2
        ));
        if i < pre.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ]\n}");

    let mut file = fs::File::create(path).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}