//! GoldWorm Vision-Semantic Infinite Training Loop
//!
//! Runs training indefinitely until Ctrl+C (SIGINT). Exports JSON state
//! every N epochs for the HTML demo visualization.
//!
//! Usage:
//!   cargo run --example train_vision_loop
//!
//! Press Ctrl+C to stop. The latest training_state.json is always available.

use goldworm::{
    PatchEncoder, SemanticTrainer, SemanticLearner, LearningRates, EpochMetrics,
    generate_test_image, PoincareBall,
};
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

static RUNNING: AtomicBool = AtomicBool::new(true);

fn main() {
    println!("=== GoldWorm Vision Loop (Ctrl+C to stop) ===\n");

    // Ctrl+C handler
    {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
        ctrlc::set_handler(move || {
            println!("\n\n🛑 Ctrl+C received — shutting down gracefully...");
            RUNNING.store(false, Ordering::SeqCst);
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");
        
        // Keep handler alive
        let r2 = running.clone();
        std::thread::spawn(move || {
            while r2.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
            }
        });
    }

    // --- Setup ---
    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());
    let mut encoder = PatchEncoder::new(4, 2, 1.0);
    let _ball = PoincareBall::new(1.0);

    // --- Dataset ---
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

    // --- Training state ---
    let mut epoch: u64 = 0;
    let mut history: Vec<EpochMetrics> = Vec::with_capacity(1000);
    let log_interval = 5;
    let mut last_export = Instant::now();

    println!("Training started. Press Ctrl+C to stop.\n");

    while RUNNING.load(Ordering::SeqCst) {
        epoch += 1;
        let start = Instant::now();
        let mut total_reward = 0.0;
        let mut samples: u64 = 0;

        // One epoch: iterate over dataset with random order
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        let mut rng = thread_rng();
        let mut shuffled: Vec<_> = dataset.iter().collect();
        shuffled.shuffle(&mut rng);

        for (pattern, label) in &shuffled {
            let img = generate_test_image(pattern, 16, 16);
            let mut visual = encoder.encode_image(&img, 16, 16);
            
            // Bind all patches to label
            for token in &mut visual {
                let _ = encoder.bind_visual_semantic(token, label);
            }

            // Sentence for semantic trainer
            let sentence = trainer.composer.build_sentence_simple(label, "sieht");
            
            let reward = trainer.train_step(&sentence, false);
            let tokens = trainer.composer.resolve(&sentence);
            if !tokens.is_empty() {
                let _ = learner.learn_from_reward(
                    &reward, &tokens, None, None,
                    &mut trainer.concept_graph, &mut trainer.lexicon,
                );
            }
            
            total_reward += reward.total;
            samples += 1;
        }

        let metrics = EpochMetrics {
            avg_reward: total_reward / samples.max(1) as f64,
            avg_embedding_shift: 0.0,
            new_edges: trainer.concept_graph.edges.len(),
        };
        history.push(metrics.clone());

        let elapsed = start.elapsed();

        // Periodic logging
        if epoch % log_interval == 0 {
            println!(
                "[Epoch {:>6}] reward={:.4} | edges={:<4} | vocab={:<3} | time={:.2?}",
                epoch, metrics.avg_reward, metrics.new_edges,
                trainer.lexicon.tokens.len(), elapsed
            );
        }

        // Periodic export every 30 seconds
        if last_export.elapsed() > Duration::from_secs(30) {
            let _ = export_state(&trainer, &history, "docs/src/development/training_state.json");
            println!("  📊 Exported training_state.json (epoch {})", epoch);
            last_export = Instant::now();
        }

        // Yield to allow Ctrl+C processing
        std::thread::sleep(Duration::from_millis(1));
    }

    // --- Final export ---
    println!("\n=== Final Export ===");
    let _ = export_state(&trainer, &history, "docs/src/development/training_state.json");
    println!("Final state exported to docs/src/development/training_state.json");

    // --- Summary ---
    println!("\n=== Training Summary ===");
    println!("Total epochs:     {}", epoch);
    println!("Final reward:     {:.4}", history.last().map(|m| m.avg_reward).unwrap_or(0.0));
    println!("Concept edges:    {}", trainer.concept_graph.edges.len());
    println!("Vocabulary size:  {}", trainer.lexicon.tokens.len());
    println!("\nOpen docs/src/development/agi demo.html to view results.");
}

fn export_state(trainer: &SemanticTrainer, history: &[EpochMetrics], path: &str) -> Result<(), String> {
    let mut json = String::from("{\n");

    // Epoch history (keep last 1000 for performance)
    json.push_str("  \"epoch_history\": [\n");
    let start = history.len().saturating_sub(1000);
    for (i, m) in history.iter().enumerate().skip(start) {
        json.push_str(&format!(
            "    {{\"epoch\": {}, \"reward\": {:.6}, \"edges\": {}}}",
            i, m.avg_reward, m.new_edges
        ));
        if i < history.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    // Tokens
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
    json.push_str("  ]\n}");

    fs::create_dir_all("docs/src/development").map_err(|e| e.to_string())?;
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}