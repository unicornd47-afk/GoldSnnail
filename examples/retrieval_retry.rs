//! Retrieval-Retry Test — Woche 2
//!
//! Testet, ob ein TRAINIERTER GridEncoder den Hybrid-Solver verbessert.
//! Vergleich: Untrainierter Encoder vs. Trainierter Encoder
//!
//! Hypothese: Trainierter Encoder → bessere Task-Router-Embeddings → höhere Accuracy

use goldworm::vision::{ArcDataset, GridEncoder};
use goldworm::vision::grid_encoder::train_grid_encoder;
use goldworm::vision::hybrid_solver::evaluate_hybrid_solver;
use std::time::Instant;

fn main() {
    println!("=== GoldWorm Retrieval-Retry Test (Woche 2) ===\n");

    // Lade Trainingsdaten für den Encoder
    println!("Loading ARC training dataset for encoder training...");
    let train_dataset = match ArcDataset::load_from_directory("data/arc-agi-2/data/training") {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("Failed to load training dataset: {}", e);
            std::process::exit(1);
        }
    };
    println!("Loaded {} training tasks", train_dataset.len());

    // Lade Evaluationsdaten für den Solver
    println!("Loading ARC evaluation dataset for solver evaluation...");
    let eval_dataset = match ArcDataset::load_from_directory("data/arc-agi-2/data/evaluation") {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("Failed to load evaluation dataset: {}", e);
            std::process::exit(1);
        }
    };
    println!("Loaded {} evaluation tasks", eval_dataset.len());

    // --- Baseline: Untrainierter Encoder ---
    println!("\n=== Baseline: Untrainierter Encoder ===");
    let untrained_encoder = GridEncoder::new(100, 32, 16, 0.75);
    let baseline_result = evaluate_hybrid_solver(&eval_dataset, &untrained_encoder, 5, 20);
    print_result(&baseline_result, "Baseline");

    // --- Trainierter Encoder ---
    println!("\n=== Training GridEncoder auf 50 Tasks ===");
    let mut trained_encoder = GridEncoder::new(100, 32, 16, 0.75);
    
    let train_start = Instant::now();
    let train_tasks: Vec<_> = train_dataset.tasks.iter().take(50).cloned().collect();
    train_grid_encoder(&mut trained_encoder, &train_tasks, 100, 0.01);
    let train_duration = train_start.elapsed();
    println!("Training completed in {:.2}s", train_duration.as_secs_f64());

    println!("\n=== Evaluierung: Trainierter Encoder ===");
    let trained_result = evaluate_hybrid_solver(&eval_dataset, &trained_encoder, 5, 20);
    print_result(&trained_result, "Trained");

    // --- Vergleich ---
    println!("\n=== Vergleich: Baseline vs. Trained ===");
    println!(
        "Accuracy:     Baseline = {:.1}% | Trained = {:.1}% | Delta = {:+.1}%",
        baseline_result.accuracy * 100.0,
        trained_result.accuracy * 100.0,
        (trained_result.accuracy - baseline_result.accuracy) * 100.0
    );
    println!(
        "Attempt Rate: Baseline = {:.1}% | Trained = {:.1}% | Delta = {:+.1}%",
        baseline_result.attempt_rate * 100.0,
        trained_result.attempt_rate * 100.0,
        (trained_result.attempt_rate - baseline_result.attempt_rate) * 100.0
    );

    // --- Go/No-Go Gate ---
    println!("\n=== Go/No-Go Gate ===");
    if trained_result.accuracy > 0.0 {
        println!("🟢 TRAINING HILFT! Accuracy > 0%");
        println!("   → Skaliere auf alle Tasks, erhöhe k, mehr Heuristiken.");
    } else if trained_result.accuracy == baseline_result.accuracy && trained_result.attempt_rate > baseline_result.attempt_rate {
        println!("🟡 TEILWEISE VERBESSERUNG: Attempt Rate höher, aber keine korrekten Lösungen.");
        println!("   → Router findet mehr Nachbarn, aber Heuristiken übertragen nicht.");
    } else if trained_result.accuracy == 0.0 && trained_result.attempt_rate == 0.0 {
        println!("🔴 TRAINING HILFT NICHT. Hybrid-Solver bleibt bei 0%.");
        println!("   → Harter No-Go für den Hybrid-Ansatz.");
        println!("   → Pivot: Efficiency-Leaderboard + ARC-AGI-2 Monitoring.");
    } else {
        println!("🟡 UNKLAR: Genauere Analyse nötig.");
    }
}

fn print_result(result: &goldworm::vision::hybrid_solver::EvaluationResult, label: &str) {
    println!("\n--- {} ---", label);
    println!("Gesamt:        {}", result.total);
    println!(
        "Versucht:      {} ({:.1}%)",
        result.attempted,
        result.attempt_rate * 100.0
    );
    println!(
        "Korrekt:       {} ({:.1}%)",
        result.correct,
        result.accuracy * 100.0
    );
}
