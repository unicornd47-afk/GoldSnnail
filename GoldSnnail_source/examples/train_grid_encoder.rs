//! GoldWorm GridEncoder Training
//!
//! Self-supervised training on ARC-AGI train pairs.
//! Minimizes euklidically approximated hyperbolic distance between
//! input and output embeddings.
//!
//! Usage:
//!   cargo run --example train_grid_encoder
//!
//! Output:
//!   Loss curve over epochs on 50 ARC train tasks.

use goldworm::vision::{ArcDataset, GridEncoder};
use std::time::Instant;

fn main() {
    println!("=== GoldWorm GridEncoder Training ===\n");

    // Dataset laden
    println!("Loading ARC-AGI training dataset...");
    let dataset = ArcDataset::load_from_directory("data/arc-agi-2/data/training")
        .expect("Failed to load ARC training dataset");

    println!("Loaded {} tasks", dataset.len());

    if dataset.is_empty() {
        eprintln!("ERROR: No tasks loaded. Check data path.");
        std::process::exit(1);
    }

    // Encoder initialisieren
    // 100D Feature-Vektor → 32D Hidden (ReLU) → 16D Output (L2-normalisiert)
    let mut encoder = GridEncoder::new(100, 32, 16, 0.75);

    // Trainings-Parameter
    let epochs = 100;
    let learning_rate = 0.01;
    let num_tasks = 50.min(dataset.len());

    // Erste num_tasks Tasks auswählen
    let tasks: Vec<_> = dataset.tasks.iter().take(num_tasks).cloned().collect();

    println!(
        "Training on {} tasks ({} total available)",
        tasks.len(),
        dataset.len()
    );
    println!("Config: epochs={}, lr={}, dim_in=100, dim_hidden=32, dim_out=16, target_radius=0.75\n", epochs, learning_rate);

    // Trainings-Loop
    let start = Instant::now();

    // Sammle Loss pro Epoch für die Kurve
    let mut loss_history: Vec<f64> = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut count = 0usize;

        use rand::seq::SliceRandom;
        use rand::thread_rng;
        let mut rng = thread_rng();

        let mut shuffled: Vec<usize> = (0..tasks.len()).collect();
        shuffled.shuffle(&mut rng);

        for &idx in &shuffled {
            let task = &tasks[idx];
            for (input_grid, output_grid) in &task.train_pairs {
                match encoder.train_step(input_grid, output_grid, learning_rate) {
                    Ok(loss) => {
                        total_loss += loss;
                        count += 1;
                    }
                    Err(e) => {
                        eprintln!("  Training error in task '{}': {}", task.id, e);
                    }
                }
            }
        }

        let avg_loss = if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        };

        loss_history.push(avg_loss);

        // Logging
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("[Epoch {:>3}] Loss = {:.6}  ({} pairs processed)", epoch, avg_loss, count);
        }
    }

    let duration = start.elapsed();

    // Zusammenfassung
    println!("\n=== Training Complete ===");
    println!("Total time: {:.2}s", duration.as_secs_f64());
    println!("Final loss: {:.6}", loss_history.last().copied().unwrap_or(0.0));
    println!("Initial loss: {:.6}", loss_history.first().copied().unwrap_or(0.0));
    println!("Loss reduction: {:.2}%",
        if loss_history.len() >= 2 {
            let init = loss_history[0];
            let final_ = loss_history[loss_history.len() - 1];
            if init > 0.0 {
                ((init - final_) / init * 100.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    );

    // Go/No-Go Gate
    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    println!("\n=== Go/No-Go Gate ===");
    if final_loss < 0.05 {
        println!("🟢 Loss < 0.05 — Encoder is trainable. Continue to Week 2 (Retrieval-Retry).");
    } else if final_loss > 0.5 {
        println!("🟡 Loss stagnates > 0.5 — Consider adjusting learning rate or architecture.");
    } else {
        println!("🟡 Intermediate result — Loss between 0.05 and 0.5. May need more epochs or LR tuning.");
    }

    // Loss-Kurve als CSV ausgeben
    let csv_path = "loss_curve.csv";
    let mut csv = String::from("epoch,loss\n");
    for (i, &loss) in loss_history.iter().enumerate() {
        csv.push_str(&format!("{},{:.6}\n", i, loss));
    }
    std::fs::write(csv_path, csv).expect("Failed to write loss_curve.csv");
    println!("\nLoss curve written to: {}", csv_path);
}
