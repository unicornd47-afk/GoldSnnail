//! GridEncoder + ARC Loader — Smoke Test
//!
//! Loads ARC tasks, encodes grids to hyperbolic embeddings,
//! and prints distance stats for the first 3 tasks.

use goldworm::ArcDataset;
use goldworm::vision::grid_encoder::{GridEncoder, train_grid_encoder};

fn main() {
    println!("=== GoldWorm GridEncoder Smoke Test ===\n");

    // 1. Load ARC tasks
    let dataset = match ArcDataset::load_from_directory("data/arc") {
        Ok(ds) => ds,
        Err(e) => {
            println!("Could not load ARC dataset: {}", e);
            println!("Place ARC JSON files in data/arc/ and retry.");
            return;
        }
    };

    println!("Loaded {} tasks", dataset.tasks.len());

    if dataset.tasks.is_empty() {
        println!("No tasks found. Add ARC JSON files to data/arc/.");
        return;
    }

    // 2. Initialize GridEncoder: 100D → 32D → 16D
    let mut encoder = GridEncoder::new(100, 32, 16, 0.75);
    println!(
        "GridEncoder initialized: {} → {} → {} (target_radius={})",
        encoder.dim_in, encoder.dim_hidden, encoder.dim_out, encoder.target_radius
    );

    // 3. Encode first train pair of first 3 tasks
    let mut success = 0;
    let mut attempts = 0;

    for task in dataset.tasks.iter().take(3) {
        if task.train_pairs.is_empty() {
            println!("Task {}: no train pairs", task.id);
            continue;
        }

        let (input_grid, output_grid) = &task.train_pairs[0];

        println!("\nTask: {}", task.id);
        println!("  Input:  {}×{}", input_grid.width, input_grid.height);
        println!("  Output: {}×{}", output_grid.width, output_grid.height);

        // Encode input
        let in_point = match encoder.encode(input_grid) {
            Ok(p) => p,
            Err(e) => {
                println!("  Encode failed: {}", e);
                continue;
            }
        };

        // Encode output
        let out_point = match encoder.encode(output_grid) {
            Ok(p) => p,
            Err(e) => {
                println!("  Encode failed: {}", e);
                continue;
            }
        };

        // Distance (Euclidean approximation in small radius)
        let dist: f64 = in_point
            .coords
            .iter()
            .zip(out_point.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        println!("  Input radius:  {:.4}", in_point.euclidean_norm());
        println!("  Output radius: {:.4}", out_point.euclidean_norm());
        println!("  Distance:      {:.4}", dist);

        attempts += 1;
        if dist < 1.0 {
            success += 1;
        }
    }

    // 4. Summary
    println!("\n=== Smoke Test Summary ===");
    if attempts > 0 {
        println!("{}/{} tasks have Input/Output distance < 1.0", success, attempts);
    } else {
        println!("No valid tasks processed.");
    }

    // 5. Optional: quick self-supervised training
    if dataset.tasks.len() >= 3 {
        println!("\n=== Quick Training (20 epochs) ===");
        let tasks: Vec<_> = dataset.tasks.iter().take(3).cloned().collect();
        train_grid_encoder(&mut encoder, &tasks, 20, 0.01);
    }
}
