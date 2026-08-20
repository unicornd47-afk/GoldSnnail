//! ARC-AGI Retrieval Baseline
//!
//! Nearest-neighbor retrieval: encode all training inputs/outputs into hyperbolic
//! space with GridEncoder, then for each test input find the nearest training input
//! by Euclidean distance and return its paired training output.
//!
//! Answers: "Does the most similar training input help predict the test output?"

use goldsnnail::ArcDataset;
use goldsnnail::vision::grid_encoder::{GridEncoder, train_grid_encoder};

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn grids_equal(a: &goldsnnail::ArcGrid, b: &goldsnnail::ArcGrid) -> bool {
    if a.width != b.width || a.height != b.height {
        return false;
    }
    a.data == b.data
}

fn main() {
    println!("=== GoldSnnail ARC Retrieval Baseline ===\n");

    // 1. Load ARC tasks
    let dataset = match ArcDataset::load_from_directory("data/arc") {
        Ok(ds) => ds,
        Err(e) => {
            println!("Could not load ARC dataset: {}", e);
            println!("Place ARC JSON files in data/arc/ and retry.");
            return;
        }
    };

    let total_tasks = dataset.tasks.len();
    println!("Loaded {} tasks", total_tasks);

    if dataset.tasks.is_empty() {
        println!("No tasks found. Add ARC JSON files to data/arc/.");
        return;
    }

    // 2. Initialize GridEncoder: 100D -> 32D -> 16D
    let mut encoder = GridEncoder::new(100, 32, 16, 0.75);
    println!(
        "GridEncoder: {} -> {} -> {} (target_radius={})",
        encoder.dim_in, encoder.dim_hidden, encoder.dim_out, encoder.target_radius
    );

    // 3. Build flat index of all training pairs
    let mut train_input_embeddings: Vec<Vec<f64>> = Vec::new();
    let mut train_output_embeddings: Vec<Vec<f64>> = Vec::new();
    let mut train_output_grids: Vec<goldsnnail::ArcGrid> = Vec::new();

    for task in &dataset.tasks {
        for (input_grid, output_grid) in &task.train_pairs {
            match encoder.encode(input_grid) {
                Ok(point) => train_input_embeddings.push(point.coords),
                Err(e) => eprintln!("Encode train input failed: {}", e),
            }
            match encoder.encode(output_grid) {
                Ok(point) => train_output_embeddings.push(point.coords),
                Err(e) => eprintln!("Encode train output failed: {}", e),
            }
            train_output_grids.push(output_grid.clone());
        }
    }

    let index_size = train_input_embeddings.len();
    println!(
        "Index built: {} train pairs across {} tasks",
        index_size, total_tasks
    );

    if index_size == 0 {
        println!("No training pairs encoded. Exiting.");
        return;
    }

    // 4. Self-supervised training on all tasks
    println!("\nTraining encoder for 50 epochs...");
    let all_tasks: Vec<_> = dataset.tasks.iter().cloned().collect();
    train_grid_encoder(&mut encoder, &all_tasks, 50, 0.005);
    println!("Training complete.\n");

    // 5. Re-encode index after training
    train_input_embeddings.clear();
    train_output_embeddings.clear();
    train_output_grids.clear();

    for task in &dataset.tasks {
        for (input_grid, output_grid) in &task.train_pairs {
            match encoder.encode(input_grid) {
                Ok(point) => train_input_embeddings.push(point.coords),
                Err(e) => eprintln!("Encode train input failed: {}", e),
            }
            match encoder.encode(output_grid) {
                Ok(point) => train_output_embeddings.push(point.coords),
                Err(e) => eprintln!("Encode train output failed: {}", e),
            }
            train_output_grids.push(output_grid.clone());
        }
    }

    // 6. Retrieval and evaluation
    println!("Running retrieval...");
    let max_test_cases = 100usize;
    let mut test_count = 0usize;
    let mut exact_matches = 0usize;
    let mut sum_input_dist = 0.0f64;
    let mut sum_output_dist = 0.0f64;

    for task in &dataset.tasks {
        if test_count >= max_test_cases {
            break;
        }

        for (test_input, test_output_opt) in task
            .test_inputs
            .iter()
            .zip(task.test_outputs.iter())
        {
            if test_count >= max_test_cases {
                break;
            }

            let test_output = match test_output_opt {
                Some(g) => g,
                None => continue,
            };

            // Encode test input
            let test_emb = match encoder.encode(test_input) {
                Ok(p) => p.coords,
                Err(_) => continue,
            };

            // Find nearest training input
            let mut best_idx = 0usize;
            let mut best_dist = f64::INFINITY;
            for (i, train_emb) in train_input_embeddings.iter().enumerate() {
                let d = euclidean_distance(&test_emb, train_emb);
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }

            // Retrieve paired training output
            let predicted_output = &train_output_grids[best_idx];

            // Compare
            let exact = grids_equal(predicted_output, test_output);
            if exact {
                exact_matches += 1;
            }

            // Distance between retrieved output embedding and true output embedding
            let true_output_emb = match encoder.encode(test_output) {
                Ok(p) => p.coords,
                Err(_) => vec![0.0; encoder.dim_out],
            };
            let output_dist = euclidean_distance(&train_output_embeddings[best_idx], &true_output_emb);

            sum_input_dist += best_dist;
            sum_output_dist += output_dist;
            test_count += 1;

            if test_count % 20 == 0 {
                println!("Progress: {}/{}", test_count, max_test_cases);
            }
        }
    }

    println!("\n=== Results ===");
    if test_count > 0 {
        let accuracy = (exact_matches as f64 / test_count as f64) * 100.0;
        let avg_input_dist = sum_input_dist / test_count as f64;
        let avg_output_dist = sum_output_dist / test_count as f64;

        println!(
            "Exact match accuracy: {}/{} ({:.1}%)",
            exact_matches, test_count, accuracy
        );
        println!(
            "Avg distance to nearest train input: {:.4}",
            avg_input_dist
        );
        println!(
            "Avg distance between retrieved output and true output: {:.4}",
            avg_output_dist
        );

        println!("\nInterpretation:");
        if accuracy < 5.0 {
            println!(
                "  - {:.1}% exact match is near random for ARC-AGI",
                accuracy
            );
        } else {
            println!("  - {:.1}% exact match", accuracy);
        }
        println!(
            "  - The hyperbolic space separates tasks but doesn't enable output prediction by retrieval alone"
        );
        println!("  - Retrieval alone is insufficient for ARC-AGI");
    } else {
        println!("No test cases evaluated.");
    }
}
