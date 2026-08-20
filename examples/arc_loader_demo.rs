//! ARC-AGI Loader Demo
//!
//! Loads a single ARC-AGI task from JSON and prints grid information,
//! including hyperbolic embedding norms and concept node generation.

use goldsnnail::{ArcDataset, ConceptGraph, PoincareBall};
use std::path::Path;

fn main() {
    println!("=== GoldSnnail ARC-AGI Loader Demo ===\n");

    let data_dir = Path::new("data/arc");
    if !data_dir.exists() {
        println!("ARC data directory not found: {}", data_dir.display());
        println!("Place ARC-AGI JSON files in data/arc/ and rerun.");
        return;
    }

    match ArcDataset::load_from_directory(data_dir) {
        Ok(dataset) => {
            println!("Loaded {} tasks from {}\n", dataset.len(), data_dir.display());

            if let Some(task) = dataset.tasks.first() {
                println!("Task ID: {}", task.id);
                println!("  Train pairs: {}", task.train_pairs.len());
                for (i, (inp, out)) in task.train_pairs.iter().enumerate() {
                    println!(
                        "    Pair {}: input {}x{}, output {}x{}",
                        i + 1,
                        inp.width,
                        inp.height,
                        out.width,
                        out.height
                    );
                    println!(
                        "      Input unique colors: {:?}",
                        inp.unique_colors()
                    );
                    println!(
                        "      Output unique colors: {:?}",
                        out.unique_colors()
                    );
                }

                println!("\n  Test inputs: {}", task.test_inputs.len());
                for (i, inp) in task.test_inputs.iter().enumerate() {
                    println!("    Test {}: {}x{}", i + 1, inp.width, inp.height);
                }

                let ball = PoincareBall::new(1.0);

                println!("\n  Hyperbolic projections:");
                for (i, (inp, out)) in task.train_pairs.iter().enumerate() {
                    let in_h = inp.to_hyperbolic(&ball, 16);
                    let out_h = out.to_hyperbolic(&ball, 16);
                    match (&in_h, &out_h) {
                        (Ok(in_p), Ok(out_p)) => {
                            println!(
                                "    Pair {}: input norm={:.4}, output norm={:.4}",
                                i + 1,
                                in_p.euclidean_norm(),
                                out_p.euclidean_norm()
                            );
                        }
                        _ => {}
                    }
                }

                println!("\n  ConceptGraph nodes:");
                let mut graph = ConceptGraph::new(1.0);
                if let Err(e) = task.to_concept_nodes(&mut graph) {
                    println!("    Error generating concept nodes: {}", e);
                } else {
                    println!("    Generated {} concept nodes", graph.nodes.len());
                    for node in &graph.nodes {
                        println!(
                            "      id={}, label={}, embedding_norm={:.4}",
                            node.id,
                            node.label,
                            node.embedding.euclidean_norm()
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!("Failed to load dataset: {}", e);
            std::process::exit(1);
        }
    }
}
