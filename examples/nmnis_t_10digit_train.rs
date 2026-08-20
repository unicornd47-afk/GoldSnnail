//! N-MNIST 10-Digit Training Pipeline
//!
//! Trains a 16D MLP projection layer on the full 10-digit N-MNIST dataset
//! using multi-scale time-surface features with per-sample timestamp normalization.
//!
//! This fixes the 10% accuracy collapse on real 10-digit data by:
//! 1. Normalizing per-sample timestamps to [0, 100ms] range
//! 2. Using 3 tau scales (10ms, 50ms, 100ms) simultaneously
//!
//! Requires the `nmnis_t_download` feature for downloading the full dataset:
//!   cargo run --example nmnis_t_10digit_train --release --features nmnis_t_download

use goldworm::{
    NmnistDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_multiscale_features, normalize_timestamps,
};
use rand::seq::SliceRandom;
use std::time::Instant;

const BINS: usize = 16;
const MULTISCALE_TAUS: [f32; 3] = [10_000.0, 50_000.0, 100_000.0];

fn features(events: &[goldworm::DvsEvent]) -> Vec<f32> {
    let normalized = normalize_timestamps(events);
    project_dvs_to_multiscale_features(&normalized, BINS, &MULTISCALE_TAUS)
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}

fn main() {
    println!("=== GoldWorm N-MNIST 10-Digit Training ===\n");
    println!("Encoding: normalized timestamps + multi-scale time-surface");
    println!("  Bins: {}x{}", BINS, BINS);
    println!("  Taus: {:?} us (10ms, 50ms, 100ms)", MULTISCALE_TAUS);
    println!("  Feature dim: {} (histogram + 3 taus × ON/OFF)\n", (1 + 2 * MULTISCALE_TAUS.len()) * BINS * BINS);

    // Load full 10-digit dataset
    println!("Loading N-MNIST dataset (all 10 digits)...");
    let dataset = NmnistDataset::load(500);
    println!("  Available digits: {:?}", dataset.available_digits);
    println!("  Total train: {} samples", dataset.train.len());
    println!("  Total test: {} samples", dataset.test.len());

    if dataset.available_digits.len() < 10 {
        println!("\nWARNING: Only {} digits available. Full 10-digit dataset requires download.", dataset.available_digits.len());
        println!("Run with: cargo run --example nmnis_t_10digit_train --release --features nmnis_t_download");
    }

    let num_classes = dataset.available_digits.len();

    // Create train/test split if needed
    let (train_set, test_set) = if dataset.test.is_empty() {
        println!("  No test set — creating 80/20 train/test split...");
        let mut train = dataset.train.clone();
        let mut rng = rand::thread_rng();
        train.shuffle(&mut rng);
        let split = (train.len() as f32 * 0.8) as usize;
        (train[..split].to_vec(), train[split..].to_vec())
    } else {
        (dataset.train.clone(), dataset.test.clone())
    };

    println!("  Train split: {} samples", train_set.len());
    println!("  Test split: {} samples", test_set.len());

    // Initialize class centers in 16D
    let class_centers = init_class_centers(num_classes, 16, 0.7);
    println!("  Class centers: {} × 16D vectors on radius 0.7", class_centers.len());

    // Create MLP projection layer (16D output for 10-class scalability)
    let input_dim = (1 + 2 * MULTISCALE_TAUS.len()) * BINS * BINS;
    let mut layer = ProjectionLayer::new(input_dim, 0.02, dataset.available_digits.clone(), 16);

    // Training loop
    println!("\nTraining 16D MLP projection layer...");
    let start = Instant::now();
    let mut total_loss = 0.0;

    for epoch in 0..300 {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;

        let lr = 0.002 + 0.018 * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / 300.0).cos());
        layer.set_learning_rate(lr);

        for sample in &train_set {
            let histogram = features(&sample.events);
            let target_index = dataset.available_digits.iter().position(|&d| d == sample.digit).unwrap_or(0);
            let loss = layer.train_step(&histogram, sample.digit, target_index, num_classes, &class_centers);
            epoch_loss += loss;
        }

        let avg_loss = epoch_loss / train_set.len() as f32;
        total_loss += avg_loss;

        if epoch % 50 == 0 || epoch == 299 {
            println!("  Epoch {}: loss={:.4}, lr={:.6}, time={:?}", epoch, avg_loss, lr, epoch_start.elapsed());
        }
    }

    let total_time = start.elapsed();
    println!("\nTraining complete in {:?}", total_time);

    // Evaluate
    println!("\nEvaluating on test set...");
    let mut correct = 0;
    let mut per_digit: Vec<(u8, usize, usize)> = Vec::new();

    for &digit in &dataset.available_digits {
        let digit_samples: Vec<_> = test_set.iter().filter(|s| s.digit == digit).collect();
        let total = digit_samples.len();
        let mut digit_correct = 0;

        for sample in digit_samples {
            let histogram = features(&sample.events);
            let output = layer.project(&histogram);
            let output_f64: Vec<f64> = output.iter().map(|&x| x as f64).collect();

            let mut best_digit = dataset.available_digits[0];
            let mut best_sim = f64::NEG_INFINITY;

            for (class_idx, &d) in dataset.available_digits.iter().enumerate() {
                let sim = cosine_similarity(&output_f64, &class_centers[class_idx]);
                if sim > best_sim {
                    best_sim = sim;
                    best_digit = d;
                }
            }

            if best_digit == digit {
                correct += 1;
                digit_correct += 1;
            }
        }

        per_digit.push((digit, digit_correct, total));
    }

    let accuracy = correct as f32 / test_set.len() as f32;
    println!("  Test accuracy: {:.1}%", accuracy * 100.0);

    for (digit, correct, total) in &per_digit {
        let acc = *correct as f32 / *total as f32;
        println!("  Digit {}: {}/{} ({:.1}%)", digit, correct, total, acc * 100.0);
    }

    // Per-digit analysis
    println!("\n--- Per-Digit Analysis ---");
    let mut min_acc = 1.0;
    let mut min_digit = 0u8;
    for (digit, correct, total) in &per_digit {
        let acc = *correct as f32 / *total as f32;
        if acc < min_acc {
            min_acc = acc;
            min_digit = *digit;
        }
    }
    println!("  Weakest class: Digit {} ({:.1}%)", min_digit, min_acc * 100.0);

    if min_acc < 0.5 {
        println!("  WARNING: Weakest class below 50%. Consider adjusting tau scales.");
    } else if min_acc < 0.7 {
        println!("  NOTE: Weakest class below 70%. Feature encoder may need more tuning.");
    } else {
        println!("  All digits above 70% — multi-scale encoder successfully scales to 10 classes.");
    }

    // Check what the model outputs for each class
    println!("\n--- Model Output Distribution (first 3 samples per class) ---");
    for &digit in &dataset.available_digits {
        let digit_samples: Vec<_> = test_set.iter().filter(|s| s.digit == digit).take(3).collect();
        for sample in &digit_samples {
            let histogram = features(&sample.events);
            let output = layer.project(&histogram);
            println!("  Digit {}: output={:?} norm={:.3}", digit, output, l2_norm(&output));
        }
    }

    // Export results
    let results = format!(
        "{{\"test_accuracy\": {:.4}, \"final_loss\": {:.4}, \"dataset_size\": {}, \"classes\": {}, \"encoding\": \"16d-radial-multiscale\", \"per_digit\": [{}]}}\n",
        accuracy,
        total_loss / 300.0,
        train_set.len() + test_set.len(),
        num_classes,
        per_digit.iter().map(|(d, c, t)| format!("{{\"digit\": {}, \"correct\": {}, \"total\": {}}}", d, c, t)).collect::<Vec<_>>().join(", ")
    );
    let _ = std::fs::write("docs/src/development/nmnis_t_10digit_results.json", results);
    println!("\nResults exported to docs/src/development/nmnis_t_10digit_results.json");
}

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}
