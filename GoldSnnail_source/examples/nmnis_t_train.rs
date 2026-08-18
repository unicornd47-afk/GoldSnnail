//! N-MNIST Training Pipeline with 8D MLP (3-digit subset)
//!
//! Trains an 8D MLP projection layer with L2-normalization and radial
//! cross-entropy loss for direction-only learning.

use goldworm::{
    NmnistDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_combined_features,
};
use rand::seq::SliceRandom;
use std::time::Instant;

const BINS: usize = 16;  // Higher resolution for time-surface
const TAU_US: f32 = 50000.0;

fn features(events: &[goldworm::DvsEvent]) -> Vec<f32> {
    project_dvs_to_combined_features(events, BINS, TAU_US)
}

fn main() {
    println!("=== GoldWorm N-MNIST 8D MLP Training (3 digits) ===\n");
    println!("Encoding: spatial histogram + time-surface ({}x{} bins, tau={}us)\n", BINS, BINS, TAU_US);

    // Load dataset
    println!("Loading N-MNIST dataset...");
    let dataset = NmnistDataset::load(1000);
    println!("  Available digits: {:?}", dataset.available_digits);
    println!("  Total train: {} samples", dataset.train.len());

    let num_classes = dataset.available_digits.len();

    // Create train/test split
    let (train_set, test_set) = if dataset.test.is_empty() {
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

    // Initialize class centers in 8D
    let class_centers = init_class_centers(num_classes, 8, 0.7);

    // Create MLP projection layer (8D output)
    let input_dim = 3 * BINS * BINS; // spatial hist + time-surface ON + time-surface OFF
    let mut layer = ProjectionLayer::new(input_dim, 0.02, dataset.available_digits.clone(), 8);

    // Training loop
    println!("\nTraining 8D MLP projection layer...");
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

        if epoch % 25 == 0 || epoch == 299 {
            println!("  Epoch {}: loss={:.4}, lr={:.6}, time={:?}", epoch, avg_loss, lr, epoch_start.elapsed());
        }
    }

    let total_time = start.elapsed();
    println!("\nTraining complete in {:?}", total_time);

    // Evaluate
    println!("\nEvaluating on test set...");
    let (accuracy, per_digit) = layer.evaluate(&test_set, BINS, &class_centers);
    println!("  Test accuracy: {:.1}%", accuracy * 100.0);

    for (digit, correct, total) in &per_digit {
        let acc = *correct as f32 / *total as f32;
        println!("  Digit {}: {}/{} ({:.1}%)", digit, correct, total, acc * 100.0);
    }

    // Check what the model outputs for each class (should be different if learning)
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
        "{{\"test_accuracy\": {:.4}, \"final_loss\": {:.4}, \"dataset_size\": {}, \"classes\": {}, \"encoding\": \"8d-radial\"}}\n",
        accuracy,
        total_loss / 300.0,
        train_set.len() + test_set.len(),
        num_classes
    );
    let _ = std::fs::write("docs/src/development/nmnis_t_results.json", results);
    println!("\nResults exported to docs/src/development/nmnis_t_results.json");
}

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}
