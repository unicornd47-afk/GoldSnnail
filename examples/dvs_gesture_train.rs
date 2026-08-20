//! DVS-Gesture Training Pipeline
//!
//! Trains a 16D MLP projection layer on the DVS-Gesture dataset (11 gesture classes).
//! Uses multi-scale time-surface features with per-sample timestamp normalization.
//!
//! This demonstrates that the multi-scale encoder generalizes beyond static digits
//! to dynamic temporal gestures (waving, swiping, circular motions).

use goldworm::{
    DvsGestureDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_multiscale_features, normalize_timestamps, GESTURE_LABELS,
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
    println!("=== GoldWorm DVS-Gesture Training ===\n");
    println!("Encoding: normalized timestamps + multi-scale time-surface");
    println!("  Bins: {}x{}", BINS, BINS);
    println!("  Taus: {:?} us (10ms, 50ms, 100ms)", MULTISCALE_TAUS);
    println!("  Feature dim: {} (histogram + 3 taus × ON/OFF)\n", (1 + 2 * MULTISCALE_TAUS.len()) * BINS * BINS);

    // Load DVS-Gesture dataset
    println!("Loading DVS-Gesture dataset...");
    let dataset = DvsGestureDataset::load(200);
    println!("  Available gestures: {:?}", dataset.available_gestures);
    println!("  Total train: {} samples", dataset.train.len());
    println!("  Total test: {} samples", dataset.test.len());

    let num_classes = dataset.available_gestures.len();

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

    // Create MLP projection layer
    let input_dim = (1 + 2 * MULTISCALE_TAUS.len()) * BINS * BINS;
    let mut layer = ProjectionLayer::new(input_dim, 0.02, dataset.available_gestures.clone(), 16);

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
            let target_index = dataset.available_gestures.iter().position(|&g| g == sample.gesture).unwrap_or(0);
            let loss = layer.train_step(&histogram, sample.gesture, target_index, num_classes, &class_centers);
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
    let mut per_gesture: Vec<(u8, usize, usize)> = Vec::new();

    for &gesture in &dataset.available_gestures {
        let gesture_samples: Vec<_> = test_set.iter().filter(|s| s.gesture == gesture).collect();
        let total = gesture_samples.len();
        let mut gesture_correct = 0;

        for sample in gesture_samples {
            let histogram = features(&sample.events);
            let output = layer.project(&histogram);
            let output_f64: Vec<f64> = output.iter().map(|&x| x as f64).collect();

            let mut best_gesture = dataset.available_gestures[0];
            let mut best_sim = f64::NEG_INFINITY;

            for (class_idx, &g) in dataset.available_gestures.iter().enumerate() {
                let sim = cosine_similarity(&output_f64, &class_centers[class_idx]);
                if sim > best_sim {
                    best_sim = sim;
                    best_gesture = g;
                }
            }

            if best_gesture == gesture {
                correct += 1;
                gesture_correct += 1;
            }
        }

        per_gesture.push((gesture, gesture_correct, total));
    }

    let accuracy = correct as f32 / test_set.len() as f32;
    println!("  Test accuracy: {:.1}%", accuracy * 100.0);

    for (gesture, correct, total) in &per_gesture {
        let acc = *correct as f32 / *total as f32;
        let label = GESTURE_LABELS.get(*gesture as usize).unwrap_or(&"unknown");
        println!("  Gesture {} ({}): {}/{} ({:.1}%)", gesture, label, correct, total, acc * 100.0);
    }

    // Per-gesture analysis
    println!("\n--- Per-Gesture Analysis ---");
    let mut min_acc = 1.0;
    let mut min_gesture = 0u8;
    for (gesture, correct, total) in &per_gesture {
        let acc = *correct as f32 / *total as f32;
        if acc < min_acc {
            min_acc = acc;
            min_gesture = *gesture;
        }
    }
    let min_label = GESTURE_LABELS.get(min_gesture as usize).unwrap_or(&"unknown");
    println!("  Weakest class: Gesture {} ({}) ({:.1}%)", min_gesture, min_label, min_acc * 100.0);

    if min_acc < 0.5 {
        println!("  WARNING: Weakest class below 50%. Consider adjusting tau scales.");
    } else if min_acc < 0.7 {
        println!("  NOTE: Weakest class below 70%. Feature encoder may need more tuning.");
    } else {
        println!("  All gestures above 70% — multi-scale encoder successfully scales to 11 dynamic classes.");
    }

    // Check what the model outputs for each class
    println!("\n--- Model Output Distribution (first 2 samples per class) ---");
    for &gesture in &dataset.available_gestures {
        let gesture_samples: Vec<_> = test_set.iter().filter(|s| s.gesture == gesture).take(2).collect();
        for sample in &gesture_samples {
            let histogram = features(&sample.events);
            let output = layer.project(&histogram);
            println!("  Gesture {}: output={:?} norm={:.3}", gesture, output, l2_norm(&output));
        }
    }

    // Export results
    let results = format!(
        "{{\"test_accuracy\": {:.4}, \"final_loss\": {:.4}, \"dataset_size\": {}, \"classes\": {}, \"encoding\": \"16d-radial-multiscale-gesture\", \"per_gesture\": [{}]}}\n",
        accuracy,
        total_loss / 300.0,
        train_set.len() + test_set.len(),
        num_classes,
        per_gesture.iter().map(|(g, c, t)| format!("{{\"gesture\": {}, \"correct\": {}, \"total\": {}}}", g, c, t)).collect::<Vec<_>>().join(", ")
    );
    let _ = std::fs::write("docs/src/development/dvs_gesture_results.json", results);
    println!("\nResults exported to docs/src/development/dvs_gesture_results.json");
}

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}
