//! N-MNIST Histogram Resolution Benchmark
//!
//! Tests different histogram resolutions to find the best feature
//! representation for N-MNIST digit classification.
//!
//! The default 8×8 histograms may be too coarse for N-MNIST digits.
//! This benchmark tests 4×4, 8×8, 16×16, and 32×32 resolutions to
//! determine which gives the best classification accuracy.

use goldsnnail::{
    NmnistDataset, ProjectionLayer,
    project_dvs_to_histogram,
};
use rand::seq::SliceRandom;
use std::time::Instant;

fn main() {
    println!("=== N-MNIST Histogram Resolution Benchmark ===\n");

    let dataset = NmnistDataset::load(1000);
    println!("  Available digits: {:?}", dataset.available_digits);
    println!("  Train: {} samples", dataset.train.len());
    println!("  Test: {} samples", dataset.test.len());

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

    println!("  Using {} train / {} test samples\n", train_set.len(), test_set.len());

    // Use a subset of training data for faster benchmarking
    let train_subset: Vec<_> = train_set.iter().take(500).cloned().collect();
    println!("Using {} training samples for benchmarking\n", train_subset.len());

    let available = &dataset.available_digits;
    let num_classes = available.len();

    let resolutions = [8, 16, 32];
    let mut results = Vec::new();

    for &bins in &resolutions {
        let input_dim = bins * bins;
        println!("Testing {}x{} histograms (input_dim={})...", bins, bins, input_dim);

        let mut layer = ProjectionLayer::new(input_dim, 0.1, dataset.available_digits.clone(), 8);
        let class_centers = goldsnnail::init_class_centers(num_classes, 8, 0.7);

        let start = Instant::now();
        for _epoch in 0..10 {
            for sample in &train_subset {
                let histogram = project_dvs_to_histogram(&sample.events, bins);
                let target_index = available.iter().position(|&d| d == sample.digit).unwrap_or(0);
                layer.train_step(&histogram, sample.digit, target_index, num_classes, &class_centers);
            }
        }
        let train_time = start.elapsed();

        let (accuracy, _per_digit) = layer.evaluate(&test_set, bins, &class_centers);
        println!("  Accuracy: {:.1}% (train time: {:?})\n", accuracy * 100.0, train_time);
        results.push((bins, accuracy, train_time));
    }

    println!("=== Summary ===");
    for (bins, acc, time) in &results {
        println!("  {}x{}: {:.1}% accuracy, {:?} training", bins, bins, acc * 100.0, time);
    }
}
