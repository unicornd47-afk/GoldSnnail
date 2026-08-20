//! Efficiency Benchmark — Hard Facts for the Report
//!
//! Measures: Parameter Count, Memory Footprint, Inference Latency, Throughput,
//! End-to-End Latency (incl. feature extraction), and Accuracy (sanity check).

use goldworm::{
    NmnistDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_multiscale_features, normalize_timestamps, DvsEvent,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::Instant;
use std::collections::HashMap;

const BINS: usize = 16;
const MULTISCALE_TAUS: [f32; 3] = [10_000.0, 50_000.0, 100_000.0];

fn features(events: &[DvsEvent]) -> Vec<f32> {
    let normalized = normalize_timestamps(events);
    project_dvs_to_multiscale_features(&normalized, BINS, &MULTISCALE_TAUS)
}

fn main() {
    println!("=== Efficiency Benchmark ===\n");

    let dataset = NmnistDataset::load(500);
    if dataset.available_digits.len() < 10 {
        println!("Need all 10 digits. Run with --features nmnis_t_download");
        return;
    }

    let input_dim = (1 + 2 * MULTISCALE_TAUS.len()) * BINS * BINS;
    let all_classes: Vec<u8> = (0..10).collect();
    let class_centers = init_class_centers(10, 16, 0.7);
    let mut layer = ProjectionLayer::new(input_dim, 0.02, all_classes.clone(), 16);

    // Quick training (20 epochs — enough for functional weights)
    println!("Training...");
    for epoch in 0..20 {
        let lr = 0.002 + 0.018 * 0.5 *
            (1.0 + (std::f32::consts::PI * epoch as f32 / 20.0).cos());
        layer.set_learning_rate(lr);
        for sample in &dataset.train {
            let hist = features(&sample.events);
            let target_idx = all_classes.iter().position(|&d| d == sample.digit).unwrap();
            let _loss = layer.train_step(&hist, sample.digit, target_idx, 10, &class_centers);
        }
    }
    println!("Training complete.\n");

    // Prepare test data by class (fallback to split if no test set)
    let test_samples: Vec<_> = if dataset.test.is_empty() {
        let mut train = dataset.train.clone();
        train.shuffle(&mut thread_rng());
        let split = (train.len() as f32 * 0.8) as usize;
        train[split..].to_vec()
    } else {
        dataset.test.clone()
    };

    let mut test_by_class: HashMap<u8, Vec<_>> = HashMap::new();
    for sample in &test_samples {
        test_by_class.entry(sample.digit).or_default().push(sample.clone());
    }

    // =====================================================================
    // 1. PARAMETER COUNT
    // =====================================================================
    let mut total_params = 0usize;
    total_params += layer.w1.nrows() * layer.w1.ncols();
    total_params += layer.b1.len();
    total_params += layer.w2.nrows() * layer.w2.ncols();
    total_params += layer.b2.len();
    total_params += layer.w3.nrows() * layer.w3.ncols();
    total_params += layer.b3.len();
    total_params += layer.w4.nrows() * layer.w4.ncols();
    total_params += layer.b4.len();
    println!("Parameter Count:              {:>10}", total_params);

    // =====================================================================
    // 2. MODEL MEMORY FOOTPRINT
    // =====================================================================
    let model_mem = measure_memory_footprint(&layer);
    println!("Model Memory Footprint:       {:>10} bytes ({:.2} KB / {:.3} MB)",
             model_mem, model_mem as f64 / 1024.0, model_mem as f64 / (1024.0 * 1024.0));

    // =====================================================================
    // 3. INFERENCE LATENCY (raw forward pass only)
    // =====================================================================
    let n_iterations = 10_000;
    let dummy_input = vec![0.0f32; input_dim];

    // Warmup
    for _ in 0..100 {
        let _ = layer.project(&dummy_input);
    }

    let lat_start = Instant::now();
    for _ in 0..n_iterations {
        let _ = layer.project(&dummy_input);
    }
    let lat_total = lat_start.elapsed();
    let avg_latency_us = lat_total.as_secs_f64() * 1_000_000.0 / n_iterations as f64;
    println!("Inference Latency (raw):      {:>10.3} µs  ({} iters)", avg_latency_us, n_iterations);

    // =====================================================================
    // 4. THROUGHPUT (full pipeline: feature extract + inference)
    // =====================================================================
    let throughput_start = Instant::now();
    let mut processed = 0usize;
    for samples in test_by_class.values() {
        for sample in samples {
            let _ = layer.project(&features(&sample.events));
            processed += 1;
        }
    }
    let throughput_dur = throughput_start.elapsed();
    let throughput = processed as f64 / throughput_dur.as_secs_f64();
    println!("Throughput (E2E):             {:>10.1} samples/sec  ({} samples in {:?})",
             throughput, processed, throughput_dur);

    // =====================================================================
    // 5. END-TO-END LATENCY (feature extraction + inference, averaged)
    // =====================================================================
    let e2e_samples = 1000usize.min(test_samples.len());
    let e2e_start = Instant::now();
    for sample in test_samples.iter().take(e2e_samples) {
        let _features = features(&sample.events);
        let _output = layer.project(&_features);
    }
    let e2e_dur = e2e_start.elapsed();
    let avg_e2e_us = if e2e_samples > 0 {
        e2e_dur.as_secs_f64() * 1_000_000.0 / e2e_samples as f64
    } else {
        0.0
    };
    println!("End-to-End Latency:           {:>10.3} µs  ({} samples)", avg_e2e_us, e2e_samples);

    // =====================================================================
    // 6. ACCURACY (sanity check — should match ~58%)
    // =====================================================================
    let mut correct = 0usize;
    let mut total = 0usize;
    for (digit, samples) in &test_by_class {
        for sample in samples {
            let hist = features(&sample.events);
            let out = layer.project(&hist);
            let out_f64: Vec<f64> = out.iter().map(|&x| x as f64).collect();

            let mut best_idx = 0usize;
            let mut best_sim = f64::NEG_INFINITY;
            for (i, center) in class_centers.iter().enumerate() {
                let sim = out_f64.iter().zip(center).map(|(a, b)| a * b).sum::<f64>();
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            if best_idx == *digit as usize {
                correct += 1;
            }
            total += 1;
        }
    }
    let accuracy = if total > 0 {
        format!("{:.1}%", correct as f64 / total as f64 * 100.0)
    } else {
        "N/A (no test data)".to_string()
    };
    println!("Accuracy (sanity):            {:>9}  ({}/{})", 
             accuracy, correct, total);

    // =====================================================================
    // 7. SYSTEM SIZE SUMMARY
    // =====================================================================
    println!("\n--- System Size Summary ---");
    println!("Input dimension:              {}", input_dim);
    println!("Hidden dimensions:            128 → 64 → 32");
    println!("Output dimension:             16");
    println!("Total trainable parameters:   {}", total_params);
    println!("Memory at load (model only):  {:.2} KB", model_mem as f64 / 1024.0);
}

fn measure_memory_footprint(layer: &ProjectionLayer) -> usize {
    let mut bytes = 0usize;
    bytes += layer.w1.nrows() * layer.w1.ncols() * std::mem::size_of::<f32>();
    bytes += layer.b1.len() * std::mem::size_of::<f32>();
    bytes += layer.w2.nrows() * layer.w2.ncols() * std::mem::size_of::<f32>();
    bytes += layer.b2.len() * std::mem::size_of::<f32>();
    bytes += layer.w3.nrows() * layer.w3.ncols() * std::mem::size_of::<f32>();
    bytes += layer.b3.len() * std::mem::size_of::<f32>();
    bytes += layer.w4.nrows() * layer.w4.ncols() * std::mem::size_of::<f32>();
    bytes += layer.b4.len() * std::mem::size_of::<f32>();
    bytes
}
