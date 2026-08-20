//! GoldWorm CIFAR-10 Similarity Evaluation
//!
//! Measures the discriminative power of the PatchEncoder by
//! comparing test images against a training cache in hyperbolic space.
//! Uses the trained weights with normalization and tanh compression.
//!
//! Usage:
//!   cargo run --example eval_cifar10_similarity

use goldworm::{
    PatchEncoder, generate_synthetic_cifar10_batch, Cifar10Loader,
    map_cifar_label_to_lexicon, HyperbolicPoint, PoincareBall,
};
use ndarray::Array1;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Ein gecachtes Trainings-Beispiel im Encoder-Raum
#[derive(Debug, Clone)]
struct EncodedSample {
    pub label: u8,
    pub label_word: String,
    /// Mittlerer Patch-Encoder-Output (repräsentativ für das Bild)
    pub centroid: HyperbolicPoint,
    /// Alle Patch-Embeddings
    pub patches: Vec<HyperbolicPoint>,
}

fn main() {
    println!("=== GoldWorm CIFAR-10 Similarity Evaluation ===\n");

    let encoder = PatchEncoder::new(8, 8, 1.0); // 8×8 patches, latent_dim=8, curvature=1.0
    let ball = PoincareBall::new(1.0);

    // === DATEN LADEN ===
    let cifar_dir = std::env::var("CIFAR10_DIR").unwrap_or_else(|_| "./cifar-10-batches-bin".to_string());
    
    let (train_images, test_images) = if Path::new(&cifar_dir).exists() {
        println!("Loading real CIFAR-10 from {}...", cifar_dir);
        let train = Cifar10Loader::load_training_set(&cifar_dir)
            .expect("Download from https://www.cs.toronto.edu/~kriz/cifar.html");
        let test = Cifar10Loader::load_test_set(&cifar_dir).unwrap();
        (train, test)
    } else {
        println!("Using synthetic CIFAR-10 (deterministic patterns)");
        let train = generate_synthetic_cifar10_batch(500, None);
        let test = generate_synthetic_cifar10_batch(100, None);
        (train, test)
    };

    // === TRAININGS-CACHE AUFBAUEN (mit Embeddings) ===
    println!("Encoding {} training images...", train_images.len());
    let mut train_encoded: Vec<EncodedSample> = Vec::with_capacity(train_images.len());

    for img in &train_images {
        let pixels_f64: Vec<f64> = img.pixels.iter().map(|&p| p as f64).collect();
        let tokens = encoder.encode_image_raw(&pixels_f64, 32, 32);
        
        // Centroid = Mittelwert aller Patch-HyperbolicPoints
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for t in &tokens {
            sum_x += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
            sum_y += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        }
        let n = tokens.len().max(1) as f64;
        let cx = sum_x / n;
        let cy = sum_y / n;
        let centroid = HyperbolicPoint::new(Array1::from_vec(vec![cx, cy])).unwrap_or_else(|_| {
            let norm = (cx * cx + cy * cy).sqrt();
            HyperbolicPoint::new(Array1::from_vec(vec![cx * 0.99 / norm.max(1e-12), cy * 0.99 / norm.max(1e-12)])).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] })
        });

        let patches: Vec<HyperbolicPoint> = tokens.into_iter().map(|t| t.hyperbolic).collect();

        train_encoded.push(EncodedSample {
            label: img.label,
            label_word: map_cifar_label_to_lexicon(img.label).to_string(),
            centroid,
            patches,
        });
    }

    // === EVALUATION: TOP-K SIMILARITY ===
    let k = 5;
    let test_samples = test_images.len().min(200);

    println!("\nEvaluating {} test images (Top-{} nearest neighbors)...", test_samples, k);

    let mut correct_top1 = 0;
    let mut correct_top5 = 0;
    let mut intra_class_dists: Vec<f64> = Vec::new();
    let mut inter_class_dists: Vec<f64> = Vec::new();

    // Für Similarity-Matrix: 10×10 Matrix [true_label][predicted_label] = count
    let mut confusion_counts: HashMap<(u8, u8), usize> = HashMap::new();

    for (idx, img) in test_images.iter().take(test_samples).enumerate() {
        let pixels_f64: Vec<f64> = img.pixels.iter().map(|&p| p as f64).collect();
        let test_tokens = encoder.encode_image_raw(&pixels_f64, 32, 32);
        
        let mut test_sum_x = 0.0;
        let mut test_sum_y = 0.0;
        for t in &test_tokens {
            test_sum_x += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
            test_sum_y += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        }
        let n = test_tokens.len().max(1) as f64;
        let test_centroid = HyperbolicPoint::new(Array1::from_vec(vec![test_sum_x / n, test_sum_y / n]))
            .unwrap_or_else(|_| {
                let norm = (test_sum_x * test_sum_x + test_sum_y * test_sum_y).sqrt();
                HyperbolicPoint::new(Array1::from_vec(vec![test_sum_x * 0.99 / norm.max(1e-12), test_sum_y * 0.99 / norm.max(1e-12)])).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] })
            });

        let true_label = img.label;
        let true_word = map_cifar_label_to_lexicon(true_label);

        // Finde k nächste Nachbarn im Trainings-Cache
        let mut neighbors: Vec<(usize, f64)> = train_encoded.iter().enumerate()
            .map(|(i, sample)| {
                let d = ball.distance(&test_centroid, &sample.centroid).unwrap_or(f64::INFINITY);
                (i, d)
            })
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let top_k = &neighbors[..k.min(neighbors.len())];
        let top1_label = train_encoded[top_k[0].0].label;
        let top1_dist = top_k[0].1;

        // Top-1 Accuracy (im Encoder-Raum!)
        if top1_label == true_label {
            correct_top1 += 1;
        }

        // Top-5 Accuracy
        if top_k.iter().any(|(i, _)| train_encoded[*i].label == true_label) {
            correct_top5 += 1;
        }

        // Distanz-Statistik
        for (train_idx, dist) in top_k {
            if train_encoded[*train_idx].label == true_label {
                intra_class_dists.push(*dist);
            } else {
                inter_class_dists.push(*dist);
            }
        }

        // Confusion-Matrix
        *confusion_counts.entry((true_label, top1_label)).or_default() += 1;

        if idx < 5 {
            println!("  Test[{}] (label={}): Top1={} (dist={:.4}), Top5 labels={:?}",
                idx,
                true_word,
                train_encoded[top_k[0].0].label_word,
                top1_dist,
                top_k.iter().map(|(i, _)| &train_encoded[*i].label_word).collect::<Vec<_>>()
            );
        }
    }

    // === STATISTIK ===
    let top1_acc = 100.0 * correct_top1 as f64 / test_samples as f64;
    let top5_acc = 100.0 * correct_top5 as f64 / test_samples as f64;

    let avg_intra = if !intra_class_dists.is_empty() {
        intra_class_dists.iter().sum::<f64>() / intra_class_dists.len() as f64
    } else { 0.0 };
    
    let avg_inter = if !inter_class_dists.is_empty() {
        inter_class_dists.iter().sum::<f64>() / inter_class_dists.len() as f64
    } else { 0.0 };

    println!("\n=== Results ===");
    println!("Top-1 Accuracy (Encoder-Similarity): {:.1}%", top1_acc);
    println!("Top-5 Accuracy (Encoder-Similarity): {:.1}%", top5_acc);
    println!("Avg Intra-Class Distance:  {:.4}", avg_intra);
    println!("Avg Inter-Class Distance:  {:.4}", avg_inter);
    println!("Separation Ratio:          {:.2}x", avg_inter / avg_intra.max(1e-12));

    if avg_inter > avg_intra * 1.5 {
        println!("✅ Encoder discriminates: same-class images are significantly closer");
    } else if avg_inter > avg_intra {
        println!("🟡 Weak discrimination: some signal, but noisy");
    } else {
        println!("❌ No discrimination: random encoder or collapsed embeddings");
    }

    // === EXPORT FÜR HTML-DEMO ===
    export_similarity_matrix(&train_encoded, &confusion_counts, top1_acc, top5_acc, avg_intra, avg_inter);
    println!("\nExported similarity data to docs/src/development/similarity_matrix.json");
    println!("Open docs/src/development/agi demo.html to view.");
}

fn export_similarity_matrix(
    train_encoded: &[EncodedSample],
    confusion: &HashMap<(u8, u8), usize>,
    top1: f64,
    top5: f64,
    avg_intra: f64,
    avg_inter: f64,
) {
    // 10×10 Similarity-Matrix (avg distance zwischen Klasse i und Klasse j)
    let mut matrix: [[f64; 10]; 10] = [[0.0; 10]; 10];
    let mut counts: [[usize; 10]; 10] = [[0; 10]; 10];

    for a in train_encoded {
        for b in train_encoded {
            if a.label < 10 && b.label < 10 {
                let d = (a.centroid.coords[0] - b.centroid.coords[0]).powi(2)
                    + (a.centroid.coords[1] - b.centroid.coords[1]).powi(2);
                matrix[a.label as usize][b.label as usize] += d.sqrt();
                counts[a.label as usize][b.label as usize] += 1;
            }
        }
    }

    for i in 0..10 {
        for j in 0..10 {
            if counts[i][j] > 0 {
                matrix[i][j] /= counts[i][j] as f64;
            }
        }
    }

    let mut json = String::from("{\n");
    json.push_str(&format!("  \"top1_accuracy\": {:.2},\n", top1));
    json.push_str(&format!("  \"top5_accuracy\": {:.2},\n", top5));
    json.push_str(&format!("  \"avg_intra_class\": {:.6},\n", avg_intra));
    json.push_str(&format!("  \"avg_inter_class\": {:.6},\n", avg_inter));
    json.push_str("  \"similarity_matrix\": [\n");
    for i in 0..10 {
        json.push_str("    [");
        for j in 0..10 {
            json.push_str(&format!("{:.6}", matrix[i][j]));
            if j < 9 { json.push_str(", "); }
        }
        json.push_str("]");
        if i < 9 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");
    json.push_str("  \"classes\": [\"airplane\", \"automobile\", \"bird\", \"cat\", \"deer\", \"dog\", \"frog\", \"horse\", \"ship\", \"truck\"],\n");
    json.push_str("  \"confusion\": [\n");
    let mut first = true;
    for ((true_l, pred_l), count) in confusion {
        if !first { json.push_str(",\n"); }
        json.push_str(&format!("    {{\"true\": {}, \"pred\": {}, \"count\": {}}}", true_l, pred_l, count));
        first = false;
    }
    json.push_str("\n  ]\n}");

    let path = "docs/src/development/similarity_matrix.json";
    fs::create_dir_all("docs/src/development").unwrap();
    let mut file = fs::File::create(path).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}
