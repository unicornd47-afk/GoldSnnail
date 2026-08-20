//! Spiking Heidelberg Digits (SHD) Evaluation
//!
//! SHD: 700 Train / 200 Test / 20 Klassen
//! Daten: zenkelab.com/datasets
//!
//! Nutzt Hyperbolic k-NN im Poincaré-Ball (100D Rate-Coding Features).
//!
//! Umgebungsvariablen:
//!   SHD_DATA — Pfad zu shd.json (konvertiert aus HDF5)

use goldworm::audio::hyperbolic_knn::HyperbolicKnn;
use goldworm::audio::shd_loader::ShdDataset;
use goldworm::geometry::HyperbolicPoint;
use std::env;
use std::path::Path;

fn main() {
    let data_path = env::var("SHD_DATA").unwrap_or_else(|_| "data/shd/shd.json".to_string());

    let dataset = match ShdDataset::from_json(Path::new(&data_path)) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("SHD-Daten nicht gefunden: {}", e);
            eprintln!("Siehe docs/SHD.md für Konvertierung.");
            std::process::exit(1);
        }
    };

    println!(
        "SHD geladen: {} Train, {} Test, {} Neuronen, {}ms",
        dataset.train.len(),
        dataset.test.len(),
        dataset.num_neurons,
        dataset.duration_ms
    );

    // Trainings-Points im Hyperbolic-Space bilden
    let train_points: Vec<_> = dataset
        .train
        .iter()
        .map(|s| {
            let vec = ShdDataset::to_feature_vector(s, dataset.num_neurons, dataset.duration_ms);
            let point = HyperbolicPoint { coords: vec };
            (s.clone(), point)
        })
        .collect();

    let knn = HyperbolicKnn::new(5);
    let mut correct = 0;

    for test in &dataset.test {
        let vec = ShdDataset::to_feature_vector(test, dataset.num_neurons, dataset.duration_ms);
        let test_point = HyperbolicPoint { coords: vec };
        let pred = knn.classify(&train_points, &test_point);
        if pred == test.label {
            correct += 1;
        }
    }

    let accuracy = if !dataset.test.is_empty() {
        correct as f64 / dataset.test.len() as f64
    } else {
        0.0
    };

    println!("Benchmark: shd");
    println!("Accuracy: {}%", accuracy * 100.0);
    println!("Correct: {}/{}", correct, dataset.test.len());
    println!("Status: {}", if accuracy > 0.05 { "SCORED" } else { "BASELINE" });
}
