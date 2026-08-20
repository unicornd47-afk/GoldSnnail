//! SHD-Daten-Generator für GoldSnnail
//!
//! Generiert synthetische SHD-ähnliche Spike-Trains und schreibt sie nach
//! `data/shd/shd.json` im erwarteten Format:
//!   { "train": [...], "test": [...], "num_neurons": 700, "duration_ms": 1000.0 }
//!
//! Nutzung:
//!   cargo run --example generate_shd_data --release

use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::Poisson;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShdSample {
    pub spikes: Vec<(f64, u32)>,
    pub label: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShdDataset {
    pub train: Vec<ShdSample>,
    pub test: Vec<ShdSample>,
    pub num_neurons: usize,
    pub duration_ms: f64,
}

fn generate_class_pattern(class: u32, rng: &mut impl Rng) -> Vec<f64> {
    // Erzeuge klassenspezifische "Spike-Muster"
    // Jede Klasse hat ein anderes Aktivitätsprofil über die 700 Neuronen
    let mut rates = vec![0.0f64; 700];
    let base_rate = 5.0 + (class as f64) * 2.0;
    
    // Bestimmte Neuronen-Gruppen sind für bestimmte Klassen aktiver
    let group_start = (class as usize * 35) % 650;
    for i in group_start..(group_start + 50).min(700) {
        rates[i] = base_rate + rng.gen_range(-2.0..2.0);
    }
    
    // Zufällige Hintergrundaktivität
    for i in 0..700 {
        if rates[i] == 0.0 {
            rates[i] = rng.gen_range(0.5..3.0);
        }
    }
    
    rates
}

fn generate_sample(class: u32, num_neurons: usize, duration_ms: f64, rng: &mut impl Rng) -> ShdSample {
    let poisson = Poisson::new(1.0).unwrap();
    let rates = generate_class_pattern(class, rng);
    let mut spikes = Vec::new();
    
    // Generiere Spike-Zeiten basierend auf Raten
    let dt = 1.0; // 1ms Zeitauflösung
    let num_bins = (duration_ms / dt) as usize;
    
    for t in 0..num_bins {
        for neuron in 0..num_neurons {
            let rate = rates[neuron] * dt / 1000.0; // spikes pro ms
            if rate > 0.0 && rng.gen_bool(rate.min(0.3)) {
                spikes.push((t as f64, neuron as u32));
            }
        }
    }
    
    ShdSample { spikes, label: class }
}

fn main() {
    println!("Generiere SHD-ähnliche Daten...");
    
    let num_neurons = 700;
    let duration_ms = 1000.0;
    let num_classes = 20;
    let train_per_class = 400;
    let test_per_class = 100;
    
    let mut rng = rand::thread_rng();
    let mut train = Vec::new();
    let mut test = Vec::new();
    
    for class in 0..num_classes {
        for _ in 0..train_per_class {
            train.push(generate_sample(class, num_neurons, duration_ms, &mut rng));
        }
        for _ in 0..test_per_class {
            test.push(generate_sample(class, num_neurons, duration_ms, &mut rng));
        }
    }
    
    // Shuffle
    train.shuffle(&mut rng);
    test.shuffle(&mut rng);
    
    let dataset = ShdDataset {
        train,
        test,
        num_neurons,
        duration_ms,
    };
    
    let output_dir = Path::new("data/shd");
    fs::create_dir_all(output_dir).expect("Konnte data/shd/ nicht erstellen");
    
    let output_path = output_dir.join("shd.json");
    let json = serde_json::to_string_pretty(&dataset).expect("Serialisierung fehlgeschlagen");
    fs::write(&output_path, json).expect("Konnte shd.json nicht schreiben");
    
    println!("\n[SUCCESS] {}", output_path.display());
    println!("  Train: {} Samples", dataset.train.len());
    println!("  Test:  {} Samples", dataset.test.len());
    println!("  Neurons: {}", dataset.num_neurons);
    println!("  Duration: {}ms", dataset.duration_ms);
    println!("\nJetzt ausführen:");
    println!("  cargo run --example eval_shd --release");
}
