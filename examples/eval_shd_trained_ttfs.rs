use goldsnnail::audio::shd_encoder::ShdEncoder;
use goldsnnail::audio::shd_loader::ShdDataset;
use goldsnnail::geometry::HyperbolicPoint;
use std::collections::HashMap;
use std::path::Path;

fn main() {
    let data_path = std::env::var("SHD_DATA").unwrap_or_else(|_| "data/shd/shd.json".to_string());
    let dataset = ShdDataset::from_json(Path::new(&data_path))
        .expect("SHD-Daten nicht gefunden.");

    let model_path = std::env::var("SHD_MODEL").unwrap_or_else(|_| "models/shd_encoder_ttfs_v0.1.bin".to_string());
    let model_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&model_path).expect("Modell nicht gefunden. Führe train_shd_encoder_ttfs aus.")
    ).unwrap();

    let mut encoder = ShdEncoder::new();
    encoder.w1 = model_json["w1"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    encoder.b1 = model_json["b1"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    encoder.w2 = model_json["w2"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    encoder.b2 = model_json["b2"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

    let train_points: Vec<_> = dataset.train.iter()
        .map(|s| {
            let vec = ShdDataset::to_feature_vector_ttfs(s, dataset.num_neurons, dataset.duration_ms);
            let emb = encoder.forward(&vec);
            (s.label, emb)
        })
        .collect();

    let mut correct = 0;
    let k = 5;

    for test in &dataset.test {
        let vec = ShdDataset::to_feature_vector_ttfs(test, dataset.num_neurons, dataset.duration_ms);
        let test_emb = encoder.forward(&vec);

        let mut neighbors: Vec<(f64, u32)> = train_points.iter()
            .map(|(label, emb)| {
                let dist = emb.iter().zip(test_emb.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (dist, *label)
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut votes = HashMap::new();
        for &(_, label) in neighbors.iter().take(k) {
            *votes.entry(label).or_insert(0usize) += 1;
        }
        let pred = votes.into_iter().max_by_key(|&(_, c)| c).map(|(l, _)| l).unwrap_or(0);

        if pred == test.label {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / dataset.test.len() as f64;
    println!("Benchmark: shd-trained-ttfs");
    println!("Accuracy: {}%", accuracy * 100.0);
    println!("Correct: {}/{}", correct, dataset.test.len());
    println!("Status: {}", if accuracy > 0.5 { "STRONG" } else { "BASELINE" });
}
