use goldsnnail::audio::shd_encoder::ShdEncoder;
use goldsnnail::audio::shd_loader::ShdDataset;
use std::path::Path;
use rand::seq::SliceRandom;

fn main() {
    let data_path = std::env::var("SHD_DATA").unwrap_or_else(|_| "data/shd/shd.json".to_string());
    let dataset = ShdDataset::from_json(Path::new(&data_path))
        .expect("SHD-Daten nicht gefunden.");

    println!("SHD Train: {} Samples, {} Klassen", dataset.train.len(), dataset.num_classes);

    let mut encoder = ShdEncoder::new();
    let epochs = 100;
    let lr = 0.001;
    let pairs_per_epoch = 2000;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for _ in 0..pairs_per_epoch {
            let same_class = rand::random::<bool>();
            
            let (idx_a, idx_b) = if same_class {
                let class = (rand::random::<u32>() % dataset.num_classes as u32) as u32;
                let class_samples: Vec<usize> = dataset.train.iter()
                    .enumerate()
                    .filter(|(_, s)| s.label == class)
                    .map(|(i, _)| i)
                    .collect();
                if class_samples.len() < 2 { continue; }
                let a = *class_samples.choose(&mut rand::thread_rng()).unwrap();
                let b = *class_samples.choose(&mut rand::thread_rng()).unwrap();
                (a, b)
            } else {
                let a = rand::random::<usize>() % dataset.train.len();
                let mut b = rand::random::<usize>() % dataset.train.len();
                while dataset.train[a].label == dataset.train[b].label {
                    b = rand::random::<usize>() % dataset.train.len();
                }
                (a, b)
            };

            // TTFS statt Rate-Coding
            let vec_a = ShdDataset::to_feature_vector_ttfs(
                &dataset.train[idx_a], dataset.num_neurons, dataset.duration_ms
            );
            let vec_b = ShdDataset::to_feature_vector_ttfs(
                &dataset.train[idx_b], dataset.num_neurons, dataset.duration_ms
            );

            let loss = encoder.train_step(&vec_a, &vec_b, same_class, lr);
            total_loss += loss;
        }

        if epoch % 10 == 0 {
            println!("Epoch {}: Avg Loss = {:.6}", epoch, total_loss / pairs_per_epoch as f64);
        }
    }

    let model_dir = std::path::PathBuf::from("models");
    std::fs::create_dir_all(&model_dir).unwrap();
    let model_path = model_dir.join("shd_encoder_ttfs_v0.1.bin");
    
    let model_data = serde_json::json!({
        "w1": encoder.w1,
        "b1": encoder.b1,
        "w2": encoder.w2,
        "b2": encoder.b2,
        "target_radius": encoder.target_radius,
    });
    std::fs::write(&model_path, serde_json::to_string_pretty(&model_data).unwrap())
        .expect("Konnte Modell nicht speichern");

    println!("Modell gespeichert: {}", model_path.display());
    println!("Training abgeschlossen.");
}
