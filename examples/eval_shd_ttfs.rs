use goldsnnail::geometry::HyperbolicPoint;
use goldsnnail::audio::hyperbolic_knn::HyperbolicKnn;
use goldsnnail::audio::shd_loader::ShdDataset;
use std::path::Path;

fn main() {
    let data_path = std::env::var("SHD_DATA").unwrap_or_else(|_| "data/shd/shd.json".to_string());
    let dataset = ShdDataset::from_json(Path::new(&data_path))
        .expect("SHD-Daten nicht gefunden.");

    println!("SHD geladen: {} Train, {} Test", dataset.train.len(), dataset.test.len());

    // Trainings-Points mit TTFS
    let train_points: Vec<_> = dataset.train.iter()
        .map(|s| {
            let vec = ShdDataset::to_feature_vector_ttfs(s, dataset.num_neurons, dataset.duration_ms);
            let point = HyperbolicPoint::new(ndarray::Array1::from(vec)).unwrap();
            (s.clone(), point)
        })
        .collect();

    let knn = HyperbolicKnn::new(5);
    let mut correct = 0;

    for test in &dataset.test {
        let vec = ShdDataset::to_feature_vector_ttfs(test, dataset.num_neurons, dataset.duration_ms);
        let test_point = HyperbolicPoint::new(ndarray::Array1::from(vec)).unwrap();
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

    println!("Benchmark: shd-ttfs");
    println!("Accuracy: {}%", accuracy * 100.0);
    println!("Correct: {}/{}", correct, dataset.test.len());
    println!("Status: {}", if accuracy > 0.25 { "STRONG" } else { "BASELINE" });
}
