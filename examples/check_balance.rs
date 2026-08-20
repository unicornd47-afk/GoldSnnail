use goldworm::NmnistDataset;

fn main() {
    let dataset = NmnistDataset::load(500);
    println!("Available digits: {:?}", dataset.available_digits);
    println!("Train samples: {}", dataset.train.len());
    println!("Test samples: {}", dataset.test.len());
    
    for &digit in &dataset.available_digits {
        let train_count = dataset.train.iter().filter(|s| s.digit == digit).count();
        let test_count = dataset.test.iter().filter(|s| s.digit == digit).count();
        println!("  Digit {}: train={}, test={}", digit, train_count, test_count);
    }
}
