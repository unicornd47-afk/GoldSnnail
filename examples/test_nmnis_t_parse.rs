use goldsnnail::{NmnistDataset, load_train_set, load_test_set};

fn main() {
    let dataset = NmnistDataset::load(10);
    println!("Train samples: {}", dataset.train.len());
    println!("Test samples: {}", dataset.test.len());

    if let Some(sample) = dataset.train.first() {
        println!("First sample: digit={}, events={}", sample.digit, sample.events.len());
        if let Some(event) = sample.events.first() {
            println!("First event: x={}, y={}, polarity={}, timestamp={}",
                event.x, event.y, event.polarity, event.timestamp_us);
        }
    }
}
