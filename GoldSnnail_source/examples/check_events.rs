use goldworm::NmnistDataset;

fn main() {
    let dataset = NmnistDataset::load(10);
    println!("Checking event counts for first 20 samples...");
    for (i, sample) in dataset.train.iter().take(20).enumerate() {
        println!("  Sample {}: digit={}, events={}", i, sample.digit, sample.events.len());
    }
    
    let mut min_ts = u32::MAX;
    let mut max_ts = 0u32;
    for sample in &dataset.train {
        for e in &sample.events {
            if e.timestamp_us < min_ts { min_ts = e.timestamp_us; }
            if e.timestamp_us > max_ts { max_ts = e.timestamp_us; }
        }
    }
    println!("Timestamp range: {} - {} (delta={})", min_ts, max_ts, max_ts - min_ts);
}
