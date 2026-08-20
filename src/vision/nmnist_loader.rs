//! N-MNIST Dataset Loader
//!
//! Loads or generates N-MNIST (Neuromorphic MNIST) data for training
//! the DVS-to-hyperbolic projection layer.
//!
//! Supports:
//! - Synthetic data generation (1000+ samples per digit)
//! - Real N-MNIST binary format parsing
//! - Automatic fallback from real to synthetic data

use crate::chat::dvs_encoder::DvsEvent;
use rand::Rng;
use rand::seq::SliceRandom;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

/// A single N-MNIST sample: a digit written in DVS events.
#[derive(Debug, Clone)]
pub struct NmnistSample {
    /// The digit label (0-9)
    pub digit: u8,
    /// Human-readable label (e.g., "digit_3")
    pub label: String,
    /// DVS events forming the digit trajectory
    pub events: Vec<DvsEvent>,
}

impl NmnistSample {
    pub fn new(digit: u8, events: Vec<DvsEvent>) -> Self {
        Self {
            digit,
            label: format!("digit_{}", digit),
            events,
        }
    }
}

/// The N-MNIST dataset with train/test splits.
#[derive(Debug, Clone)]
pub struct NmnistDataset {
    pub train: Vec<NmnistSample>,
    pub test: Vec<NmnistSample>,
    pub available_digits: Vec<u8>,
}

impl NmnistDataset {
    /// Loads an N-MNIST dataset with `samples_per_digit` per class.
    /// 
    /// First attempts to load real data from disk/download,
    /// falls back to synthetic data if not available.
    /// Supports partial datasets (only some digits present).
    pub fn load(samples_per_digit: usize) -> Self {
        let cache_dir = std::env::var("NMNIST_CACHE_DIR")
            .unwrap_or_else(|_| "data/nmnis_t".to_string());
        let train_path = Path::new(&cache_dir).join("train");
        let test_path = Path::new(&cache_dir).join("test");
        
        if train_path.exists() && test_path.exists() {
            println!("Loading N-MNIST from cache: {}", cache_dir);
            return Self::load_from_cache(&cache_dir, samples_per_digit);
        }
        
        if train_path.exists() {
            println!("Loading N-MNIST train-only from cache: {}", cache_dir);
            return Self::load_from_cache(&cache_dir, samples_per_digit);
        }
        
        // Try to download real data
        if let Ok(data_dir) = Self::download_and_extract() {
            println!("Using downloaded N-MNIST data");
            return Self::load_from_cache(&data_dir, samples_per_digit);
        }
        
        // Fallback to synthetic data
        println!("Using synthetic N-MNIST data ({} samples per digit)", samples_per_digit);
        Self::generate_synthetic(samples_per_digit)
    }
    
    /// Loads a small dataset for quick testing (100 samples per digit).
    pub fn load_mini() -> Self {
        Self::load(100)
    }
    
    /// Generates synthetic N-MNIST data with variation.
    fn generate_synthetic(samples_per_digit: usize) -> Self {
        let mut train = Vec::new();
        let mut test = Vec::new();
        let available_digits: Vec<u8> = (0..10u8).collect();
        
        for digit in 0..10u8 {
            for i in 0..samples_per_digit {
                let events = generate_digit_events_with_variation(digit, i);
                let sample = NmnistSample::new(digit, events);
                if i < samples_per_digit / 2 {
                    train.push(sample);
                }
            }
        }
        
        // Generate test set separately
        for digit in 0..10u8 {
            for i in 0..samples_per_digit / 2 {
                let events = generate_digit_events_with_variation(digit, i + samples_per_digit);
                let sample = NmnistSample::new(digit, events);
                test.push(sample);
            }
        }
        
        Self { train, test, available_digits }
    }
    
    /// Loads data from cached binary files.
    /// Supports partial datasets where only some digit directories exist.
    fn load_from_cache(cache_dir: &str, samples_per_digit: usize) -> Self {
        let mut train = Vec::new();
        let mut test = Vec::new();
        let mut available_digits = Vec::new();
        
        let train_root = Path::new(cache_dir).join("train");
        
        // Scan available digit directories
        if let Ok(entries) = fs::read_dir(&train_root) {
            for entry in entries.filter_map(|e| e.ok()) {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Ok(digit) = entry.file_name().to_string_lossy().parse::<u8>() {
                            available_digits.push(digit);
                        }
                    }
                }
            }
        }
        
        available_digits.sort_unstable();
        println!("  Available digits: {:?}", available_digits);
        
        for digit in &available_digits {
            let train_digit_dir = train_root.join(digit.to_string());
            let test_digit_dir = Path::new(cache_dir).join("test").join(digit.to_string());
            
            if let Ok(entries) = fs::read_dir(&train_digit_dir) {
                let samples: Vec<NmnistSample> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|ext| ext == "bin").unwrap_or(false))
                    .filter_map(|e| Self::parse_bin_file(&e.path(), *digit))
                    .take(samples_per_digit)
                    .collect();
                train.extend(samples);
            }
            
            if let Ok(entries) = fs::read_dir(&test_digit_dir) {
                let samples: Vec<NmnistSample> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|ext| ext == "bin").unwrap_or(false))
                    .filter_map(|e| Self::parse_bin_file(&e.path(), *digit))
                    .take(samples_per_digit)
                    .collect();
                test.extend(samples);
            }
        }
        
        Self { train, test, available_digits }
    }
    
    /// Parses a single N-MNIST binary file.
    /// 
    /// Binary format:
    /// - Each event is 8 bytes
    /// - Byte 0: x (0-127)
    /// - Byte 1: y (0-127)  
    /// - Byte 2: polarity (128=ON, 0=OFF in real N-MNIST)
    /// - Byte 3: padding/unused
    /// - Bytes 4-7: timestamp (uint32, big-endian, microseconds)
    fn parse_bin_file(path: &Path, expected_digit: u8) -> Option<NmnistSample> {
        let data = fs::read(path).ok()?;
        let num_events = data.len() / 8;
        if num_events == 0 {
            return None;
        }
        
        let mut events = Vec::new();
        for chunk in data[..num_events * 8].chunks_exact(8) {
            let x = chunk[0];
            let y = chunk[1];
            let polarity = if chunk[2] > 0 { 0 } else { 1 };
            let timestamp = u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            events.push(DvsEvent::new(x, y, polarity, timestamp));
        }
        
        Some(NmnistSample::new(expected_digit, events))
    }
    
    /// Downloads and extracts the N-MNIST dataset.
    /// Returns the path to the extracted data directory, or an error.
    fn download_and_extract() -> Result<String, String> {
        let cache_dir = std::env::var("NMNIST_CACHE_DIR")
            .unwrap_or_else(|_| "data/nmnis_t".to_string());
        // Updated working URLs from Tonic library / Mendeley direct links
        let train_url = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded";
        let test_url = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded";
        
        fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
        
        println!("Downloading N-MNIST dataset (this may take a minute)...");
        
        let train_zip = download_file(train_url)?;
        extract_zip(&train_zip, &cache_dir)?;
        
        let test_zip = download_file(test_url)?;
        extract_zip(&test_zip, &cache_dir)?;
        
        println!("N-MNIST dataset downloaded and extracted to {}", cache_dir);
        Ok(cache_dir)
    }
    
    /// Saves the dataset to binary files for caching.
    pub fn save_to_cache(&self, cache_dir: &str) -> Result<(), String> {
        fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;
        
        for sample in &self.train {
            let digit_dir = Path::new(cache_dir).join("train").join(sample.digit.to_string());
            fs::create_dir_all(&digit_dir).map_err(|e| e.to_string())?;
            
            let filename = format!("{:06}.bin", sample.events.len());
            let path = digit_dir.join(&filename);
            let mut file = File::create(&path).map_err(|e| e.to_string())?;
            
            for event in &sample.events {
                let timestamp_bytes = event.timestamp_us.to_be_bytes();
                file.write_all(&[event.x, event.y, event.polarity, 0]).map_err(|e| e.to_string())?;
                file.write_all(&timestamp_bytes).map_err(|e| e.to_string())?;
            }
        }
        
        Ok(())
    }
}

/// Downloads a file from a URL.
#[cfg(feature = "nmnis_t_download")]
fn download_file(url: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;
    
    println!("  Downloading {}...", url);
    let mut response = reqwest::blocking::get(url)
        .map_err(|e| format!("Failed to download {}: {}", url, e))?;
    
    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()));
    }
    
    let mut data = Vec::new();
    response.read_to_end(&mut data)
        .map_err(|e| format!("Failed to read response: {}", e))?;
    
    println!("  Downloaded {} bytes", data.len());
    Ok(data)
}

/// Downloads a file from a URL (stub without feature).
#[cfg(not(feature = "nmnis_t_download"))]
fn download_file(_url: &str) -> Result<Vec<u8>, String> {
    Err("N-MNIST download requires 'nmnis_t_download' feature".to_string())
}

/// Extracts a ZIP archive to a directory.
#[cfg(feature = "nmnis_t_download")]
fn extract_zip(zip_data: &[u8], target_dir: &str) -> Result<(), String> {
    use std::io::Cursor;
    
    println!("  Extracting to {}...", target_dir);
    let reader = Cursor::new(zip_data);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| format!("Failed to open ZIP: {}", e))?;
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)
            .map_err(|e| format!("Failed to read ZIP entry: {}", e))?;
        let outpath = Path::new(target_dir).join(file.name());
        
        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath).map_err(|e| e.to_string())?;
        } else {
            if let Some(p) = outpath.parent() {
                fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
            let mut outfile = File::create(&outpath).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut outfile).map_err(|e| e.to_string())?;
        }
    }
    
    Ok(())
}

/// Extracts a ZIP archive (stub without feature).
#[cfg(not(feature = "nmnis_t_download"))]
fn extract_zip(_zip_data: &[u8], _target_dir: &str) -> Result<(), String> {
    Err("N-MNIST extraction requires 'nmnis_t_download' feature".to_string())
}

/// Generates synthetic DVS events for a given digit with variation.
fn generate_digit_events_with_variation(digit: u8, seed: usize) -> Vec<DvsEvent> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut events = Vec::new();
    
    // Increased variation for harder 10-digit scalability test
    let offset_x = ((seed * 37) % 30) as i32 - 15;
    let offset_y = ((seed * 41) % 30) as i32 - 15;
    let scale = 0.7 + ((seed * 43) % 60) as f32 / 100.0;
    let rotation = ((seed * 47) % 360) as f32 * std::f32::consts::PI / 180.0;
    let polarity_flip = (seed % 3) == 0;
    let noise_ratio = 0.1 + ((seed % 5) as f32) * 0.05; // 10-30% noise
    
    match digit {
        0 => {
            for i in 0..80 {
                let angle = (i as f32 / 80.0) * std::f32::consts::PI * 2.0 + rotation;
                let x = (64.0 + offset_x as f32 + 30.0 * scale * angle.cos()) as u8;
                let y = (64.0 + offset_y as f32 + 30.0 * scale * angle.sin()) as u8;
                let pol = if polarity_flip { (i % 2) as u8 } else { ((i + 1) % 2) as u8 };
                events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
            }
        },
        1 => {
            for i in 0..80 {
                let x = (64 + offset_x + (i % 7) - 3) as u8;
                let y = (20 + offset_y + i * 2).min(127) as u8;
                let pol = if polarity_flip { 1 } else { 0 };
                events.push(DvsEvent::new(x, y, pol, i as u32 * 100));
            }
        },
        2 => {
            for i in 0..80 {
                let t = i as f32 / 80.0;
                let x = (100.0 + offset_x as f32 - t * 80.0 * scale) as u8;
                let y = (20.0 + offset_y as f32 + t * 100.0 * scale + 20.0 * (t * std::f32::consts::PI).sin()) as u8;
                let pol = if polarity_flip { (i % 2) as u8 } else { ((i + 1) % 2) as u8 };
                events.push(DvsEvent::new(x.max(0).min(127), y.max(0).min(127), pol, i as u32 * 100));
            }
        },
        3 => {
            for i in 0..80 {
                let t = i as f32 / 80.0;
                let x = (20.0 + offset_x as f32 + t * 80.0 * scale) as u8;
                let y = (20.0 + offset_y as f32 + t * 100.0 * scale + 20.0 * (t * std::f32::consts::PI).sin()) as u8;
                let pol = if polarity_flip { (i % 2) as u8 } else { ((i + 1) % 2) as u8 };
                events.push(DvsEvent::new(x.max(0).min(127), y.max(0).min(127), pol, i as u32 * 100));
            }
        },
        4 => {
            for i in 0..45 {
                let x = (50 + offset_x + (i % 7) - 3) as u8;
                let y = (20 + offset_y + i * 2).min(127) as u8;
                events.push(DvsEvent::new(x, y, 0, i as u32 * 100));
            }
            for i in 0..35 {
                let x = (30 + offset_x + i * 4) as u8;
                let y = (80 + offset_y + (i % 3) - 1).min(127) as u8;
                events.push(DvsEvent::new(x, y.min(127), 1, (45 + i) as u32 * 100));
            }
        },
        5 => {
            for i in 0..40 {
                let x = (20 + offset_x + i * 4) as u8;
                let y = (30 + offset_y + (i % 7) - 3).min(127) as u8;
                events.push(DvsEvent::new(x, y.min(127), 0, i as u32 * 100));
            }
            for i in 0..40 {
                let x = (100 + offset_x - i * 3) as u8;
                let y = (30.0 + offset_y as f32 + (40 - i) as f32 + 10.0 * ((i as f32 / 40.0) * std::f32::consts::PI).sin()) as u8;
                let pol = if polarity_flip { 0 } else { 1 };
                events.push(DvsEvent::new(x.max(0).min(127), y.max(0).min(127), pol, (40 + i) as u32 * 100));
            }
        },
        6 => {
            for i in 0..70 {
                let angle = (i as f32 / 70.0) * std::f32::consts::PI * 2.0 + rotation;
                let x = (64.0 + offset_x as f32 + 25.0 * scale * angle.cos()) as u8;
                let y = (64.0 + offset_y as f32 + 25.0 * scale * angle.sin()) as u8;
                events.push(DvsEvent::new(x.min(127), y.min(127), 0, i as u32 * 100));
            }
            for i in 0..15 {
                let x = (40 + offset_x + i * 2) as u8;
                let y = (64 + offset_y + (i % 3) - 1).min(127) as u8;
                events.push(DvsEvent::new(x, y.min(127), 1, (70 + i) as u32 * 100));
            }
        },
        7 => {
            for i in 0..80 {
                let x = (100 + offset_x - i * 2) as u8;
                let y = (20 + offset_y + i * 2).min(127) as u8;
                let pol = if polarity_flip { (i % 2) as u8 } else { ((i + 1) % 2) as u8 };
                events.push(DvsEvent::new(x.max(0).min(127), y.min(127), pol, i as u32 * 100));
            }
        },
        8 => {
            for i in 0..45 {
                let angle = (i as f32 / 45.0) * std::f32::consts::PI * 2.0 + rotation;
                let x1 = (50.0 + offset_x as f32 + 20.0 * scale * angle.cos()) as u8;
                let y1 = (40.0 + offset_y as f32 + 20.0 * scale * angle.sin()) as u8;
                events.push(DvsEvent::new(x1.min(127), y1.min(127), 0, i as u32 * 100));
            }
            for i in 0..45 {
                let angle = (i as f32 / 45.0) * std::f32::consts::PI * 2.0 + rotation;
                let x2 = (80.0 + offset_x as f32 + 15.0 * scale * angle.cos()) as u8;
                let y2 = (90.0 + offset_y as f32 + 15.0 * scale * angle.sin()) as u8;
                events.push(DvsEvent::new(x2.min(127), y2.min(127), 1, (45 + i) as u32 * 100));
            }
        },
        9 => {
            for i in 0..70 {
                let angle = (i as f32 / 70.0) * std::f32::consts::PI * 2.0 + rotation;
                let x = (64.0 + offset_x as f32 + 25.0 * scale * angle.cos()) as u8;
                let y = (64.0 + offset_y as f32 + 25.0 * scale * angle.sin()) as u8;
                events.push(DvsEvent::new(x.min(127), y.min(127), (i % 2) as u8, i as u32 * 100));
            }
            for i in 0..15 {
                let angle = (i as f32 / 15.0) * std::f32::consts::PI * 2.0 + rotation;
                let x = (64.0 + offset_x as f32 + 10.0 * scale * angle.cos()) as u8;
                let y = (64.0 + offset_y as f32 + 10.0 * scale * angle.sin()) as u8;
                events.push(DvsEvent::new(x.min(127), y.min(127), 0, (70 + i) as u32 * 100));
            }
        },
        _ => {}
    }
    
    // Add noise events (10-30% of total)
    let num_noise = ((events.len() as f32) * noise_ratio) as usize;
    for _ in 0..num_noise {
        let nx = rng.r#gen::<u8>();
        let ny = rng.r#gen::<u8>();
        let np = rng.r#gen::<u8>() % 2;
        let nt = rng.r#gen::<u32>() % 10000;
        events.push(DvsEvent::new(nx, ny, np, nt));
    }
    
    // Shuffle events to simulate realistic timing
    events.shuffle(&mut rng);
    
    // Sort by timestamp after shuffling
    events.sort_by_key(|e| e.timestamp_us);
    
    events
}

/// Loads the training set with `samples_per_digit` samples per digit.
pub fn load_train_set(samples_per_digit: usize) -> Vec<NmnistSample> {
    NmnistDataset::load(samples_per_digit).train
}

/// Loads the test set with `samples_per_digit` samples per digit.
pub fn load_test_set(samples_per_digit: usize) -> Vec<NmnistSample> {
    NmnistDataset::load(samples_per_digit).test
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn load_mini_returns_samples() {
        let dataset = NmnistDataset::load_mini();
        assert!(!dataset.train.is_empty(), "train set should not be empty");
        assert!(!dataset.available_digits.is_empty(), "available_digits should not be empty");
    }
    
    #[test]
    fn each_digit_has_correct_label() {
        let dataset = NmnistDataset::load_mini();
        for sample in &dataset.train {
            let label_digit: u8 = sample.label.strip_prefix("digit_").unwrap().parse().unwrap();
            assert_eq!(sample.digit, label_digit);
        }
    }
    
    #[test]
    fn samples_have_events() {
        let dataset = NmnistDataset::load_mini();
        for sample in &dataset.train {
            assert!(!sample.events.is_empty(), "Sample for digit {} has no events", sample.digit);
        }
    }
    
    #[test]
    #[ignore = "requires uncommitted N-MNIST data cache (data/ not in repo)"]
    fn large_dataset_loads() {
        let dataset = NmnistDataset::load(100);
        assert!(!dataset.train.is_empty(), "train set should not be empty");
        assert_eq!(dataset.available_digits, vec![3, 4, 9]);
    }
}
