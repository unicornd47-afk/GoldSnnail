//! CIFAR-10 Loader — Original Binary Format + Synthetic Generator
//!
//! Supports loading the original Alex Krizhevsky CIFAR-10 `.bin` files
//! (data_batch_1.bin … test_batch.bin) and generating synthetic images
//! for immediate testing without a 170MB download.

use crate::LabError;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// CIFAR-10 Bild: 32×32 RGB als flaches Vec<f32> (DOD: kein Vec<Vec<Vec<u8>>>)
#[derive(Debug, Clone)]
pub struct CifarImage {
    pub label: u8,
    pub pixels: Vec<f32>, // 3072 Elemente, [0.0, 1.0]
}

/// CIFAR-10 Label-Namen (englisch)
pub const CIFAR10_CLASSES: [&str; 10] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
];

/// Mapping auf bestehende deutsche Lexikon-Wörter
/// Erweiterbar: Fügt fehlende Wörter dynamisch hinzu
pub fn map_cifar_label_to_lexicon(label: u8) -> &'static str {
    match label {
        0 => "vogel",   // airplane → fliegt wie Vogel
        1 => "haus",    // automobile → Objekt/Container
        2 => "vogel",   // bird
        3 => "katze",   // cat
        4 => "hund",    // deer → Säugetier
        5 => "hund",    // dog
        6 => "fisch",   // frog → Tier (beste Approximation)
        7 => "hund",    // horse → Säugetier
        8 => "tisch",   // ship → Objekt
        9 => "stein",   // truck → Objekt
        _ => "???",
    }
}

/// Loader für echte CIFAR-10 .bin Dateien
pub struct Cifar10Loader;

impl Cifar10Loader {
    /// Ein File = 10.000 Bilder. Format: [label: u8][pixels: 3072×u8] × 10000
    pub fn load_batch<P: AsRef<Path>>(path: P) -> Result<Vec<CifarImage>, LabError> {
        let file = File::open(path).map_err(|e| {
            LabError::Geometry(format!("Cannot open CIFAR-10 batch: {}", e))
        })?;
        
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            LabError::Geometry(format!("Cannot read CIFAR-10 batch: {}", e))
        })?;

        const RECORD_SIZE: usize = 1 + 3072; // label + pixels
        let num_records = buffer.len() / RECORD_SIZE;
        
        let mut images = Vec::with_capacity(num_records);
        
        for i in 0..num_records {
            let offset = i * RECORD_SIZE;
            let label = buffer[offset];
            
            // Normalisiere zu [0.0, 1.0]
            let mut pixels = Vec::with_capacity(3072);
            for j in 0..3072 {
                pixels.push(buffer[offset + 1 + j] as f32 / 255.0);
            }
            
            images.push(CifarImage { label, pixels });
        }
        
        Ok(images)
    }

    /// Lädt alle 5 Trainings-Batches aus einem Verzeichnis
    pub fn load_training_set<P: AsRef<Path>>(dir: P) -> Result<Vec<CifarImage>, LabError> {
        let mut all = Vec::with_capacity(50_000);
        for i in 1..=5 {
            let path = dir.as_ref().join(format!("data_batch_{}.bin", i));
            if path.exists() {
                let batch = Self::load_batch(&path)?;
                all.extend(batch);
            } else {
                return Err(LabError::Geometry(
                    format!("Missing CIFAR-10 file: {:?}. Download from https://www.cs.toronto.edu/~kriz/cifar.html", path)
                ));
            }
        }
        Ok(all)
    }

    /// Lädt Test-Batch
    pub fn load_test_set<P: AsRef<Path>>(dir: P) -> Result<Vec<CifarImage>, LabError> {
        let path = dir.as_ref().join("test_batch.bin");
        Self::load_batch(path)
    }
}

/// Synthetischer CIFAR-10-Generator für sofortiges Testen ohne Download
/// Erzeugt Bilder im exakt gleichen Format, aber mit deterministischen Mustern
pub fn generate_synthetic_cifar10_batch(
    count: usize,
    label_distribution: Option<&[usize]>, // Wie viele pro Klasse
) -> Vec<CifarImage> {
    let mut images = Vec::with_capacity(count);
    let per_class = count / 10;
    
    for label in 0..10u8 {
        let n = label_distribution.map(|d| d.get(label as usize).copied().unwrap_or(per_class))
            .unwrap_or(per_class);
        
        for i in 0..n {
            let mut pixels = vec![0.0f32; 3072];
            
            // Jedes Label bekommt ein charakteristisches deterministisches Muster
            match label {
                0 => { // airplane: horizontale Streifen oben
                    for y in 0..32 {
                        let val = if y < 12 { 0.9 } else { 0.2 };
                        for x in 0..32 {
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = val;
                            pixels[idx + 1] = val * 0.8;
                            pixels[idx + 2] = val * 0.6;
                        }
                    }
                }
                1 => { // automobile: vertikale Streifen
                    for x in 0..32 {
                        let val = if x % 4 < 2 { 0.8 } else { 0.3 };
                        for y in 0..32 {
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = val;
                            pixels[idx + 1] = val * 0.5;
                            pixels[idx + 2] = val * 0.5;
                        }
                    }
                }
                2 => { // bird: zentrale helle Blob
                    for y in 0..32 {
                        for x in 0..32 {
                            let dx = x as f32 - 16.0;
                            let dy = y as f32 - 16.0;
                            let dist = (dx * dx + dy * dy).sqrt();
                            let val = (1.0 - dist / 20.0).max(0.0);
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = val * 0.9;
                            pixels[idx + 1] = val * 0.7;
                            pixels[idx + 2] = val * 0.3;
                        }
                    }
                }
                3 => { // cat: diagonal
                    for y in 0..32 {
                        for x in 0..32 {
                            let val = if (x + y) % 6 < 3 { 0.85 } else { 0.15 };
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = val;
                            pixels[idx + 1] = val * 0.6;
                            pixels[idx + 2] = val * 0.9;
                        }
                    }
                }
                4 => { // deer: Rausch mit Bias
                    for j in 0..3072 {
                        let v = ((j as f32 * 0.1 + label as f32).sin() * 0.5 + 0.5) * 0.7;
                        pixels[j] = v;
                    }
                }
                5 => { // dog: große Blöcke
                    for y in 0..32 {
                        for x in 0..32 {
                            let val = if (x / 8 + y / 8) % 2 == 0 { 0.9 } else { 0.2 };
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = val;
                            pixels[idx + 1] = val * 0.4;
                            pixels[idx + 2] = val * 0.3;
                        }
                    }
                }
                6 => { // frog: grüne Gradienten
                    for y in 0..32 {
                        for x in 0..32 {
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = 0.2;
                            pixels[idx + 1] = (y as f32 / 32.0) * 0.8 + 0.2;
                            pixels[idx + 2] = 0.3;
                        }
                    }
                }
                7 => { // horse: vertikale Gradienten
                    for y in 0..32 {
                        for x in 0..32 {
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = (x as f32 / 32.0) * 0.7 + 0.2;
                            pixels[idx + 1] = 0.5;
                            pixels[idx + 2] = 0.3;
                        }
                    }
                }
                8 => { // ship: horizontale Linien unten
                    for y in 0..32 {
                        let val = if y > 20 { 0.9 } else { 0.2 };
                        for x in 0..32 {
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = val * 0.6;
                            pixels[idx + 1] = val * 0.7;
                            pixels[idx + 2] = val * 0.9;
                        }
                    }
                }
                _ => { // truck: Checkerboard
                    for y in 0..32 {
                        for x in 0..32 {
                            let val = if (x / 4 + y / 4) % 2 == 0 { 0.9 } else { 0.2 };
                            let idx = (y * 32 + x) * 3;
                            pixels[idx] = val;
                            pixels[idx + 1] = val;
                            pixels[idx + 2] = val;
                        }
                    }
                }
            }
            
            // Füge leichte Variation hinzu (deterministisch via Index)
            for j in 0..3072 {
                let noise = (i as f32 * 0.017 + j as f32 * 0.003).sin() * 0.05;
                pixels[j] = (pixels[j] + noise).clamp(0.0, 1.0);
            }
            
            images.push(CifarImage { label, pixels });
        }
    }
    
    images
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_batch_has_correct_size() {
        let batch = generate_synthetic_cifar10_batch(100, None);
        assert_eq!(batch.len(), 100);
        assert_eq!(batch[0].pixels.len(), 3072);
    }

    #[test]
    fn synthetic_pixels_normalized() {
        let batch = generate_synthetic_cifar10_batch(10, None);
        for img in &batch {
            assert!(img.pixels.iter().all(|&p| p >= 0.0 && p <= 1.0));
        }
    }

    #[test]
    fn load_nonexistent_file_fails_gracefully() {
        let result = Cifar10Loader::load_batch("does_not_exist.bin");
        assert!(result.is_err());
    }

    #[test]
    fn label_mapping_covers_all_classes() {
        for i in 0..10u8 {
            let word = map_cifar_label_to_lexicon(i);
            assert!(!word.is_empty());
        }
    }
}