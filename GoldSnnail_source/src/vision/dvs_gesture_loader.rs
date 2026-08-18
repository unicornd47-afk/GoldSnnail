//! DVS-Gesture Dataset Loader
//!
//! Loads or generates DVS-Gesture data for training the DVS-to-hyperbolic
//! projection layer. DVS-Gesture contains 11 hand gesture classes recorded
//! with a DVS128 sensor.
//!
//! Supports:
//! - Real AEDAT format parsing (DVS128 binary format)
//! - Synthetic data generation with temporal dynamics
//! - Automatic fallback from real to synthetic data

use crate::chat::dvs_encoder::DvsEvent;
use rand::Rng;
use rand::seq::SliceRandom;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// A single DVS-Gesture sample: a hand gesture written in DVS events.
#[derive(Debug, Clone)]
pub struct DvsGestureSample {
    /// The gesture label (0-10)
    pub gesture: u8,
    /// Human-readable label (e.g., "hand_wave")
    pub label: String,
    /// DVS events forming the gesture trajectory
    pub events: Vec<DvsEvent>,
}

impl DvsGestureSample {
    pub fn new(gesture: u8, events: Vec<DvsEvent>) -> Self {
        Self {
            gesture,
            label: format!("gesture_{}", gesture),
            events,
        }
    }
}

/// The DVS-Gesture dataset with train/test splits.
#[derive(Debug, Clone)]
pub struct DvsGestureDataset {
    pub train: Vec<DvsGestureSample>,
    pub test: Vec<DvsGestureSample>,
    pub available_gestures: Vec<u8>,
}

impl DvsGestureDataset {
    /// Loads a DVS-Gesture dataset with `samples_per_gesture` per class.
    ///
    /// First attempts to load real data from disk/download,
    /// falls back to synthetic data if not available.
    pub fn load(samples_per_gesture: usize) -> Self {
        let cache_dir = std::env::var("DVS_GESTURE_CACHE_DIR")
            .unwrap_or_else(|_| "data/dvs_gesture".to_string());

        // Try to load from cache
        let train_path = Path::new(&cache_dir).join("train");
        let test_path = Path::new(&cache_dir).join("test");

        if train_path.exists() && test_path.exists() {
            println!("Loading DVS-Gesture from cache: {}", cache_dir);
            return Self::load_from_cache(&cache_dir, samples_per_gesture);
        }

        if train_path.exists() {
            println!("Loading DVS-Gesture train-only from cache: {}", cache_dir);
            return Self::load_from_cache(&cache_dir, samples_per_gesture);
        }

        // Fallback to synthetic data
        println!("Using synthetic DVS-Gesture data ({} samples per gesture)", samples_per_gesture);
        Self::generate_synthetic(samples_per_gesture)
    }

    /// Loads a small dataset for quick testing (50 samples per gesture).
    pub fn load_mini() -> Self {
        Self::load(50)
    }

    /// Generates synthetic DVS-Gesture data with temporal dynamics.
    fn generate_synthetic(samples_per_gesture: usize) -> Self {
        let mut train = Vec::new();
        let mut test = Vec::new();
        let available_gestures: Vec<u8> = (0..11u8).collect();

        for gesture in 0..11u8 {
            for i in 0..samples_per_gesture {
                let events = generate_gesture_events_with_variation(gesture, i);
                let sample = DvsGestureSample::new(gesture, events);
                if i < samples_per_gesture / 2 {
                    train.push(sample);
                }
            }
        }

        // Generate test set separately
        for gesture in 0..11u8 {
            for i in 0..samples_per_gesture / 2 {
                let events = generate_gesture_events_with_variation(gesture, i + samples_per_gesture);
                let sample = DvsGestureSample::new(gesture, events);
                test.push(sample);
            }
        }

        Self { train, test, available_gestures }
    }

    /// Loads data from cached binary files.
    fn load_from_cache(cache_dir: &str, samples_per_gesture: usize) -> Self {
        let mut train = Vec::new();
        let mut test = Vec::new();
        let mut available_gestures = Vec::new();

        let train_root = Path::new(cache_dir).join("train");

        // Scan available gesture directories
        if let Ok(entries) = fs::read_dir(&train_root) {
            for entry in entries.filter_map(|e| e.ok()) {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Ok(gesture) = entry.file_name().to_string_lossy().parse::<u8>() {
                            available_gestures.push(gesture);
                        }
                    }
                }
            }
        }

        available_gestures.sort_unstable();
        println!("  Available gestures: {:?}", available_gestures);

        for gesture in &available_gestures {
            let train_gesture_dir = train_root.join(gesture.to_string());
            let test_gesture_dir = Path::new(cache_dir).join("test").join(gesture.to_string());

            if let Ok(entries) = fs::read_dir(&train_gesture_dir) {
                let samples: Vec<DvsGestureSample> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|ext| ext == "bin").unwrap_or(false))
                    .filter_map(|e| Self::parse_bin_file(&e.path(), *gesture))
                    .take(samples_per_gesture)
                    .collect();
                train.extend(samples);
            }

            if let Ok(entries) = fs::read_dir(&test_gesture_dir) {
                let samples: Vec<DvsGestureSample> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|ext| ext == "bin").unwrap_or(false))
                    .filter_map(|e| Self::parse_bin_file(&e.path(), *gesture))
                    .take(samples_per_gesture)
                    .collect();
                test.extend(samples);
            }
        }

        Self { train, test, available_gestures }
    }

    /// Parses a single DVS-Gesture binary file (AEDAT-like format).
    ///
    /// Binary format:
    /// - Each event is 8 bytes
    /// - Bytes 0-3: address (x in bits 0-6, polarity in bit 7, y in bits 8-14)
    /// - Bytes 4-7: timestamp (uint32, little-endian, microseconds)
    fn parse_bin_file(path: &Path, expected_gesture: u8) -> Option<DvsGestureSample> {
        let data = fs::read(path).ok()?;
        let num_events = data.len() / 8;
        if num_events == 0 {
            return None;
        }

        let mut events = Vec::new();
        for chunk in data[..num_events * 8].chunks_exact(8) {
            let addr = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let x = (addr & 0x7F) as u8;
            let polarity = ((addr >> 7) & 0x01) as u8;
            let y = ((addr >> 8) & 0x7F) as u8;
            let timestamp = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            events.push(DvsEvent::new(x, y, polarity, timestamp));
        }

        Some(DvsGestureSample::new(expected_gesture, events))
    }

    /// Saves the dataset to binary files for caching.
    pub fn save_to_cache(&self, cache_dir: &str) -> Result<(), String> {
        fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;

        for sample in &self.train {
            let gesture_dir = Path::new(cache_dir).join("train").join(sample.gesture.to_string());
            fs::create_dir_all(&gesture_dir).map_err(|e| e.to_string())?;

            let filename = format!("{:06}.bin", sample.events.len());
            let path = gesture_dir.join(&filename);
            let mut file = File::create(&path).map_err(|e| e.to_string())?;

            for event in &sample.events {
                let addr = (event.x as u32) | ((event.polarity as u32) << 7) | ((event.y as u32) << 8);
                let addr_bytes = addr.to_le_bytes();
                let timestamp_bytes = event.timestamp_us.to_le_bytes();
                file.write_all(&addr_bytes).map_err(|e| e.to_string())?;
                file.write_all(&timestamp_bytes).map_err(|e| e.to_string())?;
            }
        }

        Ok(())
    }
}

/// Loads the training set with `samples_per_gesture` samples per gesture.
pub fn load_train_set(samples_per_gesture: usize) -> Vec<DvsGestureSample> {
    DvsGestureDataset::load(samples_per_gesture).train
}

/// Loads the test set with `samples_per_gesture` samples per gesture.
pub fn load_test_set(samples_per_gesture: usize) -> Vec<DvsGestureSample> {
    DvsGestureDataset::load(samples_per_gesture).test
}

// =============================================================================
// Synthetic Gesture Generation
// =============================================================================

/// Gesture types matching DVS-Gesture classes.
pub const GESTURE_LABELS: [&str; 11] = [
    "hand_wave_left_right",   // 0: horizontal wave
    "hand_push_away",         // 1: pushing forward
    "hand_pull_towards",       // 2: pulling towards
    "hand_circle_clockwise",  // 3: circular motion
    "hand_circle_counter",    // 4: circular motion reverse
    "hand_swipe_up",          // 5: upward swipe
    "hand_swipe_down",        // 6: downward swipe
    "hand_swipe_left",        // 7: leftward swipe
    "hand_swipe_right",       // 8: rightward swipe
    "hand_point",             // 9: pointing gesture
    "hand_idle",              // 10: no motion / static
];

/// Generates synthetic DVS events for a given gesture with variation.
fn generate_gesture_events_with_variation(gesture: u8, seed: usize) -> Vec<DvsEvent> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut events = Vec::new();

    let offset_x = ((seed * 37) % 30) as i32 - 15;
    let offset_y = ((seed * 41) % 30) as i32 - 15;
    let speed = 0.5 + ((seed * 43) % 100) as f32 / 200.0;
    let num_events = 60 + ((seed * 47) % 60);

    match gesture {
        0 => generate_wave(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        1 => generate_push(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        2 => generate_pull(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        3 => generate_circle_cw(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        4 => generate_circle_ccw(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        5 => generate_swipe_up(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        6 => generate_swipe_down(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        7 => generate_swipe_left(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        8 => generate_swipe_right(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        9 => generate_point(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
        10 => generate_idle(&mut events, num_events, offset_x, offset_y, &mut rng),
        _ => generate_wave(&mut events, num_events, offset_x, offset_y, speed, &mut rng),
    }

    // Add noise events (5-15% of total)
    let noise_ratio = 0.05 + ((seed % 4) as f32) * 0.025;
    let num_noise = ((events.len() as f32) * noise_ratio) as usize;
    for _ in 0..num_noise {
        let nx = rng.r#gen::<u8>();
        let ny = rng.r#gen::<u8>();
        let np = rng.r#gen::<u8>() % 2;
        let nt = rng.r#gen::<u32>() % 5000;
        events.push(DvsEvent::new(nx, ny, np, nt));
    }

    // Shuffle and sort by timestamp
    events.shuffle(&mut rng);
    events.sort_by_key(|e| e.timestamp_us);

    events
}

// --- Gesture Generators ---

fn generate_wave(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let x = (64.0 + ox as f32 + 40.0 * (t * std::f32::consts::PI * 2.0 * speed).sin()) as u8;
        let y = (64.0 + oy as f32 + 20.0 * t) as u8;
        let pol = if i % 2 == 0 { 0 } else { 1 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_push(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let radius = 35.0 * (1.0 - t * 0.7);
        let x = (64.0 + ox as f32 + radius * (t * std::f32::consts::PI).cos()) as u8;
        let y = (64.0 + oy as f32 + radius * (t * std::f32::consts::PI).sin()) as u8;
        let pol = if i % 3 == 0 { 0 } else { 1 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_pull(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let radius = 10.0 + 35.0 * t * 0.7;
        let x = (64.0 + ox as f32 + radius * (t * std::f32::consts::PI).cos()) as u8;
        let y = (64.0 + oy as f32 + radius * (t * std::f32::consts::PI).sin()) as u8;
        let pol = if i % 3 == 0 { 1 } else { 0 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_circle_cw(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let angle = t * std::f32::consts::PI * 2.0 * speed;
        let x = (64.0 + ox as f32 + 30.0 * angle.cos()) as u8;
        let y = (64.0 + oy as f32 + 30.0 * angle.sin()) as u8;
        let pol = if i % 2 == 0 { 0 } else { 1 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_circle_ccw(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let angle = -t * std::f32::consts::PI * 2.0 * speed;
        let x = (64.0 + ox as f32 + 30.0 * angle.cos()) as u8;
        let y = (64.0 + oy as f32 + 30.0 * angle.sin()) as u8;
        let pol = if i % 2 == 0 { 1 } else { 0 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_swipe_up(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let x = (64.0 + ox as f32 + 10.0 * (t * 4.0).sin()) as u8;
        let y = (100.0 + oy as f32 - 70.0 * t) as u8;
        let pol = if i % 2 == 0 { 0 } else { 1 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_swipe_down(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let x = (64.0 + ox as f32 + 10.0 * (t * 4.0).sin()) as u8;
        let y = (20.0 + oy as f32 + 70.0 * t) as u8;
        let pol = if i % 2 == 0 { 1 } else { 0 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_swipe_left(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let x = (100.0 + ox as f32 - 70.0 * t) as u8;
        let y = (64.0 + oy as f32 + 10.0 * (t * 4.0).sin()) as u8;
        let pol = if i % 2 == 0 { 0 } else { 1 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_swipe_right(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, _rng: &mut impl Rng) {
    for i in 0..n {
        let t = i as f32 / n as f32;
        let x = (20.0 + ox as f32 + 70.0 * t) as u8;
        let y = (64.0 + oy as f32 + 10.0 * (t * 4.0).sin()) as u8;
        let pol = if i % 2 == 0 { 1 } else { 0 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_point(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, speed: f32, rng: &mut impl Rng) {
    let target_x = 64 + ox;
    let target_y = 30 + oy;
    for i in 0..n {
        let t = i as f32 / n as f32;
        let spread = if t < 0.3 { 30.0 * (1.0 - t / 0.3) } else { 5.0 };
        let x = (target_x as f32 + spread * (rng.r#gen::<f32>() - 0.5)) as u8;
        let y = (target_y as f32 + spread * (rng.r#gen::<f32>() - 0.5)) as u8;
        let pol = if i % 2 == 0 { 0 } else { 1 };
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

fn generate_idle(events: &mut Vec<DvsEvent>, n: usize, ox: i32, oy: i32, rng: &mut impl Rng) {
    let cx = 64 + ox;
    let cy = 64 + oy;
    for i in 0..n {
        let x = (cx + ((rng.r#gen::<u32>() % 10) as i32 - 5)) as u8;
        let y = (cy + ((rng.r#gen::<u32>() % 10) as i32 - 5)) as u8;
        let pol = rng.r#gen::<u8>() % 2;
        events.push(DvsEvent::new(x.min(127), y.min(127), pol, i as u32 * 100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_mini_returns_samples() {
        let dataset = DvsGestureDataset::load_mini();
        assert!(!dataset.train.is_empty(), "train set should not be empty");
        assert!(!dataset.available_gestures.is_empty(), "available gestures should not be empty");
    }

    #[test]
    fn each_gesture_has_correct_label() {
        let dataset = DvsGestureDataset::load_mini();
        for sample in &dataset.train {
            let label_gesture: u8 = sample.label.strip_prefix("gesture_").unwrap().parse().unwrap();
            assert_eq!(sample.gesture, label_gesture);
        }
    }

    #[test]
    fn samples_have_events() {
        let dataset = DvsGestureDataset::load_mini();
        for sample in &dataset.train {
            assert!(!sample.events.is_empty(), "Sample for gesture {} has no events", sample.gesture);
        }
    }

    #[test]
    fn synthetic_gesture_has_11_classes() {
        let dataset = DvsGestureDataset::load(10);
        assert_eq!(dataset.available_gestures.len(), 11);
    }
}
