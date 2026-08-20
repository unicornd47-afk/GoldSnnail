use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ShdSample {
    pub spikes: Vec<(f64, u32)>, // (time_ms, neuron_id)
    pub label: u32,
}

#[derive(Debug, Deserialize)]
pub struct ShdDataset {
    pub train: Vec<ShdSample>,
    pub test: Vec<ShdSample>,
    pub num_neurons: usize,
    pub duration_ms: f64,
    pub num_classes: usize,
}

impl ShdDataset {
    pub fn from_json(path: &Path) -> Result<Self, String> {
        let data = fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
        serde_json::from_str(&data).map_err(|e| format!("parse: {}", e))
    }

    /// Rate-Coding + Downsampling auf 100D (GridEncoder-kompatibel)
    /// Binned firing rates, L2-normalisiert auf r=0.75 für Poincaré-Ball
    pub fn to_feature_vector(sample: &ShdSample, num_neurons: usize, duration_ms: f64) -> Vec<f64> {
        let bin_size = (num_neurons as f64 / 100.0).ceil().max(1.0) as usize;
        let mut bins = vec![0.0f64; 100];

        for (_time, neuron) in &sample.spikes {
            let idx = (*neuron as usize / bin_size).min(99);
            bins[idx] += 1.0;
        }

        let secs = duration_ms / 1000.0;
        for b in &mut bins {
            *b /= secs;
        }

        // L2-Norm auf target_radius = 0.75 (wie GridEncoder)
        let norm: f64 = bins.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-9 {
            let scale = 0.75 / norm;
            for b in &mut bins {
                *b *= scale;
            }
        }

        bins
    }

    /// Time-To-First-Spike Coding: 700 Neuronen → 100 Bins
    /// Frühere Spikes = höhere Werte. Zeitliche Struktur bleibt erhalten.
    pub fn to_feature_vector_ttfs(sample: &ShdSample, num_neurons: usize, duration_ms: f64) -> Vec<f64> {
        let bin_size = (num_neurons as f64 / 100.0).ceil().max(1.0) as usize;
        let mut ttfs = vec![duration_ms; 100]; // Default: keine Spike = max Dauer

        for (time, neuron) in &sample.spikes {
            let idx = (*neuron as usize / bin_size).min(99);
            let t = *time;
            if t < ttfs[idx] {
                ttfs[idx] = t;
            }
        }

        // Invertieren: frühere Spikes = höhere Aktivität
        // Normalisiert auf [0, 1], dann L2-Norm auf r=0.75
        let mut features: Vec<f64> = ttfs.iter()
            .map(|&t| 1.0 - (t / duration_ms))
            .collect();

        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-9 {
            let scale = 0.75 / norm;
            for f in &mut features {
                *f *= scale;
            }
        }

        features
    }
}
