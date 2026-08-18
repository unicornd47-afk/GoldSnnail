//! MemorySeed — Parametrische Kompression von Avalanche-Aktivierungen
//!
//! Idee: Ein Aktivierungsmuster (20 Cluster × 16D) wird als harmonische
//! Synthese mit φ-gewichteten Frequenzen approximiert. Der Seed speichert
//! die Parameter der Fundamentalwelle. Das Residual speichert die
//! quantisierte Differenz.

use std::f64::consts::PI;
use std::time::{SystemTime, UNIX_EPOCH};

/// Goldener Schnitt φ — bereits in GoldWorms Golden-Angle-Spiral genutzt
const PHI: f64 = 1.618_033_988_749_895;

/// 12-byte Seed: 4 Frequenz-Bänder × 3 Parameter (Amplitude, Phase, Decay)
///
/// Format: [amp0, phase0, decay0, amp1, phase1, decay1, ...]
/// Jedes Feld ist ein u8 (0–255), skaliert auf den Wertebereich.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seed12 {
    pub bytes: [u8; 12],
}

/// 320-byte Residual: Adaptive Quantisierung der Rekonstruktions-Differenz
///
/// 20 Cluster × 16D = 320 Werte. Jedes Residual ist ein i8 (-128..127).
/// 320 Werte × 1 Byte = 320 Bytes.
#[derive(Debug, Clone)]
pub struct Residual320 {
    pub bytes: [i8; 320],
    /// Welche 16 Dimensionen pro Cluster gespeichert werden (Index-Mapping)
    pub dim_indices: [u8; 16],
}

/// Ein komprimierter Gedächtnis-Eintrag
pub struct MemorySeed {
    pub seed: Seed12,
    pub residual: Residual320,
    /// Timestamp für zeitliche Ordnung
    pub timestamp_ms: u64,
}

/// Ein Avalanche-Aktivierungsmuster (Eingabe für die Kompression)
pub struct ActivationPattern {
    /// 20 Cluster × 16D = 320 Werte
    pub values: Vec<f32>, // Länge 320
    /// Welche Cluster aktiv waren (Indices 0..6 für Sprache, 7..13 für Vision)
    pub cluster_indices: Vec<usize>,
}

impl MemorySeed {
    // =====================================================================
    // ENCODING: Aktivierung → Seed + Residual
    // =====================================================================

    pub fn encode(pattern: &ActivationPattern) -> Self {
        assert_eq!(
            pattern.values.len(),
            320,
            "Erwarte 20×16D = 320 Werte, got {}",
            pattern.values.len()
        );

        // 1. Signal als 1D-Funktion über Cluster-Index interpretieren
        let signal = Self::flatten_to_signal(pattern);

        // 2. Harmonische Analyse: Finde die dominante φ-modulierte Frequenz
        let fundamental = Self::harmonic_analyze(&signal);

        // 3. Seed = kompakte Parameter der Fundamentalwelle
        let seed = Seed12::from_fundamental(&fundamental);

        // 4. Rekonstruiere aus dem Seed
        let reconstructed = Self::synthesize(&seed, signal.len());

        // 5. Residual = Differenz (nur wichtigste Dimensionen)
        let residual = Residual320::from_difference(&signal, &reconstructed, &pattern.values);

        MemorySeed {
            seed,
            residual,
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Flacht 320D auf ein 1D-Signal ab (mittelt über Dimensionen pro Cluster)
    fn flatten_to_signal(pattern: &ActivationPattern) -> Vec<f64> {
        let mut signal = Vec::with_capacity(20);
        for cluster_idx in 0..20 {
            let offset = cluster_idx * 16;
            let avg: f64 = pattern.values[offset..offset + 16]
                .iter()
                .map(|&v| v as f64)
                .sum::<f64>()
                / 16.0;
            signal.push(avg);
        }
        signal
    }

    /// Harmonische Analyse mit φ-gewichteten Frequenzen
    ///
    /// Wir suchen nicht die klassische FFT-Frequenz, sondern die φ-modulierte
    /// Fundamentalwelle, die das Signal am besten approximiert.
    fn harmonic_analyze(signal: &[f64]) -> FundamentalWave {
        let n = signal.len() as f64;
        let mut best_error = f64::INFINITY;
        let mut best = FundamentalWave::default();

        // Grid-Search über Frequenz und Phase (coarse, dann fine)
        for freq_idx in 1..=20 {
            let freq = freq_idx as f64 * PHI; // φ-skalierte Frequenz
            for phase_idx in 0..16 {
                let phase = phase_idx as f64 * PI / 8.0;

                let mut amp_sum = 0.0;
                for (i, &val) in signal.iter().enumerate() {
                    let t = (i as f64 / n) * 2.0 * PI;
                    let predicted = (t * freq + phase).cos();
                    amp_sum += val * predicted;
                }
                let amp = amp_sum / n;

                // Rekonstruktionsfehler berechnen
                let mut error = 0.0;
                for (i, &val) in signal.iter().enumerate() {
                    let t = (i as f64 / n) * 2.0 * PI;
                    let predicted = amp * (t * freq + phase).cos();
                    error += (val - predicted).powi(2);
                }

                if error < best_error {
                    best_error = error;
                    best = FundamentalWave { amp, freq, phase, decay: 1.0 };
                }
            }
        }

        // Fine-tuning: Decay-Parameter suchen
        for decay in [0.8f64, 0.9f64, 0.95f64, 1.0f64, 1.05f64, 1.1f64] {
            let mut error = 0.0;
            for (i, &val) in signal.iter().enumerate() {
                let t = (i as f64 / n) * 2.0 * PI;
                let predicted = best.amp * (t * best.freq + best.phase).cos()
                    * decay.powi(i as i32);
                error += (val - predicted).powi(2);
            }
            if error < best_error {
                best_error = error;
                best.decay = decay;
            }
        }

        best
    }

    /// Synthetisiert ein Signal aus dem Seed
    fn synthesize(seed: &Seed12, len: usize) -> Vec<f64> {
        let fund = seed.to_fundamental();
        let n = len as f64;
        (0..len)
            .map(|i| {
                let t = (i as f64 / n) * 2.0 * PI;
                fund.amp * (t * fund.freq + fund.phase).cos()
                    * fund.decay.powi(i as i32)
            })
            .collect()
    }

    // =====================================================================
    // DECODING: Seed + Residual → Aktivierung
    // =====================================================================

    pub fn decode(&self) -> ActivationPattern {
        // 1. Basis-Signal aus Seed rekonstruieren
        let base_signal = Self::synthesize(&self.seed, 20);

        // 2. Residual auf die 16 wichtigsten Dimensionen anwenden
        let mut values = vec![0.0f32; 320];

        for (cluster_idx, &base_val) in base_signal.iter().enumerate() {
            // Finde die 16 wichtigsten Dimensionen aus dem Residual
            for dim in 0..16 {
                let dim_idx = self.residual.dim_indices[dim] as usize;
                let residual_val = self.residual.bytes[cluster_idx * 16 + dim] as f64 / 128.0;

                let offset = cluster_idx * 16 + dim_idx;
                if offset < 320 {
                    values[offset] = (base_val + residual_val) as f32;
                }
            }
        }

        ActivationPattern {
            values,
            cluster_indices: (0..20).collect(), // Alle Cluster als "aktiv" markieren
        }
    }

    /// Rekonstruktionsfehler (L2-Norm) zwischen Original und Dekodiertem
    pub fn reconstruction_error(&self, original: &ActivationPattern) -> f64 {
        let decoded = self.decode();
        original
            .values
            .iter()
            .zip(decoded.values.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// =====================================================================
// Hilfsstrukturen
// =====================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct FundamentalWave {
    pub amp: f64,
    pub freq: f64,
    pub phase: f64,
    pub decay: f64,
}

impl Seed12 {
    pub fn from_fundamental(f: &FundamentalWave) -> Self {
        // Skaliere auf u8-Bereich (0–255)
        let amp = ((f.amp.abs().min(2.0) / 2.0) * 255.0) as u8;
        let freq = (((f.freq % (2.0 * PI)) / (2.0 * PI)) * 255.0) as u8;
        let phase = (((f.phase % (2.0 * PI)) / (2.0 * PI)) * 255.0) as u8;
        let decay = (((f.decay - 0.5) / 1.0).clamp(0.0, 1.0) * 255.0) as u8;

        Seed12 {
            bytes: [amp, freq, phase, decay, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    pub fn to_fundamental(&self) -> FundamentalWave {
        let amp = (self.bytes[0] as f64 / 255.0) * 2.0;
        let freq = (self.bytes[1] as f64 / 255.0) * 2.0 * PI * PHI; // φ-moduliert
        let phase = (self.bytes[2] as f64 / 255.0) * 2.0 * PI;
        let decay = (self.bytes[3] as f64 / 255.0) * 1.0 + 0.5;

        FundamentalWave { amp, freq, phase, decay }
    }
}

impl Residual320 {
    pub fn from_difference(
        _signal: &[f64],
        reconstructed: &[f64],
        full_values: &[f32],
    ) -> Self {
        let mut bytes = [0i8; 320];
        let mut dim_indices = [0u8; 16];

        // Finde die 16 Dimensionen mit höchster Varianz über alle Cluster
        let mut variances = vec![0.0f32; 16];
        for dim in 0..16 {
            let vals: Vec<f32> = (0..20)
                .map(|c| full_values[c * 16 + dim])
                .collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            variances[dim] = var;
        }

        // Top-16 Dimensionen (hier sind es eh nur 16, aber skalierbar)
        let mut indexed: Vec<(usize, f32)> = variances.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (rank, (dim_idx, _)) in indexed.iter().enumerate().take(16) {
            dim_indices[rank] = *dim_idx as u8;
        }

        // Quantisierung der Differenz für diese 16 Dimensionen
        for cluster_idx in 0..20 {
            for dim_rank in 0..16 {
                let dim_idx = dim_indices[dim_rank] as usize;
                let original = full_values[cluster_idx * 16 + dim_idx] as f64;
                let base = reconstructed[cluster_idx];
                let diff = original - base;

                // Skaliere auf i8 (-128..127)
                let quantized = (diff * 128.0).clamp(-128.0, 127.0) as i8;
                bytes[cluster_idx * 16 + dim_rank] = quantized;
            }
        }

        Residual320 { bytes, dim_indices }
    }
}
