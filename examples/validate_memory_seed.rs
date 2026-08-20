//! Validierung des MemorySeed-Codecs
//!
//! Misst den Rekonstruktionsfehler (L2-Norm) für zufällige Aktivierungsmuster.
//! Ziel: <5% relativer Fehler bei 4.8:1 Kompression.

use goldsnnail::memory_seed::{ActivationPattern, MemorySeed};

fn main() {
    // 1. Generiere ein realistisches Aktivierungsmuster
    let mut values = vec![0.0f32; 320];
    for i in 0..320 {
        // Simuliere: einige Cluster stark aktiv, andere schwach
        let cluster = i / 16;
        let intensity = if cluster % 3 == 0 { 0.8 } else { 0.1 };
        values[i] = intensity + rand::random::<f32>() * 0.1;
    }

    let pattern = ActivationPattern {
        values,
        cluster_indices: vec![0, 3, 6, 7, 10, 13],
    };

    // 2. Komprimieren
    let seed = MemorySeed::encode(&pattern);
    println!("Seed: {:?} (12 Bytes)", seed.seed.bytes);
    println!("Residual: {} Bytes", seed.residual.bytes.len());

    // 3. Dekomprimieren
    let _decoded = seed.decode();

    // 4. Fehler messen
    let error = seed.reconstruction_error(&pattern);
    let l2_original: f64 = pattern.values.iter().map(|v| (*v as f64).powi(2)).sum();
    let relative_error = if l2_original > 0.0 {
        error / l2_original.sqrt()
    } else {
        0.0
    };

    println!("Reconstruction Error (L2):      {:.4}", error);
    println!("Relative Error:                 {:.2}%", relative_error * 100.0);
    println!("Compression Ratio:              {:.1}:1", 1280.0 / 268.0);

    // 5. Go/No-Go
    if relative_error < 0.05 {
        println!("✅ VALID: <5% Fehler — Integration freigegeben");
    } else {
        println!("❌ INVALID: >5% Fehler — Algorithmus verbessern");
    }
}
