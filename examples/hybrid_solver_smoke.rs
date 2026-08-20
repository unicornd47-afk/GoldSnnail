//! Hybrid Solver Smoke Test — Woche 1
//!
//! Einschraenkungen: Nur 10x10 Grids, 3 Heuristiken (Identity, Rotate90, FlipHorizontal)

use goldworm::{ArcDataset, GridEncoder, evaluate_hybrid_solver};

fn main() {
    println!("=== GoldWorm Hybrid Solver Smoke Test (Woche 1) ===\n");
    println!("Einschraenkungen: Nur 10x10 Grids, 3 Heuristiken\n");

    let dataset = match ArcDataset::load_from_directory("data/arc") {
        Ok(ds) => ds,
        Err(e) => {
            println!("Failed to load ARC data: {}", e);
            println!("Download with: git clone https://github.com/fchollet/ARC.git");
            return;
        }
    };

    println!("Loaded {} tasks", dataset.tasks.len());

    // Untrainierter Encoder als Fallback
    let encoder = GridEncoder::new(100, 32, 16, 0.75);

    let result = evaluate_hybrid_solver(&dataset, &encoder, 5, 20);

    println!("\n=== Ergebnis ===");
    println!("Gesamt:        {}", result.total);
    println!(
        "Versucht:      {} ({:.1}%)",
        result.attempted,
        result.attempt_rate * 100.0
    );
    println!(
        "Korrekt:       {} ({:.1}%)",
        result.correct,
        result.accuracy * 100.0
    );

    if result.accuracy > 0.0 {
        println!("\n🎉 Hybrid-Solver funktioniert auf einigen Tasks!");
        println!("   Naechster Schritt: k erhoehen, mehr Heuristiken, auf alle 400 Tasks skalieren.");
    } else if result.attempt_rate > 0.0 {
        println!("\n⚠️ Heuristiken gefunden, aber keine korrekte Uebertragung.");
        println!("   Der Router findet Nachbarn, aber die Heuristik skaliert nicht auf neue Tasks.");
    } else {
        println!("\n❌ Keine passenden Heuristiken gefunden.");
        println!("   Entweder gibt es keine 10x10-Tasks mit diesen 3 Heuristiken,");
        println!("   oder der Router findet keine semantisch aehnlichen Nachbarn.");
    }
}
