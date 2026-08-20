//! ARC-Prize Evaluation — Docker-kompatibel
//!
//! Liest ARC-Tasks aus Env-Var, schreibt Predictions als JSON.
//! Wird von `tools/benchmark_runner/arc_entrypoint.sh` aufgerufen.
//!
//! Umgebungsvariablen:
//!   GOLDSNNAIL_ARC_INPUT  — Pfad zu tasks.json
//!   GOLDSNNAIL_ARC_OUTPUT — Pfad zu submission.json (Kaggle/ARC-Prize Format)
//!
//! Ausgabeformat (ARC-Prize / Kaggle-kompatibel):
//!   {
//!     "task_id_1": [
//!       [[0, 1], [1, 0]],
//!       [[1, 1], [0, 0]]
//!     ]
//!   }

use std::collections::HashMap;
use std::env;
use std::fs;
use serde_json;

fn main() {
    let input_path = env::var("GOLDSNNAIL_ARC_INPUT")
        .unwrap_or_else(|_| "data/arc/arc-agi_evaluation_challenges.json".to_string());

    let output_path = env::var("GOLDSNNAIL_ARC_OUTPUT")
        .unwrap_or_else(|_| "submission.json".to_string());

    let data = fs::read_to_string(&input_path)
        .unwrap_or_else(|_| "{}".to_string());

    // Parse ARC-AGI JSON (vereinfacht)
    let tasks: HashMap<String, serde_json::Value> =
        serde_json::from_str(&data).unwrap_or_default();

    let mut submission = HashMap::new();

    for (task_id, _task) in &tasks {
        // TODO: Dein Hybrid-Solver hier aufrufen
        // Aktuell: Leere Prediction als Baseline (2 Versuche pro Aufgabe)
        let attempts = vec![
            vec![vec![0]],
            vec![vec![0]],
        ];
        submission.insert(task_id.clone(), attempts);
    }

    fs::write(&output_path, serde_json::to_string_pretty(&submission).unwrap())
        .expect("Konnte Output nicht schreiben");

    println!("Benchmark: arc-prize");
    println!("Accuracy: 0%");
    println!("Tasks evaluated: {}", submission.len());
    println!("Output: {}", output_path);
    println!("Status: BASELINE");
}
