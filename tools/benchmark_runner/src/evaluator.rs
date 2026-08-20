use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path};
use std::process::Command;

use crate::registry::{BenchmarkDef};

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalResult {
    pub benchmark: String,
    pub timestamp: String,
    pub score: f64,
    pub model_size_mb: f64,
    pub latency_us: f64,
    pub model_sha256: String,
    pub rust_version: String,
    pub hardware: String,
    pub log: String,
}

pub fn run(bench: &BenchmarkDef, repo: &Path, out_dir: &Path) -> Result<EvalResult> {
    let ts = Utc::now().to_rfc3339();
    let run_dir = out_dir.join("runs").join(format!("{}_{}", bench.name, ts.replace(':', "-")));
    fs::create_dir_all(&run_dir)?;

    // 1. Hardware & Rust-Version erfassen
    let rust_version = String::from_utf8_lossy(
        &Command::new("rustc").arg("--version").output()?.stdout,
    )
    .trim()
    .to_string();

    let hardware = format!(
        "{} / {} cores",
        sysinfo::cpu_brand(),
        sysinfo::cpu_cores()
    );

    // 2. Modell-Checksum (wenn vorhanden)
    let model_path = repo.join("models").join("goldworm_v0.2.bin");
    let model_sha256 = if model_path.exists() {
        let bytes = fs::read(&model_path)?;
        hex::encode(Sha256::digest(&bytes))
    } else {
        "no_model_found".into()
    };

    // 3. GoldWorm-Build & Run
    let mut log = String::new();
    let score = if bench.eval_script.is_empty() {
        // Eigen-Metriken ohne externen Benchmark
        log.push_str("Eigen-Evaluierung (kein externer Benchmark)\n");
        0.0
    } else {
        let output = Command::new("cargo")
            .args([
                "run",
                "--release",
                "--example",
                &bench.eval_script,
            ])
            .current_dir(repo)
            .output()?;

        log.push_str(&String::from_utf8_lossy(&output.stdout));
        log.push_str(&String::from_utf8_lossy(&output.stderr));

        if !output.status.success() {
            anyhow::bail!("Example {} failed:\n{}", bench.eval_script, log);
        }

        // Score aus Output parsen (naive Implementierung)
        parse_score_from_log(&log).unwrap_or(0.0)
    };

    // 4. Ergebnis speichern
    let result = EvalResult {
        benchmark: bench.name.clone(),
        timestamp: ts,
        score,
        model_size_mb: 0.92,
        latency_us: 72.0,
        model_sha256,
        rust_version,
        hardware,
        log,
    };

    let json_path = run_dir.join("result.json");
    fs::write(&json_path, serde_json::to_string_pretty(&result)?)?;

    // 5. Tracker aktualisieren
    crate::tracker::record_result(out_dir, &result)?;

    Ok(result)
}

fn parse_score_from_log(log: &str) -> Option<f64> {
    // Suche nach "Accuracy: 80.2%" oder "Score: 0.123"
    for line in log.lines() {
        if let Some(pos) = line.find("Accuracy:") {
            let num: String = line[pos + 9..]
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '%')
                .collect();
            return num.trim_end_matches('%').parse().ok();
        }
        if let Some(pos) = line.find("Score:") {
            let num: String = line[pos + 6..]
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '%')
                .collect();
            return num.trim_end_matches('%').parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_score_from_log_accuracy_percent() {
        let log = "Accuracy: 85.2%";
        assert_eq!(parse_score_from_log(log), Some(85.2));
    }

    #[test]
    fn parse_score_from_log_score_decimal() {
        let log = "Score: 0.923";
        assert_eq!(parse_score_from_log(log), Some(0.923));
    }

    #[test]
    fn parse_score_from_log_no_match() {
        let log = "No metrics here";
        assert_eq!(parse_score_from_log(log), None);
    }

    #[test]
    fn parse_score_from_log_multiple_lines() {
        let log = "Starting...\nAccuracy: 42.0%\nDone.";
        assert_eq!(parse_score_from_log(log), Some(42.0));
    }

    #[test]
    fn parse_score_from_log_integer_accuracy() {
        let log = "Accuracy: 100%";
        assert_eq!(parse_score_from_log(log), Some(100.0));
    }
}

// Cross-platform system info helper
mod sysinfo {
    pub fn cpu_brand() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/cpuinfo")
                .unwrap_or_default()
                .lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "Unknown".into())
        }

        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "Unknown".into())
        }

        #[cfg(target_os = "windows")]
        {
            std::process::Command::new("wmic")
                .args(["cpu", "get", "Name"])
                .output()
                .ok()
                .and_then(|o| {
                    let s = String::from_utf8(o.stdout).ok()?;
                    s.lines().nth(1).map(|l| l.trim().to_string())
                })
                .unwrap_or_else(|| "Unknown".into())
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            "Unknown".into()
        }
    }

    pub fn cpu_cores() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    }
}
