use anyhow::Result;
use chrono::{DateTime, Utc};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::evaluator::EvalResult;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LeaderboardTracker {
    pub last_updated: DateTime<Utc>,
    pub entries: HashMap<String, TrackerEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrackerEntry {
    pub benchmark: String,
    pub our_best_score: f64,
    pub our_best_run: String,
    pub runs_count: u32,
    pub prize_pool: String,
    pub next_milestone: String,
    pub notes: String,
}

pub fn init(out_dir: &Path) -> Result<()> {
    let tracker = LeaderboardTracker {
        last_updated: Utc::now(),
        entries: HashMap::new(),
    };
    save(out_dir, &tracker)?;
    Ok(())
}

pub fn record_result(out_dir: &Path, result: &EvalResult) -> Result<()> {
    let path = out_dir.join("leaderboard_tracker.json");
    let mut tracker = if path.exists() {
        let data = fs::read_to_string(&path)?;
        serde_json::from_str(&data)?
    } else {
        LeaderboardTracker::default()
    };

    let entry = tracker.entries.entry(result.benchmark.clone()).or_insert_with(|| TrackerEntry {
        benchmark: result.benchmark.clone(),
        our_best_score: 0.0,
        our_best_run: String::new(),
        runs_count: 0,
        prize_pool: "Unknown".into(),
        next_milestone: "Submit first result".into(),
        notes: String::new(),
    });

    entry.runs_count += 1;

    if result.score > entry.our_best_score {
        entry.our_best_score = result.score;
        entry.our_best_run = result.timestamp.clone();
        let sha_prefix = if result.model_sha256.len() > 16 {
            &result.model_sha256[..16]
        } else {
            &result.model_sha256
        };
        entry.notes = format!("Model: {} | Latency: {}µs", sha_prefix, result.latency_us);
    }

    // Milestones automatisch setzen
    match result.benchmark.as_str() {
        "arc-prize" => {
            if result.score == 0.0 {
                entry.next_milestone = "Baseline etabliert — Reasoning-Modul benötigt".into();
            } else if result.score < 0.85 {
                entry.next_milestone = format!("85% für $100k Prize (aktuell: {:.1}%)", result.score * 100.0);
            }
        }
        "n-mnist" => {
            entry.next_milestone = format!("SOTA ist ~99.4% (aktuell: {:.1}%)", result.score);
        }
        _ => {}
    }

    tracker.last_updated = Utc::now();
    save(out_dir, &tracker)?;
    Ok(())
}

pub fn print_status(out_dir: &Path) -> Result<()> {
    let path = out_dir.join("leaderboard_tracker.json");
    if !path.exists() {
        println!("{}", "⚠️  Kein Tracker gefunden. Führe `init` aus.".yellow());
        return Ok(());
    }

    let data = fs::read_to_string(&path)?;
    let tracker: LeaderboardTracker = serde_json::from_str(&data)?;

    println!("{}\n", "📊 Leaderboard Tracker".bold().underline());
    println!("Letztes Update: {}\n", tracker.last_updated);

    for (name, e) in &tracker.entries {
        let status = if e.our_best_score > 0.0 {
            format!("{:.2}%", e.our_best_score * 100.0).green()
        } else {
            "0% (Baseline)".red()
        };

        println!("  {} {}", name.bold(), status);
        println!("     Runs: {} | Best: {}", e.runs_count, e.our_best_run);
        println!("     Prize: {} | Next: {}", e.prize_pool.dimmed(), e.next_milestone.italic());
        println!("     Notes: {}\n", e.notes.dimmed());
    }

    Ok(())
}

fn save(out_dir: &Path, tracker: &LeaderboardTracker) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let path = out_dir.join("leaderboard_tracker.json");
    fs::write(&path, serde_json::to_string_pretty(tracker)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile;

    #[test]
    fn init_creates_tracker_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = init(dir.path());
        assert!(result.is_ok());
        assert!(dir.path().join("leaderboard_tracker.json").exists());
    }

    #[test]
    fn record_result_updates_best_score() {
        let dir = tempfile::tempdir().unwrap();
        init(dir.path()).unwrap();

        let eval_result = EvalResult {
            benchmark: "arc-prize".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            score: 0.85,
            model_size_mb: 0.92,
            latency_us: 72.0,
            model_sha256: "abc123".into(),
            rust_version: "rustc 1.85.0".into(),
            hardware: "Test CPU / 4 cores".into(),
            log: String::new(),
        };

        record_result(dir.path(), &eval_result).unwrap();
        let data = fs::read_to_string(dir.path().join("leaderboard_tracker.json")).unwrap();
        let tracker: LeaderboardTracker = serde_json::from_str(&data).unwrap();
        let entry = tracker.entries.get("arc-prize").unwrap();
        assert_eq!(entry.our_best_score, 0.85);
        assert_eq!(entry.runs_count, 1);
    }

    #[test]
    fn record_result_does_not_overwrite_best_score() {
        let dir = tempfile::tempdir().unwrap();
        init(dir.path()).unwrap();

        let low = EvalResult {
            benchmark: "arc-prize".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            score: 0.5,
            model_size_mb: 0.92,
            latency_us: 72.0,
            model_sha256: "abc123".into(),
            rust_version: "rustc 1.85.0".into(),
            hardware: "Test CPU / 4 cores".into(),
            log: String::new(),
        };

        let high = EvalResult {
            benchmark: "arc-prize".into(),
            timestamp: "2026-01-02T00:00:00Z".into(),
            score: 0.9,
            model_size_mb: 0.92,
            latency_us: 72.0,
            model_sha256: "def456".into(),
            rust_version: "rustc 1.85.0".into(),
            hardware: "Test CPU / 4 cores".into(),
            log: String::new(),
        };

        record_result(dir.path(), &low).unwrap();
        record_result(dir.path(), &high).unwrap();

        let data = fs::read_to_string(dir.path().join("leaderboard_tracker.json")).unwrap();
        let tracker: LeaderboardTracker = serde_json::from_str(&data).unwrap();
        let entry = tracker.entries.get("arc-prize").unwrap();
        assert_eq!(entry.our_best_score, 0.9);
        assert_eq!(entry.runs_count, 2);
    }

    #[test]
    fn arc_prize_milestone_set_correctly() {
        let dir = tempfile::tempdir().unwrap();
        init(dir.path()).unwrap();

        let eval_result = EvalResult {
            benchmark: "arc-prize".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            score: 0.0,
            model_size_mb: 0.92,
            latency_us: 72.0,
            model_sha256: "abc123".into(),
            rust_version: "rustc 1.85.0".into(),
            hardware: "Test CPU / 4 cores".into(),
            log: String::new(),
        };

        record_result(dir.path(), &eval_result).unwrap();

        let data = fs::read_to_string(dir.path().join("leaderboard_tracker.json")).unwrap();
        let tracker: LeaderboardTracker = serde_json::from_str(&data).unwrap();
        let entry = tracker.entries.get("arc-prize").unwrap();
        assert!(entry.next_milestone.contains("Reasoning-Modul"));
    }
}
