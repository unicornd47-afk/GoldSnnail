//! DSL Oracle Analysis on ARC Training Tasks
//!
//! Loads all 400 training tasks, runs the DSL solver (max_length=3),
//! and collects statistics on which operations are used in solving programs.

use goldsnnail::ArcTask;
use goldsnnail::vision::dsl_solver::{find_solving_program, Op};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn load_training_set(dir: &str) -> Vec<ArcTask> {
    let path = Path::new(dir);
    if !path.exists() || !path.is_dir() {
        eprintln!("Directory '{}' not found", dir);
        return Vec::new();
    }

    let entries = match fs::read_dir(path) {
        Ok(e) => e.filter_map(|e| e.ok()).collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("Failed to read directory '{}': {}", dir, e);
            return Vec::new();
        }
    };

    let mut tasks = Vec::new();
    for entry in entries {
        let entry_path = entry.path();
        if entry_path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        let id = entry_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let data = match fs::read_to_string(&entry_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to read '{}': {}", entry_path.display(), e);
                continue;
            }
        };

        let value: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to parse JSON '{}': {}", id, e);
                continue;
            }
        };

        match ArcTask::from_json(&id, &value) {
            Ok(task) => tasks.push(task),
            Err(e) => eprintln!("Skipping invalid ARC file '{}': {}", id, e),
        }
    }

    tasks
}

fn main() {
    println!("=== DSL Oracle Analysis on ARC Training Tasks ===\n");

    let tasks = load_training_set("data/arc-agi-repo/data/training");
    println!("Loaded {} training tasks\n", tasks.len());

    let mut op_counts: HashMap<Op, usize> = HashMap::new();
    let mut depth_counts: HashMap<usize, usize> = HashMap::new();
    let mut length_counts: HashMap<usize, usize> = HashMap::new();
    let mut total_length: usize = 0;
    let mut solved_count: usize = 0;
    let mut unsolved_count: usize = 0;
    let mut max_depth_seen: usize = 0;

    for task in &tasks {
        match find_solving_program(task, 3) {
            Some(program) => {
                solved_count += 1;
                let len = program.ops.len();
                total_length += len;
                *length_counts.entry(len).or_insert(0) += 1;

                if len > max_depth_seen {
                    max_depth_seen = len;
                }

                *depth_counts.entry(len).or_insert(0) += 1;

                for &op in &program.ops {
                    *op_counts.entry(op).or_insert(0) += 1;
                }
            }
            None => {
                unsolved_count += 1;
            }
        }
    }

    let total_ops: usize = op_counts.values().sum();
    let avg_length = if solved_count > 0 {
        total_length as f64 / solved_count as f64
    } else {
        0.0
    };

    println!("=== Solve Statistics ===");
    println!("Total tasks:        {}", tasks.len());
    println!("Solved:             {} ({:.1}%)", solved_count, solved_count as f64 / tasks.len() as f64 * 100.0);
    println!("Unsolved:           {} ({:.1}%)", unsolved_count, unsolved_count as f64 / tasks.len() as f64 * 100.0);
    println!("Average length:     {:.2}", avg_length);
    println!("Max depth:          {}", max_depth_seen);

    println!("\n=== Program Length Distribution ===");
    let mut lengths: Vec<_> = length_counts.iter().collect();
    lengths.sort_by_key(|&(k, _)| *k);
    for (len, count) in lengths {
        let pct = *count as f64 / solved_count as f64 * 100.0;
        println!("  Length {}: {:>4} tasks ({:5.1}%)", len, count, pct);
    }

    println!("\n=== Operation Frequency Table ===");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));

    println!("{:>6}  {:>12}  {:>8}  {:>10}  Op", "Rank", "Count", "Pct", "PerTask");
    println!("{}", "-".repeat(55));

    for (rank, (op, count)) in sorted_ops.iter().enumerate() {
        let pct = **count as f64 / total_ops as f64 * 100.0;
        let per_task = **count as f64 / solved_count as f64;
        println!("{:>6}  {:>12}  {:>7.2}%  {:>10.2}  {}", 
            rank + 1, 
            count, 
            pct, 
            per_task, 
            op.name()
        );
    }

    println!("\nTotal operation instances: {}", total_ops);
    println!("\n=== Depth Distribution ===");
    let mut depths: Vec<_> = depth_counts.iter().collect();
    depths.sort_by_key(|&(k, _)| *k);
    for (depth, count) in depths {
        let pct = *count as f64 / solved_count as f64 * 100.0;
        println!("  Depth {}: {:>4} tasks ({:5.1}%)", depth, count, pct);
    }
}
