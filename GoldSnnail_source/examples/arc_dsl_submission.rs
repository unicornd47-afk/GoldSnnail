//! ARC Kaggle Submission mit DSL-Solver
//!
//! Für jede Eval-Task:
//! 1. Versuche DSL-Solver (max_length=2) auf Train-Paaren
// 2. Wenn gefunden: Wende auf Test-Inputs an
//! 3. Wenn nicht gefunden: Fallback = Identity (Input = Output)

use goldworm::{ArcTask, find_solving_program};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn load_evaluation_set(dir: &str) -> Vec<ArcTask> {
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
    println!("=== ARC Kaggle Submission (DSL Solver + Identity Fallback) ===\n");

    let tasks = load_evaluation_set("data/arc/evaluation");
    if tasks.is_empty() {
        println!("No evaluation tasks loaded.");
        return;
    }

    println!("Loaded {} evaluation tasks\n", tasks.len());

    let mut dsl_solved = 0;
    let mut identity_fallback = 0;
    let mut submission: HashMap<String, HashMap<String, Vec<Vec<Vec<u8>>>>> = HashMap::new();

    for task in &tasks {
        let mut task_entry: HashMap<String, Vec<Vec<Vec<u8>>>> = HashMap::new();
        
        // Try DSL solver
        let program = find_solving_program(task, 2);
        
        if let Some(prog) = program {
            dsl_solved += 1;
            let mut attempt1 = Vec::new();
            let mut attempt2 = Vec::new();
            
            for input_grid in &task.test_inputs {
                let output1 = prog.apply(input_grid).unwrap_or_else(|| input_grid.clone());
                let output2 = input_grid.clone(); // Identity fallback for attempt 2
                
                attempt1.push(output1.data.clone());
                attempt2.push(output2.data.clone());
            }
            
            task_entry.insert("attempt_1".to_string(), attempt1);
            task_entry.insert("attempt_2".to_string(), attempt2);
        } else {
            identity_fallback += 1;
            let mut attempt1 = Vec::new();
            let mut attempt2 = Vec::new();
            
            for input_grid in &task.test_inputs {
                attempt1.push(input_grid.data.clone());
                attempt2.push(input_grid.data.clone());
            }
            
            task_entry.insert("attempt_1".to_string(), attempt1);
            task_entry.insert("attempt_2".to_string(), attempt2);
        }
        
        submission.insert(task.id.clone(), task_entry);
    }

    let submission_path = "data/arc_submission/submission_dsl_v1.json";
    if let Err(e) = fs::create_dir_all("data/arc_submission") {
        eprintln!("Failed to create output directory: {}", e);
        return;
    }

    let submission_json = match serde_json::to_string_pretty(&submission) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Failed to serialize submission: {}", e);
            return;
        }
    };
    
    let file_size_kb = submission_json.len() as f64 / 1024.0;

    match fs::write(submission_path, &submission_json) {
        Ok(_) => {
            println!("Kaggle submission saved to {}", submission_path);
            println!("File size: {:.1} KB", file_size_kb);
        }
        Err(e) => println!("Failed to write submission: {}", e),
    }

    println!("\n=== Summary ===");
    println!("Total tasks:             {}", tasks.len());
    println!("DSL solved:              {} ({:.1}%)", dsl_solved, dsl_solved as f64 / tasks.len() as f64 * 100.0);
    println!("Identity fallback:       {} ({:.1}%)", identity_fallback, identity_fallback as f64 / tasks.len() as f64 * 100.0);
}
