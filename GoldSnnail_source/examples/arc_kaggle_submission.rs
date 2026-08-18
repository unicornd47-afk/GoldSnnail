use goldworm::ArcTask;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn load_dataset(dir: &str) -> Vec<ArcTask> {
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

fn grid_equal(a: &goldworm::ArcGrid, b: &goldworm::ArcGrid) -> bool {
    a == b
}

fn horizontal_flip(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let mut data = grid.data.clone();
    for row in &mut data {
        row.reverse();
    }
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn vertical_flip(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let mut data = grid.data.clone();
    data.reverse();
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn rotate_180(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let mut data = grid.data.clone();
    for row in &mut data {
        row.reverse();
    }
    data.reverse();
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn rotate_90_ccw(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let mut data = vec![vec![0u8; grid.height]; grid.width];
    for r in 0..grid.height {
        for c in 0..grid.width {
            data[grid.width - 1 - c][r] = grid.data[r][c];
        }
    }
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn rotate_90_cw(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let mut data = vec![vec![0u8; grid.height]; grid.width];
    for r in 0..grid.height {
        for c in 0..grid.width {
            data[c][grid.height - 1 - r] = grid.data[r][c];
        }
    }
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn transpose(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let mut data = vec![vec![0u8; grid.height]; grid.width];
    for r in 0..grid.height {
        for c in 0..grid.width {
            data[c][r] = grid.data[r][c];
        }
    }
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn most_common_color(grid: &goldworm::ArcGrid) -> u8 {
    let mut counts = [0usize; 10];
    for row in &grid.data {
        for &c in row {
            counts[c as usize] += 1;
        }
    }
    let mut max_color = 0;
    let mut max_count = 0;
    for (color, &count) in counts.iter().enumerate() {
        if count > max_count {
            max_count = count;
            max_color = color;
        }
    }
    max_color as u8
}

fn most_common_color_fill(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let color = most_common_color(grid);
    let data = vec![vec![color; grid.width]; grid.height];
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn infer_color_mapping(input: &goldworm::ArcGrid, output: &goldworm::ArcGrid) -> Option<Vec<u8>> {
    if input.width != output.width || input.height != output.height {
        return None;
    }
    
    let mut mapping = vec![255u8; 10];
    let mut used = vec![false; 10];
    
    for (in_cell, out_cell) in input.data.iter().flatten().zip(output.data.iter().flatten()) {
        let in_color = *in_cell as usize;
        let out_color = *out_cell as usize;
        
        if mapping[in_color] == 255 {
            mapping[in_color] = out_color as u8;
            used[out_color] = true;
        } else if mapping[in_color] != out_color as u8 {
            return None; // Inconsistent mapping
        }
    }
    
    Some(mapping)
}

fn apply_color_mapping(grid: &goldworm::ArcGrid, mapping: &Vec<u8>) -> goldworm::ArcGrid {
    let data: Vec<Vec<u8>> = grid.data.iter()
        .map(|row| row.iter().map(|&c| mapping[c as usize]).collect())
        .collect();
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn infer_transformation(task: &ArcTask) -> (Option<String>, Option<Vec<u8>>) {
    if task.train_pairs.is_empty() {
        return (None, None);
    }

    let mut transform_scores: HashMap<String, usize> = HashMap::new();
    let mut mapping_scores: usize = 0;
    let mut best_mapping: Option<Vec<u8>> = None;
    
    for (input, output) in &task.train_pairs {
        if input.width != output.width || input.height != output.height {
            continue;
        }
        
        let h_flip = horizontal_flip(input);
        let v_flip = vertical_flip(input);
        let rot_180 = rotate_180(input);
        let rot_90_ccw = rotate_90_ccw(input);
        let rot_90_cw = rotate_90_cw(input);
        let trans = transpose(input);
        let mcc = most_common_color_fill(input);
        
        if grid_equal(input, output) {
            *transform_scores.entry("identity".to_string()).or_insert(0) += 1;
        }
        if grid_equal(&h_flip, output) {
            *transform_scores.entry("h_flip".to_string()).or_insert(0) += 1;
        }
        if grid_equal(&v_flip, output) {
            *transform_scores.entry("v_flip".to_string()).or_insert(0) += 1;
        }
        if grid_equal(&rot_180, output) {
            *transform_scores.entry("rot_180".to_string()).or_insert(0) += 1;
        }
        if grid_equal(&rot_90_ccw, output) {
            *transform_scores.entry("rot_90_ccw".to_string()).or_insert(0) += 1;
        }
        if grid_equal(&rot_90_cw, output) {
            *transform_scores.entry("rot_90_cw".to_string()).or_insert(0) += 1;
        }
        if grid_equal(&trans, output) {
            *transform_scores.entry("transpose".to_string()).or_insert(0) += 1;
        }
        if grid_equal(&mcc, output) {
            *transform_scores.entry("mcc_fill".to_string()).or_insert(0) += 1;
        }
        
        if let Some(mapping) = infer_color_mapping(input, output) {
            mapping_scores += 1;
            if best_mapping.is_none() {
                best_mapping = Some(mapping);
            }
        }
    }
    
    let best_transform = transform_scores.into_iter().max_by_key(|(_, count)| *count).map(|(name, _)| name);
    
    if mapping_scores == task.train_pairs.len() && task.train_pairs.len() > 0 {
        return (Some("color_map".to_string()), best_mapping);
    }
    
    (best_transform, None)
}

fn apply_transformation(grid: &goldworm::ArcGrid, transform: &str, mapping: Option<&Vec<u8>>) -> goldworm::ArcGrid {
    match transform {
        "identity" => grid.clone(),
        "h_flip" => horizontal_flip(grid),
        "v_flip" => vertical_flip(grid),
        "rot_180" => rotate_180(grid),
        "rot_90_ccw" => rotate_90_ccw(grid),
        "rot_90_cw" => rotate_90_cw(grid),
        "transpose" => transpose(grid),
        "mcc_fill" => most_common_color_fill(grid),
        "color_map" => {
            if let Some(m) = mapping {
                apply_color_mapping(grid, m)
            } else {
                grid.clone()
            }
        }
        _ => grid.clone(),
    }
}

fn main() {
    println!("=== ARC Kaggle Submission (Expanded Heuristics) ===\n");

    // Load evaluation set for proper measurement
    let eval_tasks = load_dataset("data/arc/evaluation");
    if eval_tasks.is_empty() {
        println!("No evaluation tasks loaded.");
        return;
    }

    println!("Loaded {} evaluation tasks\n", eval_tasks.len());

    let mut total_test_cases = 0usize;
    let mut exact_matches = 0usize;
    let mut total_pixels = 0usize;
    let mut transform_counts: HashMap<String, usize> = HashMap::new();

    // Kaggle submission format:
    // For tasks with 1 test case: attempt_1 is a single grid
    // For tasks with 2+ test cases: attempt_1 is a list of grids
    let mut submission: HashMap<String, serde_json::Value> = HashMap::new();

    for task in &eval_tasks {
        total_test_cases += task.test_inputs.len();
        
        let (transform, mapping) = infer_transformation(task);
        
        if let Some(ref t) = transform {
            *transform_counts.entry(t.clone()).or_insert(0) += 1;
        } else {
            *transform_counts.entry("unknown".to_string()).or_insert(0) += 1;
        }
        
        let mut attempt1_grids = Vec::new();
        let mut attempt2_grids = Vec::new();
        
        for input_grid in &task.test_inputs {
            total_pixels += input_grid.width * input_grid.height;
            
            let output1 = apply_transformation(input_grid, transform.as_deref().unwrap_or("identity"), mapping.as_ref());
            let output2 = most_common_color_fill(input_grid);
            
            attempt1_grids.push(output1.to_json_value());
            attempt2_grids.push(output2.to_json_value());
        }
        
        // Format: single test case -> grid directly, multiple -> list of grids
        let attempt1 = if attempt1_grids.len() == 1 {
            attempt1_grids.into_iter().next().unwrap()
        } else {
            serde_json::Value::Array(attempt1_grids)
        };
        
        let attempt2 = if attempt2_grids.len() == 1 {
            attempt2_grids.into_iter().next().unwrap()
        } else {
            serde_json::Value::Array(attempt2_grids)
        };
        
        let mut task_entry = serde_json::json!({
            "attempt_1": attempt1,
            "attempt_2": attempt2
        });
        
        submission.insert(task.id.clone(), task_entry);
    }

    // Calculate accuracy if we have ground truth
    for task in &eval_tasks {
        let (transform, mapping) = infer_transformation(task);
        
        for (i, input_grid) in task.test_inputs.iter().enumerate() {
            let output1 = apply_transformation(input_grid, transform.as_deref().unwrap_or("identity"), mapping.as_ref());
            let output2 = most_common_color_fill(input_grid);
            
            if let Some(Some(expected)) = task.test_outputs.get(i) {
                if grid_equal(&output1, expected) || grid_equal(&output2, expected) {
                    exact_matches += 1;
                }
            }
        }
    }

    let accuracy = if total_test_cases > 0 {
        (exact_matches as f64 / total_test_cases as f64) * 100.0
    } else {
        0.0
    };

    println!("=== Metrics ===");
    println!("Total tasks:              {}", eval_tasks.len());
    println!("Total test cases:         {}", total_test_cases);
    println!("Exact matches:            {} ({:.1}%)", exact_matches, accuracy);
    println!("Average grid size:        {:.1} pixels", total_pixels as f64 / total_test_cases as f64);

    println!("\n=== Transform Distribution ===");
    for (transform, count) in transform_counts {
        println!("  {}: {} tasks", transform, count);
    }

    let submission_path = "data/arc_submission/submission_kaggle_v1.json";
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
            println!("\nKaggle submission saved to {}", submission_path);
            println!("File size: {:.1} KB", file_size_kb);
        }
        Err(e) => println!("Failed to write submission: {}", e),
    }
}
