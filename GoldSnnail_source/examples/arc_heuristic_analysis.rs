use goldworm::ArcTask;
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

fn border_fill(grid: &goldworm::ArcGrid) -> goldworm::ArcGrid {
    let mut data = grid.data.clone();
    let border_color = most_common_color(grid);
    
    // Top and bottom rows
    if grid.height > 0 {
        for c in 0..grid.width {
            data[0][c] = border_color;
            data[grid.height - 1][c] = border_color;
        }
    }
    // Left and right columns
    for r in 0..grid.height {
        data[r][0] = border_color;
        data[r][grid.width - 1] = border_color;
    }
    
    goldworm::ArcGrid::from_data(data).unwrap()
}

fn main() {
    println!("=== ARC Heuristic Analysis ===\n");

    let tasks = load_training_set("data/arc/training");
    if tasks.is_empty() {
        println!("No tasks loaded.");
        return;
    }

    println!("Loaded {} training tasks\n", tasks.len());

    let mut total_test_cases = 0usize;
    let mut same_size_count = 0usize;
    let mut larger_count = 0usize;
    let mut smaller_count = 0usize;

    let mut identity_matches = 0usize;
    let mut mcc_fill_matches = 0usize;
    let mut h_flip_matches = 0usize;
    let mut v_flip_matches = 0usize;
    let mut rot_180_matches = 0usize;
    let mut border_fill_matches = 0usize;

    for task in &tasks {
        for (i, input_grid) in task.test_inputs.iter().enumerate() {
            total_test_cases += 1;
            
            if let Some(Some(output_grid)) = task.test_outputs.get(i) {
                let input_pixels = input_grid.width * input_grid.height;
                let output_pixels = output_grid.width * output_grid.height;
                
                if input_pixels == output_pixels {
                    same_size_count += 1;
                } else if input_pixels < output_pixels {
                    larger_count += 1;
                } else {
                    smaller_count += 1;
                }
                
                // Only apply size-preserving heuristics when sizes match
                if input_grid.width == output_grid.width && input_grid.height == output_grid.height {
                    if grid_equal(input_grid, output_grid) {
                        identity_matches += 1;
                    }
                    if grid_equal(&most_common_color_fill(input_grid), output_grid) {
                        mcc_fill_matches += 1;
                    }
                    if grid_equal(&horizontal_flip(input_grid), output_grid) {
                        h_flip_matches += 1;
                    }
                    if grid_equal(&vertical_flip(input_grid), output_grid) {
                        v_flip_matches += 1;
                    }
                    if grid_equal(&rotate_180(input_grid), output_grid) {
                        rot_180_matches += 1;
                    }
                    if grid_equal(&border_fill(input_grid), output_grid) {
                        border_fill_matches += 1;
                    }
                }
            }
        }
    }

    println!("=== Size Analysis ===");
    println!("Total test cases:          {}", total_test_cases);
    println!("Same size (input==output): {} ({:.1}%)", same_size_count, same_size_count as f64 / total_test_cases as f64 * 100.0);
    println!("Larger output:             {} ({:.1}%)", larger_count, larger_count as f64 / total_test_cases as f64 * 100.0);
    println!("Smaller output:            {} ({:.1}%)", smaller_count, smaller_count as f64 / total_test_cases as f64 * 100.0);
    
    println!("\n=== Heuristic Accuracy (on same-size test cases only) ===");
    println!("Identity:                  {} / {} ({:.1}%)", identity_matches, same_size_count, identity_matches as f64 / same_size_count as f64 * 100.0);
    println!("Most common color fill:    {} / {} ({:.1}%)", mcc_fill_matches, same_size_count, mcc_fill_matches as f64 / same_size_count as f64 * 100.0);
    println!("Horizontal flip:           {} / {} ({:.1}%)", h_flip_matches, same_size_count, h_flip_matches as f64 / same_size_count as f64 * 100.0);
    println!("Vertical flip:             {} / {} ({:.1}%)", v_flip_matches, same_size_count, v_flip_matches as f64 / same_size_count as f64 * 100.0);
    println!("180 rotation:              {} / {} ({:.1}%)", rot_180_matches, same_size_count, rot_180_matches as f64 / same_size_count as f64 * 100.0);
    println!("Border fill:               {} / {} ({:.1}%)", border_fill_matches, same_size_count, border_fill_matches as f64 / same_size_count as f64 * 100.0);
    
    // Calculate combined accuracy (any heuristic matches)
    let same_size_slice = same_size_count;
    let combined = identity_matches + mcc_fill_matches + h_flip_matches + v_flip_matches + rot_180_matches + border_fill_matches;
    // This overcounts if multiple heuristics match the same task, so it's an upper bound
    println!("\n=== Combined Upper Bound ===");
    println!("If we could magically pick the right heuristic: ~{:.1}% on same-size tasks", 
        (combined as f64 / same_size_count as f64 * 100.0).min(100.0));
}
