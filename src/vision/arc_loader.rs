//! ARC-AGI Dataset Loader
//!
//! Parses ARC-AGI JSON tasks into GoldSnnail-native structures:
//! - `ArcGrid`: flat `Vec<Vec<u8>>` memory layout (DOD pattern)
//! - `ArcTask`: train pairs + test inputs/outputs
//! - `ArcDataset`: collection of tasks loaded from a directory
//!
//! Integration:
//! - `ArcGrid::to_hyperbolic()` → `HyperbolicPoint` via `PoincareBall::exp_map_origin`
//! - `ArcTask::to_concept_nodes()` → `ConceptGraph` nodes per unique color

use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::semantics::{ConceptGraph};
use ndarray::Array1;
use serde_json::Value;
use std::fs;
use std::path::Path;

/// A 2D ARC grid using flat `Vec<Vec<u8>>` memory layout (DOD pattern).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArcGrid {
    pub data: Vec<Vec<u8>>,
    pub width: usize,
    pub height: usize,
}

impl ArcGrid {
    /// Creates a new empty grid.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Creates a grid from 2D data, validating rectangular shape.
    pub fn from_data(data: Vec<Vec<u8>>) -> Result<Self, String> {
        if data.is_empty() {
            return Ok(Self {
                data: Vec::new(),
                width: 0,
                height: 0,
            });
        }
        let height = data.len();
        let width = data[0].len();
        if !data.iter().all(|row| row.len() == width) {
            return Err("All rows must have equal length".to_string());
        }
        Ok(Self { data, width, height })
    }

    /// Creates a grid from a reference to 2D data, cloning the contents.
    pub fn from_2d(data: &Vec<Vec<u8>>) -> Self {
        let height = data.len();
        let width = if height > 0 { data[0].len() } else { 0 };
        let data_clone = data.clone();
        Self { data: data_clone, width, height }
    }

    /// Parses a grid from a JSON value `[[0,0,0],[1,2,3]]`.
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let arr = value.as_array().ok_or("Expected JSON array for grid")?;
        let mut data = Vec::with_capacity(arr.len());
        for row in arr {
            let row_arr = row.as_array().ok_or("Expected row to be JSON array")?;
            let row_data: Result<Vec<u8>, String> = row_arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| format!("Expected integer color value, got {}", v))
                        .and_then(|n| {
                            if n <= 9 {
                                Ok(n as u8)
                            } else {
                                Err(format!("Color value {} out of range 0-9", n))
                            }
                        })
                })
                .collect();
            data.push(row_data?);
        }
        Self::from_data(data)
    }

    /// Serializes the grid to a JSON value `[[0,0,0],[1,2,3]]`.
    pub fn to_json_value(&self) -> Value {
        let rows: Vec<Value> = self
            .data
            .iter()
            .map(|row| {
                let cells: Vec<Value> = row.iter().map(|&c| Value::from(c as u64)).collect();
                Value::Array(cells)
            })
            .collect();
        Value::Array(rows)
    }

    /// Returns a reference to the raw grid data.
    pub fn colors(&self) -> &[Vec<u8>] {
        &self.data
    }

    /// Returns the set of unique color values in the grid (sorted).
    pub fn unique_colors(&self) -> Vec<u8> {
        let mut colors: Vec<u8> = self.data.iter().flatten().copied().collect();
        colors.sort_unstable();
        colors.dedup();
        colors
    }

    /// Normalizes grid values from [0, 9] to [0.0, 1.0].
    pub fn normalize(&self) -> Vec<Vec<f64>> {
        self.data
            .iter()
            .map(|row| row.iter().map(|&c| c as f64 / 9.0).collect())
            .collect()
    }

    /// Flattens the grid to a flat `Vec<f64>` normalized to [0, 1].
    pub fn flatten(&self) -> Vec<f64> {
        self.normalize().into_iter().flatten().collect()
    }

    /// 100D Feature-Vektor für ARC-Grids (DOD-konform: flaches Vec<f64>).
    ///
    /// Struktur:
    /// [0..9]   = Farb-Histogramm (10D)
    /// [10..19] = Zeilen-Mittelwerte (10D)
    /// [20..29] = Spalten-Mittelwerte (10D)
    /// [30..54] = 5×5 Zentrum (25D)
    /// [55..74] = Rand-Mittelwerte + Eckpunkte (20D)
    /// [75..99] = Symmetrie-Features (25D)
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = vec![0.0f32; 100];

        // === [0..9] Farb-Histogramm ===
        let mut hist = [0.0f32; 10];
        for row in &self.data {
            for &c in row {
                hist[c as usize] += 1.0;
            }
        }
        let total = (self.width * self.height).max(1) as f32;
        for i in 0..10 {
            features[i] = hist[i] / total;
        }

        // === [10..19] Zeilen-Mittelwerte ===
        for row in 0..self.height.min(10) {
            let sum: u32 = self.data[row].iter().map(|&c| c as u32).sum();
            features[10 + row] = (sum as f32 / self.width.max(1) as f32) / 9.0;
        }

        // === [20..29] Spalten-Mittelwerte ===
        for col in 0..self.width.min(10) {
            let sum: u32 = (0..self.height)
                .map(|r| self.data[r][col.min(self.width - 1)] as u32)
                .sum();
            features[20 + col] = (sum as f32 / self.height.max(1) as f32) / 9.0;
        }

        // === [30..54] 5×5 Zentrum ===
        let center_r = self.height / 2;
        let center_c = self.width / 2;
        for dr in 0..5 {
            for dc in 0..5 {
                let r = center_r.saturating_sub(2) + dr;
                let c = center_c.saturating_sub(2) + dc;
                let idx = 30 + dr * 5 + dc;
                if r < self.height && c < self.width {
                    features[idx] = self.data[r][c] as f32 / 9.0;
                }
            }
        }

        // === [55..74] Rand-Features ===
        // Top/Bottom row means
        if self.height > 0 {
            let top_mean: u32 = self.data[0].iter().map(|&c| c as u32).sum();
            features[55] = top_mean as f32 / (self.width.max(1) as f32 * 9.0);
            let bot_mean: u32 = self.data[self.height - 1].iter().map(|&c| c as u32).sum();
            features[56] = bot_mean as f32 / (self.width.max(1) as f32 * 9.0);
        }
        // Left/Right col means
        let left_mean: u32 = (0..self.height)
            .map(|r| self.data[r][0] as u32).sum();
        features[57] = left_mean as f32 / (self.height.max(1) as f32 * 9.0);
        let right_mean: u32 = (0..self.height)
            .map(|r| self.data[r][self.width - 1] as u32).sum();
        features[58] = right_mean as f32 / (self.height.max(1) as f32 * 9.0);
        // 4 corners
        if self.height > 0 && self.width > 0 {
            features[59] = self.data[0][0] as f32 / 9.0; // TL
            features[60] = self.data[0][self.width - 1] as f32 / 9.0; // TR
            features[61] = self.data[self.height - 1][0] as f32 / 9.0; // BL
            features[62] = self.data[self.height - 1][self.width - 1] as f32 / 9.0; // BR
        }
        // Edge transitions
        features[63] = self.count_edge_transitions() as f32 / 40.0;
        // Border fill ratio
        if self.height > 2 && self.width > 2 {
            let border_cells = 2 * (self.width + self.height) - 4;
            let filled_border: u32 = (0..self.width)
                .map(|c| self.data[0][c] as u32 + self.data[self.height-1][c] as u32)
                .sum::<u32>()
                + (1..self.height-1)
                .map(|r| self.data[r][0] as u32 + self.data[r][self.width-1] as u32)
                .sum::<u32>();
            features[64] = filled_border as f32 / (border_cells.max(1) as f32 * 9.0);
        }
        // Center-to-border ratio
        if self.height > 4 && self.width > 4 {
            let center_sum: u32 = (center_r-2..center_r+3)
                .flat_map(|r| (center_c-2..center_c+3)
                    .map(move |c| self.data[r][c] as u32))
                .sum();
            features[65] = center_sum as f32 / (25.0 * 9.0);
        }

        // === [75..99] Symmetrie-Features ===
        // Horizontal symmetry
        let mut h_sym = 0.0f32;
        let h_pairs = (self.height / 2) * self.width;
        for r in 0..self.height / 2 {
            for c in 0..self.width {
                let top = self.data[r][c];
                let bot = self.data[self.height - 1 - r][c];
                if top == bot { h_sym += 1.0; }
            }
        }
        features[75] = if h_pairs > 0 { h_sym / h_pairs as f32 } else { 0.0 };

        // Vertical symmetry
        let mut v_sym = 0.0f32;
        let v_pairs = self.height * (self.width / 2);
        for r in 0..self.height {
            for c in 0..self.width / 2 {
                let left = self.data[r][c];
                let right = self.data[r][self.width - 1 - c];
                if left == right { v_sym += 1.0; }
            }
        }
        features[76] = if v_pairs > 0 { v_sym / v_pairs as f32 } else { 0.0 };

        // Diagonal symmetry (TL-BR)
        let mut d_sym = 0.0f32;
        let diag_len = self.height.min(self.width);
        for i in 0..diag_len {
            if self.data[i][i] == self.data[i][i] { d_sym += 1.0; } // always true
        }
        for r in 0..self.height {
            for c in 0..self.width {
                if r != c && c < self.width && r < self.height {
                    let a = self.data[r][c];
                    let b = if c < self.height && r < self.width { self.data[c][r] } else { a };
                    if a == b { d_sym += 1.0; }
                }
            }
        }
        features[77] = d_sym / (self.width * self.height).max(1) as f32;

        // Unique color count
        let mut seen = [false; 10];
        let mut unique = 0;
        for row in &self.data {
            for &c in row {
                if !seen[c as usize] {
                    seen[c as usize] = true;
                    unique += 1;
                }
            }
        }
        features[78] = unique as f32 / 10.0;

        // Aspect ratio (width / height, normalized)
        features[79] = self.width as f32 / self.height.max(1) as f32;

        // Padding for remaining indices [80..99] stays 0.0
        features
    }

    /// Counts the number of color transitions along grid edges.
    fn count_edge_transitions(&self) -> usize {
        let mut transitions = 0;
        // Horizontal
        for r in 0..self.height {
            for c in 0..self.width - 1 {
                if self.data[r][c] != self.data[r][c + 1] {
                    transitions += 1;
                }
            }
        }
        // Vertical
        for r in 0..self.height - 1 {
            for c in 0..self.width {
                if self.data[r][c] != self.data[r + 1][c] {
                    transitions += 1;
                }
            }
        }
        transitions
    }

    /// Converts the grid to a `HyperbolicPoint` by flattening, normalizing,
    /// truncating/padding to `target_dim`, and projecting onto the Poincaré ball
    /// via the exponential map at origin.
    ///
    /// All hyperbolic operations verify `norm() < 1.0` internally.
    pub fn to_hyperbolic(
        &self,
        ball: &PoincareBall,
        target_dim: usize,
    ) -> Result<HyperbolicPoint, String> {
        let mut flat = self.flatten();
        if target_dim > 0 && flat.len() != target_dim {
            if flat.len() > target_dim {
                flat.truncate(target_dim);
            } else {
                flat.resize(target_dim, 0.0);
            }
        }
        let arr = Array1::from_vec(flat);
        ball.exp_map_origin(&arr).map_err(|e| format!("{:?}", e))
    }

    /// Pads the grid to a target size (width × height) with zeros.
    /// If the grid is already larger, it will be truncated.
    pub fn pad_to(&self, target_width: usize, target_height: usize) -> Self {
        let mut data = vec![vec![0u8; target_width]; target_height];
        for (i, row) in self.data.iter().enumerate() {
            if i >= target_height { break; }
            for (j, &val) in row.iter().enumerate() {
                if j >= target_width { break; }
                data[i][j] = val;
            }
        }
        Self { data, width: target_width, height: target_height }
    }

    /// Resizes grid to exactly target_width × target_height.
    /// Uses nearest-neighbor upsampling or truncation.
    pub fn resize(&self, target_width: usize, target_height: usize) -> Self {
        if self.width == target_width && self.height == target_height {
            return self.clone();
        }
        
        let mut data = vec![vec![0u8; target_width]; target_height];
        let x_ratio = self.width as f64 / target_width as f64;
        let y_ratio = self.height as f64 / target_height as f64;
        
        for y in 0..target_height {
            let src_y = (y as f64 * y_ratio).min((self.height - 1) as f64) as usize;
            for x in 0..target_width {
                let src_x = (x as f64 * x_ratio).min((self.width - 1) as f64) as usize;
                data[y][x] = self.data[src_y][src_x];
            }
        }
        
        Self { data, width: target_width, height: target_height }
    }
}

impl Default for ArcGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// A single ARC-AGI task: train pairs + test inputs/outputs.
#[derive(Debug, Clone)]
pub struct ArcTask {
    pub id: String,
    pub train_pairs: Vec<(ArcGrid, ArcGrid)>,
    pub test_inputs: Vec<ArcGrid>,
    /// `None` for evaluation sets where the output is absent.
    pub test_outputs: Vec<Option<ArcGrid>>,
}

impl ArcTask {
    /// Creates an empty task.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            train_pairs: Vec::new(),
            test_inputs: Vec::new(),
            test_outputs: Vec::new(),
        }
    }

    /// Parses a task from a JSON value.
    pub fn from_json(id: impl Into<String>, value: &Value) -> Result<Self, String> {
        let id = id.into();
        let obj = value.as_object().ok_or("Expected JSON object for task")?;

        let train = obj.get("train").ok_or("Missing 'train' field")?;
        let train_arr = train.as_array().ok_or("'train' must be an array")?;

        let mut train_pairs = Vec::with_capacity(train_arr.len());
        for pair in train_arr {
            let pair_obj = pair.as_object().ok_or("Train pair must be object")?;
            let input = pair_obj.get("input").ok_or("Missing 'input' in train pair")?;
            let output = pair_obj.get("output").ok_or("Missing 'output' in train pair")?;
            train_pairs.push((
                ArcGrid::from_json(input)?,
                ArcGrid::from_json(output)?,
            ));
        }

        let test = obj.get("test").ok_or("Missing 'test' field")?;
        let test_arr = test.as_array().ok_or("'test' must be an array")?;

        let mut test_inputs = Vec::with_capacity(test_arr.len());
        let mut test_outputs = Vec::with_capacity(test_arr.len());

        for case in test_arr {
            let case_obj = case.as_object().ok_or("Test case must be object")?;
            let input = case_obj.get("input").ok_or("Missing 'input' in test case")?;
            test_inputs.push(ArcGrid::from_json(input)?);

            let output = case_obj.get("output");
            if let Some(output_val) = output {
                test_outputs.push(Some(ArcGrid::from_json(output_val)?));
            } else {
                test_outputs.push(None);
            }
        }

        Ok(Self {
            id,
            train_pairs,
            test_inputs,
            test_outputs,
        })
    }

    /// Returns the total number of train pairs.
    pub fn train_len(&self) -> usize {
        self.train_pairs.len()
    }

    /// Returns the total number of test cases.
    pub fn test_len(&self) -> usize {
        self.test_inputs.len()
    }

    /// Creates `ConceptNode` entries for each unique color in the task
    /// and adds them to the given `ConceptGraph`.
    ///
    /// Colors are mapped to 1D positions in the Poincaré ball:
    /// color `c` → position `(c / 10.0) * 0.8 - 0.4` (range [-0.4, 0.4]).
    pub fn to_concept_nodes(&self, graph: &mut ConceptGraph) -> Result<(), String> {
        let mut colors: Vec<u8> = self
            .train_pairs
            .iter()
            .flat_map(|(i, o)| i.unique_colors().into_iter().chain(o.unique_colors()))
            .collect();
        colors.sort_unstable();
        colors.dedup();

        for &color in &colors {
            let pos = (color as f64 / 10.0) * 0.8 - 0.4;
            let coords = ndarray::array![pos];
            let point = HyperbolicPoint::new(coords).map_err(|e| format!("{:?}", e))?;
            let label = format!("arc_color_{}", color);
            graph.add_concept(&label, point);
        }

        Ok(())
    }
}

/// A dataset of ARC-AGI tasks loaded from a directory of JSON files.
#[derive(Debug, Clone, Default)]
pub struct ArcDataset {
    pub tasks: Vec<ArcTask>,
}

impl ArcDataset {
    /// Creates an empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads all ARC-AGI tasks from a directory of JSON files.
    pub fn load_from_directory(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        if !dir.exists() || !dir.is_dir() {
            return Err(format!("Path '{}' is not a directory", dir.display()));
        }

        let mut tasks = Vec::new();
        let mut entries: Vec<_> = fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
            .collect();

        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let data = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let value: Value = serde_json::from_str(&data).map_err(|e| e.to_string())?;

            match ArcTask::from_json(id.clone(), &value) {
                Ok(task) => tasks.push(task),
                Err(e) => eprintln!("Skipping invalid ARC file '{}': {}", id, e),
            }
        }

        Ok(Self { tasks })
    }

    /// Loads a single task by ID (filename stem without `.json`) from the directory.
    pub fn load_task(dir: impl AsRef<Path>, task_id: &str) -> Result<ArcTask, String> {
        let dir = dir.as_ref();
        let path = dir.join(format!("{}.json", task_id));

        if !path.exists() {
            return Err(format!("Task file '{}' not found", path.display()));
        }

        let data = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let value: Value = serde_json::from_str(&data).map_err(|e| e.to_string())?;

        ArcTask::from_json(task_id, &value)
    }

    /// Returns the number of tasks in the dataset.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Returns `true` if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_grid_from_json_simple() {
        let json = serde_json::json!([[0, 1, 2], [3, 4, 5]]);
        let grid = ArcGrid::from_json(&json).unwrap();
        assert_eq!(grid.width, 3);
        assert_eq!(grid.height, 2);
        assert_eq!(grid.data[0], vec![0, 1, 2]);
        assert_eq!(grid.data[1], vec![3, 4, 5]);
    }

    #[test]
    fn arc_grid_unique_colors() {
        let json = serde_json::json!([[0, 1, 2], [1, 2, 3]]);
        let grid = ArcGrid::from_json(&json).unwrap();
        let colors = grid.unique_colors();
        assert_eq!(colors, vec![0, 1, 2, 3]);
    }

    #[test]
    fn arc_grid_normalize() {
        let json = serde_json::json!([[0, 5], [9, 9]]);
        let grid = ArcGrid::from_json(&json).unwrap();
        let norm = grid.normalize();
        assert!((norm[0][0] - 0.0).abs() < 1e-10);
        assert!((norm[0][1] - 5.0 / 9.0).abs() < 1e-10);
        assert!((norm[1][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn arc_grid_to_hyperbolic_valid_norm() {
        let json = serde_json::json!([[0, 1], [2, 3]]);
        let grid = ArcGrid::from_json(&json).unwrap();
        let ball = PoincareBall::new(1.0);
        let point = grid.to_hyperbolic(&ball, 4).unwrap();
        assert!(point.euclidean_norm() < 1.0);
        assert_eq!(point.coords.len(), 4);
    }

    #[test]
    fn arc_grid_to_hyperbolic_truncates() {
        let json = serde_json::json!([[0, 1, 2, 3, 4]]);
        let grid = ArcGrid::from_json(&json).unwrap();
        let ball = PoincareBall::new(1.0);
        let point = grid.to_hyperbolic(&ball, 2).unwrap();
        assert_eq!(point.coords.len(), 2);
        assert!(point.euclidean_norm() < 1.0);
    }

    #[test]
    fn arc_grid_to_hyperbolic_pads() {
        let json = serde_json::json!([[0]]);
        let grid = ArcGrid::from_json(&json).unwrap();
        let ball = PoincareBall::new(1.0);
        let point = grid.to_hyperbolic(&ball, 8).unwrap();
        assert_eq!(point.coords.len(), 8);
        assert!(point.euclidean_norm() < 1.0);
    }

    #[test]
    fn arc_task_from_json() {
        let json = serde_json::json!({
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
            ],
            "test": [
                {"input": [[0, 0], [1, 1]]}
            ]
        });
        let task = ArcTask::from_json("test", &json).unwrap();
        assert_eq!(task.id, "test");
        assert_eq!(task.train_pairs.len(), 1);
        assert_eq!(task.test_inputs.len(), 1);
        assert_eq!(task.test_outputs.len(), 1);
        assert!(task.test_outputs[0].is_none());
    }

    #[test]
    fn arc_task_from_json_with_output() {
        let json = serde_json::json!({
            "train": [
                {"input": [[0, 1]], "output": [[1, 0]]}
            ],
            "test": [
                {"input": [[1, 1]], "output": [[0, 0]]}
            ]
        });
        let task = ArcTask::from_json("test2", &json).unwrap();
        assert_eq!(task.test_outputs.len(), 1);
        assert!(task.test_outputs[0].is_some());
        assert_eq!(task.test_outputs[0].as_ref().unwrap().data[0], vec![0, 0]);
    }

    #[test]
    fn arc_task_to_concept_nodes() {
        let json = serde_json::json!({
            "train": [
                {"input": [[0, 1]], "output": [[1, 0]]}
            ],
            "test": [
                {"input": [[1, 1]]}
            ]
        });
        let task = ArcTask::from_json("test3", &json).unwrap();
        let mut graph = ConceptGraph::new(1.0);
        task.to_concept_nodes(&mut graph).unwrap();

        assert_eq!(graph.nodes.len(), 2);
        assert!(graph.index.contains_key("arc_color_0"));
        assert!(graph.index.contains_key("arc_color_1"));
    }

    #[test]
    fn arc_dataset_load_from_directory() {
        let dir = std::env::temp_dir().join("goldsnnail_arc_test");
        fs::create_dir_all(&dir).unwrap();

        let task_json = serde_json::json!({
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}
            ],
            "test": [
                {"input": [[0, 0], [1, 1]]}
            ]
        });
        fs::write(dir.join("task1.json"), task_json.to_string()).unwrap();

        let dataset = ArcDataset::load_from_directory(&dir).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.tasks[0].id, "task1");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn arc_dataset_load_task() {
        let dir = std::env::temp_dir().join("goldsnnail_arc_test2");
        fs::create_dir_all(&dir).unwrap();

        let task_json = serde_json::json!({
            "train": [
                {"input": [[0]], "output": [[1]]}
            ],
            "test": [
                {"input": [[2]]}
            ]
        });
        fs::write(dir.join("mytask.json"), task_json.to_string()).unwrap();

        let task = ArcDataset::load_task(&dir, "mytask").unwrap();
        assert_eq!(task.id, "mytask");
        assert_eq!(task.train_pairs.len(), 1);

        fs::remove_dir_all(&dir).unwrap();
    }
}
