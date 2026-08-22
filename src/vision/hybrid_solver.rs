//! Hybrid Solver — Woche 1 Prototyp
//!
//! Kombiniert hyperbolic Router (k-NN) mit 3 expliziten Heuristiken:
//! - Identity
//! - Rotate90
//! - FlipHorizontal
//!
//! Einschraenkungen:
//! - Nur 10x10 Grids (Encoder-Beschraenkung)
//! - Lineare k-NN Suche
//! - Encoder sollte trainiert sein fuer beste Router-Performance

use crate::geometry::HyperbolicPoint;
use crate::vision::{
    ArcGrid, ArcTask, ArcDataset, GridEncoder, transform_codec, transform_memory, committee,
};

/// Deterministische Grid-Operationen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Heuristic {
    Identity,
    Rotate90,
    FlipHorizontal,
    // Transform-Codec heuristics (Dynamic, from training)
    TransformCodec,
}

impl Heuristic {
    /// Testet, ob diese Heuristik den Input exakt auf den Output abbildet
    pub fn works_on(&self, input: &ArcGrid, output: &ArcGrid) -> bool {
        let transformed = self.apply(input);
        grid_eq(&transformed, output)
    }

    /// Wendet die Heuristik auf ein Grid an
    pub fn apply(&self, grid: &ArcGrid) -> ArcGrid {
        match self {
            Heuristic::Identity => grid_clone(grid),
            Heuristic::Rotate90 => rotate90(grid),
            Heuristic::FlipHorizontal => flip_horizontal(grid),
            Heuristic::TransformCodec => grid_clone(grid), // Placeholder - real apply needs context
        }
    }
}

/// Prueft, ob ein Grid exakt 10x10 ist
pub fn is_10x10(grid: &ArcGrid) -> bool {
    grid.width == 10 && grid.height == 10
}

/// Vergleicht zwei Grids pixelweise
fn grid_eq(a: &ArcGrid, b: &ArcGrid) -> bool {
    a.width == b.width && a.height == b.height && a.data == b.data
}

/// Klont ein ArcGrid
fn grid_clone(grid: &ArcGrid) -> ArcGrid {
    ArcGrid {
        data: grid.data.clone(),
        width: grid.width,
        height: grid.height,
    }
}

/// Rotiert ein Grid 90 Grad im Uhrzeigersinn
fn rotate90(grid: &ArcGrid) -> ArcGrid {
    let mut new_data = vec![vec![0u8; grid.height]; grid.width];
    for r in 0..grid.height {
        for c in 0..grid.width {
            new_data[c][grid.height - 1 - r] = grid.data[r][c];
        }
    }
    ArcGrid {
        data: new_data,
        width: grid.height,
        height: grid.width,
    }
}

/// Spiegelt ein Grid horizontal (links/rechts tauschen)
fn flip_horizontal(grid: &ArcGrid) -> ArcGrid {
    let mut new_data = grid.data.clone();
    for row in &mut new_data {
        row.reverse();
    }
    ArcGrid {
        data: new_data,
        width: grid.width,
        height: grid.height,
    }
}

/// Der Hybrid-Solver: Hyperbolic Router + explizite Heuristiken + Transform-Codec
pub struct HybridSolver<'a> {
    encoder: &'a GridEncoder,
    all_tasks: &'a [ArcTask],
    exclude_id: Option<String>,
    heuristics: Vec<Heuristic>,
    k: usize,
    // Transform-Codec memory
    transform_memory: transform_memory::TransformMemory,
}

impl<'a> HybridSolver<'a> {
    pub fn new(encoder: &'a GridEncoder, all_tasks: &'a [ArcTask], k: usize) -> Self {
        // Build transform memory from all training tasks
        let mut transform_memory = transform_memory::TransformMemory::new(5);
        for task in all_tasks {
            for (ex_idx, (input, output)) in task.train_pairs.iter().enumerate() {
                if transform_codec::is_10x10(input) && transform_codec::is_10x10(output) {
                    let code = transform_codec::extract_transform(input, output);
                    if code.residual < 0.1 { // Only store high-confidence transforms
                        transform_memory.add(code, task.id.clone(), ex_idx);
                    }
                }
            }
        }
        
        Self {
            encoder,
            all_tasks,
            exclude_id: None,
            heuristics: vec![
                Heuristic::Identity,
                Heuristic::Rotate90,
                Heuristic::FlipHorizontal,
            ],
            k,
            transform_memory,
        }
    }

    /// Setzt den Task, der ausgeschlossen werden soll (Leave-One-Out)
    pub fn exclude_task(&mut self, task_id: String) {
        self.exclude_id = Some(task_id);
    }

    /// Findet die erste Heuristik, die auf einem Train-Paar funktioniert
    pub fn find_working_heuristic(&self, task: &ArcTask) -> Option<Heuristic> {
        for (input_grid, output_grid) in &task.train_pairs {
            if !is_10x10(input_grid) || !is_10x10(output_grid) {
                continue;
            }

            for h in &self.heuristics {
                if h.works_on(input_grid, output_grid) {
                    return Some(*h);
                }
            }
        }
        None
    }

    /// Findet den Transform-Codec, der auf einem Train-Paar funktioniert
    fn find_working_transform(&self, task: &ArcTask) -> Option<transform_codec::TransformCode> {
        for (input_grid, output_grid) in &task.train_pairs {
            if !is_10x10(input_grid) || !is_10x10(output_grid) {
                continue;
            }
            
            let code = transform_codec::extract_transform(input_grid, output_grid);
            if code.residual < 0.1 {
                return Some(code);
            }
        }
        None
    }

    /// Loest einen neuen Task durch k-NN + Heuristik-Transfer
    pub fn solve(&self, task: &ArcTask) -> Option<ArcGrid> {
        let test_input = task.test_inputs.first()?;

        if !is_10x10(test_input) {
            return None;
        }

        let test_point = self.encoder.encode(test_input).ok()?;
        let neighbors = self.find_nearest_train_tasks(&test_point);

        // Try static heuristics first
        for neighbor in &neighbors {
            if let Some(heuristic) = self.find_working_heuristic(neighbor) {
                let result = heuristic.apply(test_input);
                return Some(result);
            }
        }

        // Try Transform-Codec from nearest neighbors
        for neighbor in &neighbors {
            if let Some(code) = self.find_working_transform(neighbor) {
                if let Some(result) = transform_codec::apply_transform(test_input, &code) {
                    return Some(result);
                }
            }
        }

        None
    }

    /// Lineare k-NN ueber Train-Task-Embeddings
    fn find_nearest_train_tasks(&self, point: &HyperbolicPoint) -> Vec<&'a ArcTask> {
        let mut scored: Vec<(f64, &'a ArcTask)> = self
            .all_tasks
            .iter()
            .filter(|task| {
                if let Some(ref exclude) = self.exclude_id {
                    task.id != *exclude
                } else {
                    true
                }
            })
            .filter_map(|task| {
                let (input_grid, _) = task.train_pairs.first()?;
                if !is_10x10(input_grid) {
                    return None;
                }
                let emb = self.encoder.encode(input_grid).ok()?;
                let dist = point.euclidean_norm() - emb.euclidean_norm();
                Some((dist.abs(), task))
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(self.k).map(|(_, task)| task).collect()
    }
}

/// Leave-One-Out Evaluation auf 10x10-Tasks
pub struct EvaluationResult {
    pub total: usize,
    pub attempted: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub attempt_rate: f64,
}

pub fn evaluate_hybrid_solver(
    dataset: &ArcDataset,
    encoder: &GridEncoder,
    k: usize,
    n_tasks: usize,
) -> EvaluationResult {
    let tasks_10x10: Vec<&ArcTask> = dataset
        .tasks
        .iter()
        .filter(|t| is_task_10x10(t))
        .take(n_tasks)
        .collect();

    println!(
        "Hybrid Solver: {} 10x10 Tasks gefunden (gefordert: {})",
        tasks_10x10.len(),
        n_tasks
    );

    let mut correct = 0;
    let mut attempted = 0;
    let mut total = 0;

    for held_out in &tasks_10x10 {
        total += 1;

        let mut solver = HybridSolver::new(encoder, &dataset.tasks, k);
        solver.exclude_task(held_out.id.clone());

        if let Some(prediction) = solver.solve(held_out) {
            attempted += 1;
            let true_output = match held_out.test_outputs.first().and_then(|o| o.as_ref()) {
                Some(g) => g,
                None => continue,
            };
            if grid_eq(&prediction, true_output) {
                correct += 1;
                println!("Task {}: ✅ CORRECT", held_out.id);
            } else {
                println!(
                    "Task {}: ❌ WRONG (Heuristik fand, aber Output mismatch)",
                    held_out.id
                );
            }
        } else {
            println!(
                "Task {}: ⚠️ NO SOLUTION (kein Nachbar mit passender Heuristik)",
                held_out.id
            );
        }
    }

    EvaluationResult {
        total,
        attempted,
        correct,
        accuracy: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
        attempt_rate: if total > 0 {
            attempted as f64 / total as f64
        } else {
            0.0
        },
    }
}

/// Prueft, ob ein Task ausschließlich 10x10 Grids verwendet
fn is_task_10x10(task: &ArcTask) -> bool {
    task.train_pairs.iter().all(|(inp, out)| {
        is_10x10(inp) && is_10x10(out)
    }) && task.test_inputs.iter().all(is_10x10)
}
