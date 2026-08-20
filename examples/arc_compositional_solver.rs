//! ARC Compositional Solver — CLI Example
//!
//! Standalone CLI for solving ARC tasks with the compositional solver.
//! Supports single-task solving, batch benchmarking, and TUI output.
//!
//! Usage:
//!   cargo run --example arc_compositional_solver -- <task_id>
//!   cargo run --example arc_compositional_solver -- --benchmark <dir> [max_depth]
//!   cargo run --example arc_compositional_solver -- --list <dir>

use goldworm::arc_apply::apply_program;
use goldworm::arc_program::{ArcOpCode, ArcOpToken, ArcProgram};
use goldworm::arc_search::{search_program, SearchConfig, SearchResult};
use goldworm::vision::{ArcDataset, ArcTask};

use std::path::PathBuf;
use std::time::{Duration, Instant};

// ─── CLI Argument Parsing ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Mode {
    Solve { task_id: Option<String> },
    Benchmark { dir: PathBuf, max_depth: usize },
    List { dir: PathBuf },
}

impl Mode {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        if args.len() == 1 {
            print_usage();
            std::process::exit(0);
        }

        if args[1] == "--benchmark" {
            let dir = args.get(2).map(|s| PathBuf::from(s)).unwrap_or_else(|| {
                eprintln!("Error: --benchmark requires a directory path");
                std::process::exit(1);
            });
            let max_depth = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
            return Mode::Benchmark { dir, max_depth };
        }

        if args[1] == "--list" {
            let dir = args.get(2).map(|s| PathBuf::from(s)).unwrap_or_else(|| {
                eprintln!("Error: --list requires a directory path");
                std::process::exit(1);
            });
            return Mode::List { dir };
        }

        Mode::Solve {
            task_id: Some(args[1].clone()),
        }
    }
}

fn print_usage() {
    println!("ARC Compositional Solver");
    println!();
    println!("USAGE:");
    println!("  cargo run --example arc_compositional_solver -- <task_id>");
    println!("  cargo run --example arc_compositional_solver -- --benchmark <dir> [max_depth]");
    println!("  cargo run --example arc_compositional_solver -- --list <dir>");
    println!();
    println!("MODES:");
    println!("  <task_id>     Solve a single task by ID");
    println!("  --benchmark   Run benchmark on all tasks in a directory");
    println!("  --list        List all task IDs in a directory");
    println!();
    println!("EXAMPLES:");
    println!("  cargo run --example arc_compositional_solver -- 007bbfb7");
    println!("  cargo run --example arc_compositional_solver -- --benchmark data/arc/training/ 3");
}

// ─── Solver Output ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SolveOutput {
    task_id: String,
    result: SearchResult,
    grid_in: Vec<Vec<u8>>,
    grid_out: Option<Vec<Vec<u8>>>,
}

fn solve_task(task: &ArcTask) -> SolveOutput {
    let config = SearchConfig::default();
    let result = search_program(task, config);
    let grid_in = task.test_inputs.first().map(|g| g.data.clone()).unwrap_or_default();
    let grid_out = result.program.as_ref().and_then(|prog| {
        task.test_inputs.first().and_then(|input| apply_program(input, prog))
    }).map(|g| g.data);

    SolveOutput {
        task_id: task.id.clone(),
        result,
        grid_in,
        grid_out,
    }
}

// ─── Benchmark Runner ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BenchmarkResult {
    total: usize,
    solved: usize,
    failed: usize,
    total_time: Duration,
    depth_distribution: [usize; 4], // depth 1-3, 4+
    avg_time: Duration,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n{}", "═".repeat(60));
        println!("BENCHMARK RESULTS");
        println!("{}", "═".repeat(60));
        println!("Total tasks:    {}", self.total);
        println!("Solved:         {} ({:.1}%)", self.solved, self.percentage(self.solved));
        println!("Failed:         {} ({:.1}%)", self.failed, self.percentage(self.failed));
        println!("Total time:     {:.2}s", self.total_time.as_secs_f64());
        println!("Avg time:       {:.0}ms", self.avg_time.as_millis());

        println!("\nDepth distribution:");
        for d in 0..4 {
            let label = if d == 3 { "Depth 4+".to_string() } else { format!("Depth {}", d + 1) };
            let pct = self.percentage(self.depth_distribution[d]);
            println!("  {:>10}: {} tasks ({:.1}%)", label, self.depth_distribution[d], pct);
        }
        println!("{}", "═".repeat(60));
    }

    fn percentage(&self, n: usize) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (n as f64 / self.total as f64) * 100.0
        }
    }
}

fn run_benchmark(dir: PathBuf, max_depth: usize) -> Result<BenchmarkResult, String> {
    println!("Loading dataset from: {}", dir.display());
    let dataset = ArcDataset::load_from_directory(dir)?;
    println!("Loaded {} tasks\n", dataset.tasks.len());

    let total = dataset.tasks.len();
    let mut solved = 0;
    let mut failed = 0;
    let mut depth_distribution = [0usize; 4];
    let start = Instant::now();

    for (i, task) in dataset.tasks.iter().enumerate() {
        if i % 50 == 0 {
            print!("Progress: {}/{} tasks...\r", i, total);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }

        let config = SearchConfig {
            max_depth,
            ..Default::default()
        };
        let result = search_program(task, config);

        if result.program.is_some() {
            solved += 1;
            let depth = result.program.unwrap().len();
            if depth <= 3 {
                depth_distribution[depth - 1] += 1;
            } else {
                depth_distribution[3] += 1;
            }
        } else {
            failed += 1;
        }
    }

    let total_time = start.elapsed();
    let avg_time = total_time / total as u32;

    println!("Progress: {}/{} tasks... done!", total, total);

    Ok(BenchmarkResult {
        total,
        solved,
        failed,
        total_time,
        depth_distribution,
        avg_time,
    })
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let mode = Mode::from_args();

    match mode {
        Mode::Solve { task_id } => {
            let task_id = task_id.unwrap_or_else(|| {
                eprintln!("Error: task ID required");
                std::process::exit(1);
            });

            // Try to load from default locations
            let dirs = vec![
                PathBuf::from("data/arc-agi-repo/data/training"),
                PathBuf::from("data/arc/training"),
            ];

            let task = dirs.iter().find_map(|dir| {
                match ArcDataset::load_task(dir, &task_id) {
                    Ok(task) => Some(task),
                    Err(_) => None,
                }
            });

            let task = match task {
                Some(t) => t,
                None => {
                    eprintln!("Error: task '{}' not found in any dataset directory", task_id);
                    std::process::exit(1);
                }
            };

            println!("Solving task: {}", task_id);
            println!("Train pairs: {}", task.train_pairs.len());
            println!("Test inputs: {}\n", task.test_inputs.len());

            let output = solve_task(&task);

            println!("Result: {}", if output.result.program.is_some() { "SOLVED" } else { "FAILED" });
            if let Some(ref prog) = output.result.program {
                println!("Program: {}", prog.describe());
                println!("Candidates evaluated: {}", output.result.candidates_evaluated);
            }

            if let Some(ref grid) = output.grid_out {
                println!("\nInput grid:");
                print_grid(&output.grid_in);
                println!("\nOutput grid:");
                print_grid(grid);
            }
        }

        Mode::Benchmark { dir, max_depth } => {
            match run_benchmark(dir, max_depth) {
                Ok(result) => {
                    result.print();
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Mode::List { dir } => {
            match ArcDataset::load_from_directory(dir) {
                Ok(dataset) => {
                    for task in &dataset.tasks {
                        println!("{}", task.id);
                    }
                    println!("\nTotal: {} tasks", dataset.tasks.len());
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

fn print_grid(grid: &[Vec<u8>]) {
    for row in grid {
        for &cell in row {
            print!("{}", cell);
        }
        println!();
    }
}
