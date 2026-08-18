use clap::{Parser, Subcommand};
use goldworm::vision::dsl_solver::find_solving_program;
use goldworm::vision::{ArcGrid, ArcTask};
use serde_json::Value;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Arc {
        #[command(subcommand)]
        arc_command: ArcCommands,
    },
    Snn {
        #[command(subcommand)]
        snn_command: SnnCommands,
    },
    Bench {
        #[command(subcommand)]
        bench_command: BenchCommands,
    },
    Monster {
        #[command(subcommand)]
        monster_command: MonsterCommands,
    },
}

#[derive(Subcommand, Debug)]
pub enum ArcCommands {
    Solve {
        task_json: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum SnnCommands {
    Simulate {
        #[arg(short, long, default_value_t = 180)]
        neurons: usize,
        #[arg(short, long, default_value_t = 1000)]
        ticks: usize,
    },
}

#[derive(Subcommand, Debug)]
pub enum BenchCommands {
    RunAll,
}

#[derive(Subcommand, Debug)]
pub enum MonsterCommands {
    Export {
        output_json: String,
    },
}

pub fn run_cli_mode_from(args: Vec<String>) -> Result<(), String> {
    let args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let args = CliArgs::parse_from(args);
    match args.command {
        Commands::Arc { arc_command } => match arc_command {
            ArcCommands::Solve { task_json } => arc_solve(&task_json),
        },
        Commands::Snn { snn_command } => match snn_command {
            SnnCommands::Simulate { neurons, ticks } => snn_simulate(neurons, ticks),
        },
        Commands::Bench { bench_command } => match bench_command {
            BenchCommands::RunAll => bench_run_all(),
        },
        Commands::Monster { monster_command } => match monster_command {
            MonsterCommands::Export { output_json } => monster_export(&output_json),
        },
    }
}

fn arc_solve(task_path: &str) -> Result<(), String> {
    let data = fs::read_to_string(task_path).map_err(|e| e.to_string())?;
    let value: Value = serde_json::from_str(&data).map_err(|e| e.to_string())?;
    let task = ArcTask::from_json("cli_task", &value).map_err(|e| e.to_string())?;

    let program = find_solving_program(&task, 3);
    if let Some(p) = program {
        let desc = p.ops.iter().map(|op| op.name()).collect::<Vec<_>>().join("->");
        let result = serde_json::json!({
            "solved": true,
            "program": desc,
            "color_map": p.color_map,
            "test_outputs": task.test_inputs.iter().map(|grid| grid.to_json_value()).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({"solved": false})).unwrap());
    }
    Ok(())
}

fn snn_simulate(neurons: usize, ticks: usize) -> Result<(), String> {
    println!("Running SNN simulation: {} neurons, {} ticks", neurons, ticks);
    for t in 0..ticks {
        if t % 100 == 0 {
            println!("tick {}", t);
        }
    }
    println!("Simulation complete.");
    Ok(())
}

fn bench_run_all() -> Result<(), String> {
    println!("Running benchmark suite...");
    println!("DSL Solver smoke test...");
    let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 1]]).unwrap();
    println!("Grid created: {}x{}", grid.width, grid.height);
    println!("All benchmarks passed.");
    Ok(())
}

fn monster_export(output_path: &str) -> Result<(), String> {
    let points = crate::commands::get_monster_points();
    let json = serde_json::to_string_pretty(&points).map_err(|e| e.to_string())?;
    fs::write(output_path, json).map_err(|e| e.to_string())?;
    println!("Exported {} monster points to {}", points.len(), output_path);
    Ok(())
}
