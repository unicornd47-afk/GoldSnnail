use clap::{Arg, Command};
use std::env;
use std::process;

mod commands;
mod cli;
mod gpu;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let matches = Command::new("goldworm")
        .version("0.3.0")
        .author("GoldWorm Research Team")
        .about("GoldWorm Desktop App")
        .arg(
            Arg::new("cli")
                .long("cli")
                .help("Run in CLI mode instead of GUI")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("gpu")
                .long("gpu")
                .help("Run GPU detection")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    if matches.get_flag("cli") {
        let args: Vec<String> = env::args().skip(2).collect();
        if let Err(e) = cli::run_cli_mode_from(args) {
            eprintln!("CLI mode error: {}", e);
            process::exit(1);
        }
        process::exit(0);
    }

    if matches.get_flag("gpu") {
        match gpu::detect_gpu() {
            Ok(info) => println!("GPU detected: {}", info),
            Err(e) => {
                eprintln!("GPU detection failed: {}", e);
                process::exit(1);
            }
        }
        process::exit(0);
    }

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::init_snn_core,
            commands::step_snn,
            commands::solve_arc_task,
            commands::list_arc_tasks,
            commands::get_arc_task,
            commands::apply_arc_token,
            commands::run_arc_benchmark,
            commands::get_monster_points,
            commands::get_status,
            commands::list_supported_types,
            commands::encode_spike_stream,
            commands::encode_state_arena,
            commands::encode_arc_grid,
            commands::encode_hyperbolic_point,
            commands::encode_quaternion,
            commands::encode_weight_matrix,
            commands::encode_dvs_batch,
            commands::encode_lexicon_token,
            commands::decode_payload,
            commands::decode_to_json
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run()
}


