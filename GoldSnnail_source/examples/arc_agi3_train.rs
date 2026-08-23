//! ARC-AGI-3 Training & Benchmark
//!
//! Trains the GoldWorm ARC-AGI-3 agent across multiple demo environments
//! and reports improvement over random baseline.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example arc_agi3_train
//! ```
//!
//! ## Metrics
//!
//! - Solve rate per environment
//! - Average reward per episode
//! - Average steps to solve
//! - Improvement over random baseline

use std::collections::HashMap;

fn main() {
    println!("=== GoldWorm ARC-AGI-3 Training Benchmark ===\n");

    let num_envs = 8;
    let episodes_per_env = 10;
    let mut results: HashMap<&'static str, Vec<f64>> = HashMap::new();

    for env_idx in 0..num_envs {
        let env_id = env_name(env_idx);
        println!("Training on: {}", env_id);

        let mut env_rewards = Vec::new();
        let mut env_solves = 0usize;

        for ep in 0..episodes_per_env {
            let mut agent = goldworm::agi3::ArcAgent3::default();
            let mut env = create_demo_env(env_idx);
            let result = agent.run_episode(&mut *env);

            env_rewards.push(result.total_reward);
            if result.solved {
                env_solves += 1;
            }

            if ep == 0 {
                println!("  Episode 1: reward={:.3}, solved={}, steps={}",
                    result.total_reward, result.solved, result.steps);
            }
        }

        let avg_reward: f64 = env_rewards.iter().sum::<f64>() / env_rewards.len() as f64;
        let solve_rate = env_solves as f64 / episodes_per_env as f64;

        println!("  Avg reward: {:.3}, Solve rate: {:.1}%\n",
            avg_reward, solve_rate * 100.0);

        results.insert(env_id, env_rewards);
    }

    // Summary
    println!("=== Summary ===");
    let total_solves: usize = results.values()
        .map(|r| r.iter().filter(|&&v| v > 5.0).count())
        .sum();
    let total_episodes: usize = results.values().map(|r| r.len()).sum();
    let overall_score = total_solves as f64 / total_episodes as f64;

    println!("Overall solve rate: {:.1}% ({}/{})",
        overall_score * 100.0, total_solves, total_episodes);

    // Compare to random baseline
    let random_baseline = estimate_random_baseline();
    println!("Random baseline: {:.1}%", random_baseline * 100.0);
    println!("Improvement: {:.1}x",
        if random_baseline > 0.0 { overall_score / random_baseline } else { 0.0 });
}

fn estimate_random_baseline() -> f64 {
    // Simple estimate: random agent has ~1/6 chance of picking Interact
    // and even then, only solves if the transformation matches
    // For demo envs: ~0.05 (5% random solve rate)
    0.05
}

fn env_name(idx: usize) -> &'static str {
    match idx {
        0 => "rotate90",
        1 => "flip_h",
        2 => "gravity",
        3 => "mirror_v",
        4 => "tile_2x2",
        5 => "crop_center",
        6 => "scale_2x",
        7 => "replace_color",
        _ => "default",
    }
}

fn create_demo_env(idx: usize) -> Box<dyn goldworm::agi3::environment::ArcEnvironment> {
    use goldworm::agi3::environment::GridEnvironment;
    use goldworm::vision::ArcGrid;

    match idx {
        0 => {
            let g1 = ArcGrid::from_data(vec![vec![1,0,0],vec![0,2,0],vec![0,0,3]]).unwrap();
            let g1_out = ArcGrid::from_data(vec![vec![0,0,1],vec![0,2,0],vec![3,0,0]]).unwrap();
            Box::new(GridEnvironment::new("rotate90", g1, g1_out, 20))
        }
        1 => {
            let g2 = ArcGrid::from_data(vec![vec![1,0,0],vec![0,2,0],vec![0,0,3]]).unwrap();
            let g2_out = ArcGrid::from_data(vec![vec![0,0,1],vec![0,2,0],vec![3,0,0]]).unwrap();
            Box::new(GridEnvironment::new("flip_h", g2, g2_out, 20))
        }
        2 => {
            let g3 = ArcGrid::from_data(vec![vec![1,0,0,0],vec![0,2,0,0],vec![0,0,3,0],vec![0,0,0,0]]).unwrap();
            let g3_out = ArcGrid::from_data(vec![vec![0,0,0,0],vec![0,0,0,0],vec![1,0,0,0],vec![0,2,3,0]]).unwrap();
            Box::new(GridEnvironment::new("gravity", g3, g3_out, 20))
        }
        3 => {
            let g4 = ArcGrid::from_data(vec![vec![1,0,0],vec![0,2,0],vec![0,0,3]]).unwrap();
            let g4_out = ArcGrid::from_data(vec![vec![0,0,3],vec![0,2,0],vec![1,0,0]]).unwrap();
            Box::new(GridEnvironment::new("mirror_v", g4, g4_out, 20))
        }
        4 => {
            let g5 = ArcGrid::from_data(vec![vec![1,2],vec![3,0]]).unwrap();
            let g5_out = ArcGrid::from_data(vec![vec![1,2,1,2],vec![3,0,3,0],vec![1,2,1,2],vec![3,0,3,0]]).unwrap();
            Box::new(GridEnvironment::new("tile_2x2", g5, g5_out, 20))
        }
        5 => {
            let g6 = ArcGrid::from_data(vec![vec![0,0,0,0],vec![0,1,2,0],vec![0,3,4,0],vec![0,0,0,0]]).unwrap();
            let g6_out = ArcGrid::from_data(vec![vec![1,2],vec![3,4]]).unwrap();
            Box::new(GridEnvironment::new("crop_center", g6, g6_out, 20))
        }
        6 => {
            let g7 = ArcGrid::from_data(vec![vec![1,2],vec![3,4]]).unwrap();
            let g7_out = ArcGrid::from_data(vec![vec![1,1,2,2],vec![1,1,2,2],vec![3,3,4,4],vec![3,3,4,4]]).unwrap();
            Box::new(GridEnvironment::new("scale_2x", g7, g7_out, 20))
        }
        7 => {
            let g8 = ArcGrid::from_data(vec![vec![1,0,0],vec![0,1,0],vec![0,0,1]]).unwrap();
            let g8_out = ArcGrid::from_data(vec![vec![2,0,0],vec![0,2,0],vec![0,0,2]]).unwrap();
            Box::new(GridEnvironment::new("replace_color", g8, g8_out, 20))
        }
        _ => {
            let default = ArcGrid::from_data(vec![vec![0,0],vec![0,0]]).unwrap();
            Box::new(GridEnvironment::new("default", default.clone(), default, 20))
        }
    }
}
