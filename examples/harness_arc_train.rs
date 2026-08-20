//! Harness ARC Training — Self-Learning SNN on Identity ARC Tasks
//!
//! Demonstrates Phase 1+2 of the self-learning harness:
//! - Replay buffer for experience storage
//! - Forward engine with noise annealing
//! - Reward engine (extrinsic + intrinsic)
//! - Plasticity engine (R-STDP + homeostasis)
//! - Meta-controller (mode switching, checkpointing)
//! - Curriculum generator (task sequencing)
//!
//! Usage:
//!   cargo run --example harness_arc_train

use goldsnnail::{
    vision::{ArcDataset, ArcGrid, ArcTask},
    harness::*,
    swarm::snn_core::INPUT_NEURONS,
};
use std::time::Instant;

const MAX_EPISODES: usize = 50;
const STEPS_PER_EPISODE: usize = 8;
const REPLAY_CAPACITY: usize = 5_000;
const BATCH_SIZE: usize = 16;

fn main() {
    println!("=== GoldSnnail Self-Learning Harness — ARC Identity Training ===\n");

    // --- Load ARC dataset ---
    let dataset = match ArcDataset::load_from_directory("data/arc-agi-repo/data/training") {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("Failed to load ARC dataset: {}", e);
            eprintln!("Falling back to synthetic tasks.");
            create_synthetic_dataset()
        }
    };

    println!("Loaded {} ARC tasks", dataset.tasks.len());

    // --- Filter for identity-like tasks (input == output) ---
    let identity_tasks: Vec<_> = dataset.tasks
        .iter()
        .filter(|t| is_identity_task(t))
        .cloned()
        .collect();

    println!("Found {} identity tasks", identity_tasks.len());

    let tasks: Vec<ArcTask> = if identity_tasks.is_empty() {
        eprintln!("No identity tasks found. Using synthetic tasks.");
        create_synthetic_dataset().tasks
    } else {
        identity_tasks.into_iter().take(20).collect()
    };

    if tasks.is_empty() {
        eprintln!("No tasks available. Exiting.");
        return;
    }

    // --- Build curriculum ---
    let dataset_for_curriculum = ArcDataset { tasks: tasks.clone() };
    let mut curriculum = Curriculum::from_arc_dataset(&dataset_for_curriculum, 2);
    println!("Curriculum stages: {}", curriculum.stage_count());

    // --- Create harness components ---
    let mut forward = ForwardEngine::new(0.06, 0.3);
    let plasticity_config = PlasticityConfig {
        stdp_lr: 0.02,
        target_firing_rate: 8.0,
        pruning: true,
        prune_threshold: 0.03,
        ..Default::default()
    };
    let plasticity = PlasticityEngine::with_config(plasticity_config);
    let mut reward_engine = RewardEngine::new();
    let mut replay = ReplayBuffer::new(REPLAY_CAPACITY);
    let mut meta = MetaController::new();
    let mut tracker = EvalTracker::new(1000);

    // --- Training loop ---
    let start_time = Instant::now();

    for episode in 0..MAX_EPISODES {
        meta.tick();
        forward.noise_std = meta.noise_std();

        // Get next task from curriculum
        let task = match curriculum.next_task() {
            Some(t) => t,
            None => break,
        };

        // Train on all train pairs of this task
        let mut episode_reward = 0.0;
        let mut episode_correct = 0;
        let mut episode_total = 0;
        let mut prev_spike_count = 0;

        for (input_grid, expected_output) in &task.train_pairs {
            // Encode input grid to spike pattern
            let input_spikes = encode_grid_to_spikes(input_grid, INPUT_NEURONS);
            if input_spikes.is_empty() {
                continue;
            }

            // Run SNN forward pass
            forward.reset();
            let mut last_spikes = Vec::new();
            for _ in 0..STEPS_PER_EPISODE {
                let result = forward.step(&input_spikes);
                last_spikes = forward.output_spikes();
                prev_spike_count = result.spike_count;
            }

            // Decode output spikes to predicted grid
            let predicted = decode_spikes_to_grid(&last_spikes, expected_output.width, expected_output.height);

            // Compute reward
            let similarity = reward_engine.grid_similarity(&predicted, expected_output);
            let total_reward = reward_engine.compute_total(similarity, prev_spike_count, prev_spike_count);
            episode_reward += total_reward;

            // Track accuracy
            let cells = (expected_output.width * expected_output.height).max(1);
            let mut correct = 0;
            for r in 0..expected_output.height {
                for c in 0..expected_output.width {
                    if predicted.data[r][c] == expected_output.data[r][c] {
                        correct += 1;
                    }
                }
            }
            episode_correct += correct;
            episode_total += cells;

            // Store transition in replay buffer
            replay.push(Transition::new(
                input_spikes.clone(),
                last_spikes.clone(),
                total_reward,
                input_spikes.clone(),
                last_spikes.clone(),
                true,
            ));

            // Train plasticity on replay batch
            if replay.len() >= BATCH_SIZE {
                let batch = replay.sample(BATCH_SIZE, SamplingStrategy::Uniform);
                let refs: Vec<&Transition> = batch.iter().cloned().collect();
                plasticity.update_batch(&mut forward.core.swarm, &refs, meta.lr());
            }
        }

        let accuracy = if episode_total > 0 {
            episode_correct as f64 / episode_total as f64
        } else {
            0.0
        };

        // Evaluate metrics
        let active_synapses = plasticity.count_active_synapses(&forward.core.swarm);
        let metrics = EvalMetrics {
            epoch: episode as u64,
            mode: meta.mode,
            accuracy,
            avg_reward: episode_reward / task.train_pairs.len().max(1) as f64,
            avg_loss: 0.0,
            forgetting: tracker.detect_forgetting(5),
            firing_rate: forward.core.swarm.spike_count() as f32,
            active_synapses,
        };

        tracker.record(metrics.clone());

        // Meta-controller decisions
        let new_mode = meta.on_epoch_end(&metrics);
        if new_mode != meta.mode {
            println!("  [Mode Switch] {} -> {}", meta.mode.as_str(), new_mode.as_str());
        }

        if meta.should_checkpoint() {
            let ckpt_path = format!("benchmark_artifacts/runs/ckpt_epoch_{}", episode);
            meta.mark_checkpoint(&ckpt_path);
            println!("  [Checkpoint] Epoch {} -> {}", episode, ckpt_path);
        }

        // Logging
        if episode % 5 == 0 || episode == MAX_EPISODES - 1 {
            let elapsed = start_time.elapsed();
            println!(
                "[Epoch {:>3}] acc={:.1}% reward={:.3} synapses={} mode={} time={:.1}s",
                episode,
                accuracy * 100.0,
                metrics.avg_reward,
                active_synapses,
                meta.mode.as_str(),
                elapsed.as_secs_f64()
            );
        }
    }

    // --- Final summary ---
    println!("\n=== Training Complete ===");
    println!("Total episodes:    {}", MAX_EPISODES);
    println!("Best accuracy:     {:.1}%", tracker.best_accuracy * 100.0);
    println!("Best epoch:        {}", tracker.best_epoch);
    println!("Final replay size: {}", replay.len());
    println!("Curriculum prog:   {:.1}%", curriculum.progress() * 100.0);

    // Export metrics
    let _ = std::fs::create_dir_all("docs/src/development");
    let json = tracker.to_json();
    let _ = std::fs::write("docs/src/development/harness_training_metrics.json", json);
    println!("Metrics exported to docs/src/development/harness_training_metrics.json");
}

// ---------------------------------------------------------------------------
// Encoding / Decoding helpers
// ---------------------------------------------------------------------------

/// Encodes an ARC grid to a spike pattern for the input neurons.
///
/// Strategy: each cell (x, y) with color > 0 maps to input neuron
/// `(y * width + x) % INPUT_NEURONS`. Multiple cells may share a neuron.
fn encode_grid_to_spikes(grid: &ArcGrid, num_neurons: usize) -> Vec<usize> {
    let mut spikes = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if grid.data[y][x] > 0 {
                let neuron = (y * grid.width + x) % num_neurons;
                if !spikes.contains(&neuron) {
                    spikes.push(neuron);
                }
            }
        }
    }
    spikes
}

/// Decodes a spike pattern from output neurons back to an ARC grid.
///
/// Strategy: if output neuron i fired, set cell `(i % width, i / width)` to 1.
fn decode_spikes_to_grid(spikes: &[usize], width: usize, height: usize) -> ArcGrid {
    let mut data = vec![vec![0u8; width]; height];
    for &neuron in spikes {
        let x = neuron % width;
        let y = neuron / width;
        if y < height && x < width {
            data[y][x] = 1;
        }
    }
    ArcGrid::from_data(data).unwrap_or_else(|_| ArcGrid::new())
}

/// Checks if an ARC task is an identity task (all train pairs have input == output).
fn is_identity_task(task: &ArcTask) -> bool {
    task.train_pairs.iter().all(|(i, o)| i == o)
}

// ---------------------------------------------------------------------------
// Synthetic dataset fallback
// ---------------------------------------------------------------------------

fn create_synthetic_dataset() -> ArcDataset {
    let mut tasks = Vec::new();
    for i in 0..10 {
        let size = 3 + (i % 4);
        let mut task = ArcTask::new(format!("synthetic_{}", i));
        let grid = vec![vec![0u8; size]; size];
        task.train_pairs.push((ArcGrid::from_data(grid.clone()).unwrap(), ArcGrid::from_data(grid).unwrap()));
        tasks.push(task);
    }
    ArcDataset { tasks }
}
