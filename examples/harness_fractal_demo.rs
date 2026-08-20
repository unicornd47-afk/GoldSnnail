//! Fractal Harness Demo — 3-1-4-1 Self-Similar Architecture
//!
//! Demonstrates the fractal scaling pattern:
//!   1. Build a minimal 3141 frozen core
//!   2. Wrap it in a FractalLayer (3 components, 1 contract, 4 scales, 1 frozen core)
//!   3. Stack layers into a FractalNetwork with residual connections
//!   4. Scale the network up/down while preserving the 3-1-4-1 pattern
//!
//! Usage:
//!   cargo run --example harness_fractal_demo

use goldsnnail::{
    harness::*,
    vision::{ArcDataset, ArcGrid},
};
use std::time::Instant;

const INPUT_DIM: usize = 32;
const OUTPUT_DIM: usize = 32;

fn main() {
    println!("=== GoldSnnail Fractal Harness — 3-1-4-1 Architecture Demo ===\n");

    // --- Step 1: Build the 3141 frozen core ---
    println!("--- Step 1: 3141 Frozen Core ---");
    let core = build_3141_fractal(INPUT_DIM, OUTPUT_DIM);
    println!(
        "Built 3141 fractal: 1 layer, width={}, depth={}, recurrence={}, plasticity={}",
        core.base_scale.width,
        core.base_scale.depth,
        core.base_scale.recurrence,
        core.base_scale.plasticity
    );
    println!("Total params (adapters only): {}", core.param_count());
    println!("Compute cost proxy: {}", core.compute_cost());

    // --- Step 2: Forward pass through the minimal network ---
    println!("\n--- Step 2: Forward Pass (3141) ---");
    let mut net3141 = core;
    let input = vec![0.1f32; INPUT_DIM];
    let start = Instant::now();
    let result = net3141.forward(input);
    let elapsed = start.elapsed();
    println!("Output dim: {}", result.output.len());
    println!("Total spikes: {}", result.total_spikes);
    println!("Total ticks: {}", result.total_ticks);
    println!("Time: {:.2?}", elapsed);

    // --- Step 3: Scale up by factor 2 ---
    println!("\n--- Step 3: Scale x2 ---");
    let scaled2 = scale_3141(&net3141, 1);
    println!(
        "Scaled x2: {} layers, width={}, depth={}, recurrence={}",
        scaled2.layers.len(),
        scaled2.layers[0].scale.width,
        scaled2.base_scale.depth,
        scaled2.layers[0].scale.recurrence
    );
    println!("Total params: {}", scaled2.param_count());
    println!("Compute cost: {}", scaled2.compute_cost());

    // --- Step 4: Scale up by factor 4 ---
    println!("\n--- Step 4: Scale x4 ---");
    let scaled4 = scale_3141(&net3141, 2);
    println!(
        "Scaled x4: {} layers, width={}, depth={}, recurrence={}",
        scaled4.layers.len(),
        scaled4.layers[0].scale.width,
        scaled4.base_scale.depth,
        scaled4.layers[0].scale.recurrence
    );
    println!("Total params: {}", scaled4.param_count());
    println!("Compute cost: {}", scaled4.compute_cost());

    // --- Step 5: Demonstrate adapter learning on synthetic task ---
    println!("\n--- Step 5: Adapter Learning ---");
    let mut learnable = build_3141_fractal(INPUT_DIM, OUTPUT_DIM);
    let target = vec![0.9f32; OUTPUT_DIM];

    for epoch in 0..20 {
        let input = vec![0.1f32; INPUT_DIM];
        let result = learnable.forward(input.clone());
        let output = result.output;

        // Compute simple MSE gradient
        let mut grad = vec![0.0f32; OUTPUT_DIM];
        let mut loss = 0.0f32;
        for i in 0..OUTPUT_DIM {
            let diff = output[i] - target[i];
            loss += diff * diff;
            grad[i] = 2.0 * diff / OUTPUT_DIM as f32;
        }
        loss /= OUTPUT_DIM as f32;

        // Adapt all layers
        learnable.adapt(&input, &grad, 0.01);

        if epoch % 5 == 0 {
            println!("  Epoch {:>2}: loss={:.6} spikes={}", epoch, loss, result.total_spikes);
        }
    }

    // --- Step 6: Final evaluation ---
    println!("\n--- Step 6: Final Evaluation ---");
    let test_input = vec![0.1f32; INPUT_DIM];
    let final_result = learnable.forward(test_input);
    let mut final_loss = 0.0f32;
    for i in 0..OUTPUT_DIM {
        let diff = final_result.output[i] - target[i];
        final_loss += diff * diff;
    }
    final_loss /= OUTPUT_DIM as f32;
    println!("Final MSE: {:.6}", final_loss);
    println!("Adapter params: {}", learnable.param_count());
    println!("Frozen core synapses: {}", learnable.layers[0].core.active_synapses());

    // --- Step 7: Scale summary table ---
    println!("\n--- Step 7: Scale Comparison ---");
    println!("{:<6} {:<6} {:<6} {:<10} {:<12} {:<12}", "Factor", "Layers", "Width", "Recurrence", "Params", "Cost");
    for factor in 1..5 {
        let net = scale_3141(&net3141, factor);
        println!(
            "{:<6} {:<6} {:<6} {:<10} {:<12} {:<12}",
            format!("x{}", 1 << factor),
            net.layers.len(),
            net.layers[0].scale.width,
            net.layers[0].scale.recurrence,
            net.param_count(),
            net.compute_cost()
        );
    }

    println!("\n=== Demo Complete ===");
}

