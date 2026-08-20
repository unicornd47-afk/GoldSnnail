//! Fractal Harness Demo — 3-1-4-1 Equal Scaling
//!
//! Demonstrates:
//!   1. Build the 3141 frozen core (3 input regions, 1 core, 4 output, 1 gating)
//!   2. First NoteCoreLayer wrapping the frozen core
//!   3. Equal fractal scaling up and down
//!   4. Forward pass at each scale level

use goldsnnail::harness::*;
use std::time::Instant;

const INPUT_DIM: usize = 32;
const OUTPUT_DIM: usize = 32;

fn main() {
    println!("=== GoldSnnail 3-1-4-1 Fractal Scaling Demo ===\n");

    // --- Step 1: Build the 3141 frozen core ---
    println!("--- Step 1: 3141 Frozen Core ---");
    let core = build_3141_fractal(INPUT_DIM, OUTPUT_DIM);
    println!(
        "3141 Core: 1 layer, width={}, depth={}, recurrence={}, plasticity={}",
        core.layers[0].scale.width,
        core.layers[0].scale.depth,
        core.layers[0].scale.recurrence,
        core.layers[0].scale.plasticity
    );
    println!("Params: {}, Cost: {}\n", core.param_count(), core.compute_cost());

    // --- Step 2: Build first NoteCoreLayer ---
    println!("--- Step 2: First NoteCoreLayer ---");
    let layer = FractalLayer::new(
        0,
        ScaleProfile::base(),
        INPUT_DIM,
        32,
        OUTPUT_DIM,
    );
    println!(
        "Layer 0: input_adapter[{}x{}] -> FrozenCore[180 neurons] -> output_adapter[{}x{}]",
        layer.input_adapter.input_dim,
        layer.input_adapter.output_dim,
        layer.output_adapter.input_dim,
        layer.output_adapter.output_dim,
    );
    println!("Layer params: {}\n", layer.param_count());

    // --- Step 3: Equal scaling up ---
    println!("--- Step 3: Equal Scale Up ---");
    let mut current = core.clone();
    for level in 1..=3 {
        let mut scaled = scale_network(&current, level, ScaleDir::Up);
        println!(
            "Level {}: {} layers, width={}, depth={}, recurrence={}, params={}, cost={}",
            level,
            scaled.layers.len(),
            scaled.layers[0].scale.width,
            scaled.layers[0].scale.depth,
            scaled.layers[0].scale.recurrence,
            scaled.param_count(),
            scaled.compute_cost()
        );

        // Run forward pass
        let input = vec![0.1f32; INPUT_DIM];
        let start = Instant::now();
        let result = scaled.forward(input);
        println!("  -> Forward: {} spikes, {} ticks, {:.2}ms",
            result.total_spikes, result.total_ticks, start.elapsed().as_secs_f64() * 1000.0);

        current = scaled;
    }

    // --- Step 4: Equal scaling down ---
    println!("\n--- Step 4: Equal Scale Down ---");
    for level in (1..=2).rev() {
        let mut scaled = scale_network(&current, level, ScaleDir::Down);
        println!(
            "Level {}: {} layers, width={}, depth={}, recurrence={}, params={}, cost={}",
            level,
            scaled.layers.len(),
            scaled.layers[0].scale.width,
            scaled.layers[0].scale.depth,
            scaled.layers[0].scale.recurrence,
            scaled.param_count(),
            scaled.compute_cost()
        );

        let input = vec![0.1f32; INPUT_DIM];
        let result = scaled.forward(input);
        println!("  -> Forward: {} spikes, {} ticks", result.total_spikes, result.total_ticks);

        current = scaled;
    }

    // --- Step 5: Verify symmetry ---
    println!("\n--- Step 5: Verify Scale Symmetry ---");
    let original = FractalNetwork::new(INPUT_DIM, OUTPUT_DIM, 2, ScaleProfile::base());
    let up1 = scale_network(&original, 1, ScaleDir::Up);
    let down1 = scale_network(&up1, 1, ScaleDir::Down);
    println!(
        "Original: {} layers, width={}",
        original.layers.len(),
        original.layers[0].scale.width
    );
    println!(
        "Up then Down: {} layers, width={}",
        down1.layers.len(),
        down1.layers[0].scale.width
    );
    assert_eq!(original.layers.len(), down1.layers.len(), "Depth symmetry");
    assert_eq!(original.layers[0].scale.width, down1.layers[0].scale.width, "Width symmetry");
    println!("Symmetry check: PASSED");

    // --- Step 6: Frontier architecture summary ---
    println!("\n--- Step 6: Frontier Template Mapping ---");
    println!("Transformer   -> Attention(SpikePattern) + FrozenCore + Adapter FFN");
    println!("Mamba/RWKV    -> Temporal Recurrence + Gated State Update + Core");
    println!("FractalNet    -> Recursive FractalLayer with Residual Connections");
    println!("Universal Approx -> Frozen Basis (Core) + Linear Adapter = Function Space");
    println!("This Architecture: 3-1-4-1 NoteCoreLayer -> FractalNetwork -> Equal Scaling");

    println!("\n=== Demo Complete ===");
}


