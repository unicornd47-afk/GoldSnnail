//! NoteCoreLayer + Fractal Scaling Demo
//!
//! Demonstrates:
//!   1. 3141 frozen core topology
//!   2. First NoteCoreLayer with gated state update
//!   3. Equal fractal scaling up/down
//!   4. Forward pass metrics at each scale level

use goldsnnail::harness::*;
use std::time::Instant;

const INPUT_DIM: usize = 32;
const OUTPUT_DIM: usize = 32;

fn main() {
    println!("=== GoldSnnail 3-1-4-1 NoteCoreLayer + Equal Scaling ===\n");

    // --- Step 1: 3141 frozen core ---
    println!("--- Step 1: 3141 Frozen Core ---");
    let core = build_3141_fractal(INPUT_DIM, OUTPUT_DIM);
    println!(
        "3141 Core: {} layer, width={}, depth={}, recurrence={}",
        core.layers.len(),
        core.layers[0].scale.width,
        core.layers[0].scale.depth,
        core.layers[0].scale.recurrence,
    );
    println!("Params: {}, Cost: {}\n", core.param_count(), core.compute_cost());

    // --- Step 2: First NoteCoreLayer ---
    println!("--- Step 2: First NoteCoreLayer ---");
    let layer = NoteCoreLayer::new(
        0,
        ScaleProfile::base(),
        INPUT_DIM,
        32,
        OUTPUT_DIM,
    );
    println!(
        "Layer 0: input_adapter[{}x{}] -> FrozenCore[180 neurons] -> output_adapter[6x{}] + gate_adapter[{}x1]",
        layer.input_adapter.input_dim,
        layer.input_adapter.output_dim,
        layer.output_adapter.output_dim,
        layer.gate_adapter.input_dim,
    );
    println!("Layer params: {}\n", layer.param_count());

    // --- Step 3: Equal scale up ---
    println!("--- Step 3: Equal Scale Up ---");
    let mut current = core.clone();
    for level in 1..=3 {
        let scaled = scale_network(&current, level, ScaleDir::Up);
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
        let start = Instant::now();
        let mut net = scaled;
        let result = net.forward(input);
        println!(
            "  -> Forward: {} spikes, {} ticks, {:.2}ms",
            result.total_spikes,
            result.total_ticks,
            start.elapsed().as_secs_f64() * 1000.0
        );

        current = net;
    }

    // --- Step 4: Equal scale down ---
    println!("\n--- Step 4: Equal Scale Down ---");
    for level in (1..=2).rev() {
        let scaled = scale_network(&current, level, ScaleDir::Down);
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
        let mut net = scaled;
        let result = net.forward(input);
        println!(
            "  -> Forward: {} spikes, {} ticks",
            result.total_spikes, result.total_ticks
        );

        current = net;
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
