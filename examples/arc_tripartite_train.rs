//! ARC Tripartite Training — End-to-End ARC Pipeline Demo
//!
//! Demonstrates the new ARC tripartite encoding + NoteCoreLayer pipeline:
//! - ArcTripartiteEncoder: maps pixels to 3-region spike codes
//! - NoteCoreLayer: processes codes through frozen SNN core
//! - ArcStreamingLoop: end-to-end task streaming
//! - ArcGridDecoder: decodes semantic vectors back to grids
//!
//! Usage:
//!   cargo run --example arc_tripartite_train

use goldworm::harness::{
    ArcTripartiteEncoder, ArcPhase, ArcStreamingLoop, ArcGridDecoder, NoteCoreLayer, ScaleProfile,
};

fn main() {
    println!("=== ARC Tripartite Training Demo ===\n");

    // --- Define a simple ARC task ---
    // Demo 1: Input is a 3x3 grid, output is the same grid reversed
    let demo_input_1 = vec![
        vec![0u8, 1, 2],
        vec![3, 4, 5],
        vec![6, 7, 8],
    ];
    let demo_output_1 = vec![
        vec![8, 7, 6],
        vec![5, 4, 3],
        vec![2, 1, 0],
    ];

    // Demo 2: Another transformation
    let demo_input_2 = vec![
        vec![1u8, 1, 1],
        vec![2, 2, 2],
        vec![3, 3, 3],
    ];
    let demo_output_2 = vec![
        vec![3, 3, 3],
        vec![2, 2, 2],
        vec![1, 1, 1],
    ];

    // Test input: predict the output
    let test_input = vec![
        vec![0u8, 0, 1],
        vec![1, 1, 2],
        vec![2, 2, 2],
    ];

    println!("Demo 1 Input:");
    print_grid(&demo_input_1);
    println!("Demo 1 Output:");
    print_grid(&demo_output_1);

    println!("Demo 2 Input:");
    print_grid(&demo_input_2);
    println!("Demo 2 Output:");
    print_grid(&demo_output_2);

    println!("Test Input:");
    print_grid(&test_input);

    // --- Create streaming loop ---
    let width = 3;
    let height = 3;
    let scale_width = 1;
    let mut loop_ = ArcStreamingLoop::new(width, height, scale_width);

    println!("\n--- Phase 1: Run without training (random weights) ---");
    let result = loop_.stream(
        &[demo_input_1.clone(), demo_input_2.clone()],
        &[demo_output_1.clone(), demo_output_2.clone()],
        &test_input,
    );
    println!("Predicted Output (untrained):");
    print_grid(&result.predicted_grid);
    println!("Is valid: {}", result.is_valid);
    println!("Tick: {}", result.tick);

    // --- Train on demo pairs ---
    println!("\n--- Phase 2: Train on demo pairs ---");
    let mut total_reward = 0.0f32;
    for epoch in 0..5 {
        let reward1 = loop_.adapt_from_demo(&demo_input_1, &demo_output_1, 0.1);
        let reward2 = loop_.adapt_from_demo(&demo_input_2, &demo_output_2, 0.1);
        total_reward = (reward1 + reward2) / 2.0;
        println!("Epoch {}: avg reward = {:.4}", epoch, total_reward);
    }

    // --- Run prediction after training ---
    println!("\n--- Phase 3: Run after training ---");
    let result = loop_.stream(
        &[demo_input_1.clone(), demo_input_2.clone()],
        &[demo_output_1.clone(), demo_output_2.clone()],
        &test_input,
    );
    println!("Predicted Output (trained):");
    print_grid(&result.predicted_grid);
    println!("Is valid: {}", result.is_valid);
    println!("Tick: {}", result.tick);

    // --- Test encoder scaling ---
    println!("\n--- Phase 4: Test encoder scaling ---");
    let enc1 = ArcTripartiteEncoder::new(width, height, 1);
    let enc2 = ArcTripartiteEncoder::new(width, height, 2);
    let code1 = enc1.encode(5, 1, 1, ArcPhase::TestInput);
    let code2 = enc2.encode(5, 1, 1, ArcPhase::TestInput);
    println!("Scale=1 code length: {}", code1.len());
    println!("Scale=2 code length: {}", code2.len());
    println!("Scale=2 is exactly 2x Scale=1: {}", code2.len() == 2 * code1.len());

    // --- Test NoteCoreLayer ARC mode ---
    println!("\n--- Phase 5: Test NoteCoreLayer ARC mode ---");
    let scale = ScaleProfile::base();
    let mut layer = NoteCoreLayer::new_arc(0, scale, enc1.scaled_dim());
    let code = enc1.encode(5, 1, 1, ArcPhase::TestInput);
    let result = layer.forward_arc(&code);
    println!("Semantic vectors length: {}", result.semantic_vectors.len());
    println!("Stage means: {:?}", result.stage_means);
    println!("Tick: {}", result.tick);

    // --- Test decoder ---
    println!("\n--- Phase 6: Test ArcGridDecoder ---");
    let mut decoder = ArcGridDecoder::new(width, height);
    let semantics = vec![0.5, -0.3, 0.8, -0.1];
    let decoded = decoder.decode_to_grid(&semantics);
    println!("Decoded from semantics {:?}:", semantics);
    print_grid(&decoded);

    println!("\n=== Demo Complete ===");
}

fn print_grid(grid: &[Vec<u8>]) {
    for row in grid {
        for &cell in row {
            print!("{} ", cell);
        }
        println!();
    }
}
