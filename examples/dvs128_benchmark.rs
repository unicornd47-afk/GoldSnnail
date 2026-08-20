//! DVS128 Benchmark — Event-based vision to semantic concepts
//!
//! Measures the DVS128 pipeline performance:
//! 1. Event encoding throughput
//! 2. Semantic projection accuracy (histogram -> hyperbolic -> nearest neighbor)
//! 3. Integration with AvalancheGuidedSelector

use goldworm::{
    SemanticTrainer, TokenSpikeEncoder, SpikeTokenDecoder,
    build_extended_lexicon, DvsEvent, DvsEncoder, DvsEncoderConfig,
    project_dvs_to_histogram, histogram_to_hyperbolic,
};
use std::time::Instant;

fn main() {
    println!("=== GoldWorm DVS128 Benchmark ===\n");
    
    let mut trainer = SemanticTrainer::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
    
    // Add recurrent connections for criticality
    trainer.concept_graph.add_self_connections();
    trainer.concept_graph.add_recurrent_connections(0.01);
    
    let config = DvsEncoderConfig {
        window_size_us: 1000,
        spikes_per_event: 1,
        max_delay_ticks: 10,
        use_polarity: true,
    };
    let mut dvs_encoder = DvsEncoder::with_config(config);
    
    println!("Lexicon: {} words", trainer.lexicon.tokens.len());
    println!("DVS128 encoder: {} potential neurons", 2 * 128 * 128);
    
    // Generate synthetic DVS events for known concepts
    let test_cases = vec![
        ("hund", vec![(10, 20), (11, 21), (12, 22)]),   // Moving dot pattern
        ("katze", vec![(50, 50), (51, 51), (52, 52)]),  // Center pattern
        ("vogel", vec![(100, 10), (101, 11), (102, 12)]), // Top-right pattern
    ];
    
    let mut total_events = 0usize;
    let mut total_spikes = 0usize;
    let mut correct_matches = 0usize;
    
    for (concept, pixel_pattern) in &test_cases {
        println!("\nTesting concept: {}", concept);
        
        // Generate synthetic events for this concept
        let events: Vec<DvsEvent> = pixel_pattern.iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                DvsEvent::new(
                    x as u8,
                    y as u8,
                    (i % 2) as u8,
                    (i as u32) * 1000,
                )
            })
            .collect();
        
        total_events += events.len();
        
        // Encode to spikes
        let start = Instant::now();
        let spikes = dvs_encoder.feed_batch(&events);
        let encode_time = start.elapsed();
        total_spikes += spikes.len();
        
        println!("  Events: {}", events.len());
        println!("  Spikes: {}", spikes.len());
        println!("  Encode time: {:?}", encode_time);
        
        // Project to histogram and then to hyperbolic space
        let histogram = project_dvs_to_histogram(&events, 8);
        let hp = histogram_to_hyperbolic(&histogram);
        
        // Find nearest concept in lexicon
        if let Ok(neighbors) = trainer.concept_graph.nearest_neighbors(&hp, 1) {
            if let Some((node_id, _)) = neighbors.first() {
                if let Some(node) = trainer.concept_graph.nodes.get(*node_id) {
                    println!("  Nearest concept: {}", node.label);
                    if node.label == *concept {
                        correct_matches += 1;
                    }
                }
            }
        }
    }
    
    println!("\n=== Summary ===");
    println!("Total events processed: {}", total_events);
    println!("Total spikes generated: {}", total_spikes);
    println!("Correct concept matches: {}/{}", correct_matches, test_cases.len());
    println!("Accuracy: {:.1}%", (correct_matches as f64 / test_cases.len() as f64) * 100.0);
}
