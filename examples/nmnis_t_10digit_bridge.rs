//! 10-Digit Bridge Edges — Full Cross-Modal Pipeline
//!
//! Demonstrates the complete bridge-edge architecture for all 10 N-MNIST digits:
//! 1. Build multimodal ConceptGraph with visual and language nodes
//! 2. Add bridge edges for all 10 digit ↔ word pairs
// 3. Train 16D MLP projection layer on 10-digit data
//! 4. Test cross-modal propagation and avalanche generation
//!
//! This is the full multi-modal SNN prototype: spike trains → 10-digit recognition → German number words → grammatical responses.

use goldsnnail::{
    ConceptGraph, BridgeEdge, RelationType,
    HyperbolicPoint,
    NmnistDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_combined_features,
    simulate_avalanche, PowerLawObserver,
};
use ndarray::array;
use rand::seq::SliceRandom;
use std::time::Instant;

const BINS: usize = 16;
const TAU_US: f32 = 50000.0;

fn main() {
    println!("=== GoldSnnail 10-Digit Bridge Edges ===\n");

    // 1. Build multimodal ConceptGraph
    let mut graph = build_multimodal_graph();

    println!("ConceptGraph: {} nodes, {} edges, {} bridge edges",
        graph.nodes.len(), graph.edges.len(), graph.bridge_edges.len());

    // 2. Add bridge edges for all 10 digits
    add_all_digit_bridges(&mut graph);

    println!("Bridge edges:");
    for bridge in graph.bridges() {
        let v = &graph.nodes[bridge.visual_cluster].label;
        let l = &graph.nodes[bridge.language_cluster].label;
        println!("  {} ↔ {} (w={:.2})", v, l, bridge.weight);
    }

    // 3. Train 16D MLP on 10-digit data
    println!("\nTraining 16D MLP projection layer...");
    let dataset = NmnistDataset::load(300);
    let (train_set, test_set) = split_dataset(&dataset);

    let num_classes = dataset.available_digits.len();
    let class_centers = init_class_centers(num_classes, 16, 0.7);
    let mut mlp = ProjectionLayer::new(3 * BINS * BINS, 0.02, dataset.available_digits.clone(), 16);

    let start = Instant::now();
    for epoch in 0..100 {
        let lr = 0.002 + 0.018 * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / 100.0).cos());
        mlp.set_learning_rate(lr);
        for sample in &train_set {
            let histogram = project_dvs_to_combined_features(&sample.events, BINS, TAU_US);
            let target_index = dataset.available_digits.iter().position(|&d| d == sample.digit).unwrap_or(0);
            mlp.train_step(&histogram, sample.digit, target_index, num_classes, &class_centers);
        }
    }
    println!("  Training time: {:?}", start.elapsed());

    // 4. Evaluate
    let (accuracy, per_digit) = mlp.evaluate(&test_set, BINS, &class_centers);
    println!("\nTest accuracy: {:.1}%", accuracy * 100.0);
    for (digit, correct, total) in &per_digit {
        let acc = *correct as f32 / *total as f32;
        println!("  Digit {}: {}/{} ({:.1}%)", digit, correct, total, acc * 100.0);
    }

    // 5. Test cross-modal propagation
    println!("\n--- Cross-Modal Propagation ---");
    let mut bridge_hits = 0;
    let mut bridge_total = 0;

    for sample in &test_set {
        let histogram = project_dvs_to_combined_features(&sample.events, BINS, TAU_US);
        let output = mlp.project(&histogram);
        let output_f64: Vec<f64> = output.iter().map(|&x| x as f64).collect();

        let mut best_visual_idx = 0usize;
        let mut best_sim = f64::NEG_INFINITY;
        for (class_idx, center) in class_centers.iter().enumerate() {
            let sim = cosine_similarity(&output_f64, center);
            if sim > best_sim {
                best_sim = sim;
                best_visual_idx = class_idx;
            }
        }

        let lang_clusters = graph.propagate_visual_to_language(best_visual_idx);
        bridge_total += 1;
        if !lang_clusters.is_empty() && best_sim > 0.3 {
            bridge_hits += 1;
        }

        if bridge_total <= 5 {
            let visual_label = &graph.nodes[best_visual_idx].label;
            let lang_labels: Vec<String> = lang_clusters.iter()
                .filter_map(|&idx| graph.nodes.get(idx).map(|n| n.label.clone()))
                .collect();
            println!("  Sample digit={}: visual='{}' (sim={:.3}) → lang={:?}",
                sample.digit, visual_label, best_sim, lang_labels);
        }
    }

    let bridge_rate = bridge_hits as f64 / bridge_total as f64;
    println!("\nBridge activation rate: {:.1}% ({}/{})", bridge_rate * 100.0, bridge_hits, bridge_total);

    // 6. Test bidirectional avalanche
    println!("\n--- Bidirectional Avalanche ---");
    test_bidirectional_avalanche(&graph);

    // 7. Verify criticality
    println!("\n--- Criticality Check ---");
    verify_criticality(&graph);

    println!("\n=== 10-Digit Bridge Edges Complete ===");
}

/// Builds multimodal ConceptGraph with visual and language nodes.
fn build_multimodal_graph() -> ConceptGraph {
    let mut graph = ConceptGraph::new(1.0);

    // Visual shell: digits at larger radius
    for i in 0..10 {
        let angle = (i as f64 / 10.0) * 2.0 * std::f64::consts::PI;
        let x = 0.55 * angle.cos();
        let y = 0.55 * angle.sin();
        graph.add_concept(&format!("digit_{}", i), HyperbolicPoint::new(array![x, y]).unwrap());
    }

    // Language shell: German number words
    let words = ["null", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun"];
    for (i, &word) in words.iter().enumerate() {
        let angle = (i as f64 / 10.0) * 2.0 * std::f64::consts::PI;
        let x = 0.20 * angle.cos();
        let y = 0.20 * angle.sin();
        graph.add_concept(word, HyperbolicPoint::new(array![x, y]).unwrap());
    }

    // Intra-modal edges
    for i in 0..10 {
        let j = (i + 1) % 10;
        let _ = graph.add_edge(&format!("digit_{}", i), &format!("digit_{}", j), RelationType::RelatedTo, 0.2);
        let _ = graph.add_edge(words[i], words[j], RelationType::RelatedTo, 0.2);
    }

    graph
}

/// Adds bridge edges for all 10 digits.
fn add_all_digit_bridges(graph: &mut ConceptGraph) {
    for i in 0..10u8 {
        let visual_label = format!("digit_{}", i);
        let lang_label = format!("{}", ["null", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun"][i as usize]);
        
        let visual_id = graph.index.get(&visual_label).copied();
        let lang_id = graph.index.get(&lang_label).copied();
        
        if let (Some(v), Some(l)) = (visual_id, lang_id) {
            graph.add_bridge(v, l);
        }
    }
}

/// Train/test split
fn split_dataset(dataset: &NmnistDataset) -> (Vec<goldsnnail::NmnistSample>, Vec<goldsnnail::NmnistSample>) {
    if dataset.test.is_empty() {
        let mut train = dataset.train.clone();
        let mut rng = rand::thread_rng();
        train.shuffle(&mut rng);
        let split = (train.len() as f32 * 0.8) as usize;
        (train[..split].to_vec(), train[split..].to_vec())
    } else {
        (dataset.train.clone(), dataset.test.clone())
    }
}

/// Tests bidirectional avalanche propagation.
fn test_bidirectional_avalanche(graph: &ConceptGraph) {
    let mut rng = rand::thread_rng();

    if let Some(&visual_id) = graph.index.get("digit_3") {
        let size = simulate_avalanche(graph, visual_id, 3, &mut rng);
        println!("  Avalanche from 'digit_3' (visual): size={}", size);
    }

    if let Some(&lang_id) = graph.index.get("neun") {
        let size = simulate_avalanche(graph, lang_id, 3, &mut rng);
        println!("  Avalanche from 'neun' (language): size={}", size);
    }
}

/// Verifies criticality metrics.
fn verify_criticality(graph: &ConceptGraph) {
    let mut observer = PowerLawObserver::new(100);
    let mut rng = rand::thread_rng();

    let seeds: Vec<usize> = graph.nodes.iter().take(10).map(|n| n.id).collect();
    for &seed in &seeds {
        let size = simulate_avalanche(graph, seed, 5, &mut rng);
        observer.record(size);
    }

    if let Some(fit) = observer.fit() {
        println!("  Tau = {:.2} (target ≈ -1.5)", fit.tau);
        println!("  R² = {:.3} (target > 0.8)", fit.r_squared);
        println!("  Status: {}", fit.status());
    } else {
        println!("  Not enough data for power-law fit");
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
