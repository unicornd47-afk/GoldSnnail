//! Bridge Edges — Cross-Modal Connections Between Visual and Language Clusters
//!
//! Demonstrates supervised pairing of N-MNIST visual clusters (digits 3, 4, 9)
//! with language clusters ("drei", "vier", "neun") via fixed-weight BridgeEdges.
//!
//! This is the first step toward multi-modal generation: visual input → bridge →
//! language avalanche → response generation.

use goldsnnail::{
    ConceptGraph, RelationType,
    NmnistDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_combined_features,
    simulate_avalanche, PowerLawObserver, HyperbolicPoint,
};
use ndarray::array;
use rand::seq::SliceRandom;
use std::time::Instant;

const BINS: usize = 16;
const TAU_US: f32 = 50000.0;

fn main() {
    println!("=== GoldSnnail Bridge Edges — Visual ↔ Language ===\n");

    // 1. Build ConceptGraph with visual and language nodes
    let mut graph = build_multimodal_graph();

    println!("ConceptGraph: {} nodes, {} semantic edges, {} bridge edges",
        graph.nodes.len(),
        graph.edges.len(),
        graph.bridge_edges.len(),
    );

    // 2. Add supervised bridge edges (digit ↔ word)
    add_digit_bridges(&mut graph);

    println!("Bridge edges:");
    for bridge in graph.bridges() {
        let visual_label = &graph.nodes[bridge.visual_cluster].label;
        let lang_label = &graph.nodes[bridge.language_cluster].label;
        println!("  {} ↔ {} (weight={:.2}, bidirectional={})",
            visual_label, lang_label, bridge.weight, bridge.bidirectional);
    }

    // 3. Test cross-modal propagation
    println!("\n--- Cross-Modal Propagation Tests ---");

    // Visual → Language: digit 3 should activate "drei"
    let lang_from_3 = graph.propagate_visual_to_language(0); // node 0 = digit_3
    println!("Visual 'digit_3' → Language clusters: {:?}", lang_from_3);
    assert!(!lang_from_3.is_empty(), "Bridge should activate language cluster");

    // Language → Visual: "vier" (node 4) should activate digit_4 (node 1)
    let visual_from_vier = graph.propagate_language_to_visual(4); // node 4 = vier
    println!("Language 'vier' → Visual clusters: {:?}", visual_from_vier);
    assert!(!visual_from_vier.is_empty(), "Bridge should activate visual cluster");

    // 4. Load N-MNIST and test MLP projection → bridge activation
    println!("\n--- MLP Projection + Bridge Activation ---");
    let dataset = NmnistDataset::load(100);
    let (train_set, test_set) = split_dataset(&dataset);

    let num_classes = dataset.available_digits.len();
    let class_centers = init_class_centers(num_classes, 8, 0.7);

    let mut mlp = ProjectionLayer::new(3 * BINS * BINS, 0.02, dataset.available_digits.clone(), 8);

    // Quick training (50 epochs for demo speed)
    println!("Training MLP (50 epochs)...");
    let start = Instant::now();
    for epoch in 0..50 {
        let lr = 0.002 + 0.018 * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / 50.0).cos());
        mlp.set_learning_rate(lr);

        for sample in &train_set {
            let histogram = project_dvs_to_combined_features(&sample.events, BINS, TAU_US);
            let target_index = dataset.available_digits.iter().position(|&d| d == sample.digit).unwrap_or(0);
            mlp.train_step(&histogram, sample.digit, target_index, num_classes, &class_centers);
        }
    }
    println!("  Training time: {:?}", start.elapsed());

    // Test: project DVS → find visual cluster → activate bridge → language cluster
    let mut bridge_hits = 0;
    let mut bridge_total = 0;

    for sample in &test_set {
        let histogram = project_dvs_to_combined_features(&sample.events, BINS, TAU_US);
        let output = mlp.project(&histogram);
        let output_f64: Vec<f64> = output.iter().map(|&x| x as f64).collect();

        // Find nearest visual cluster by cosine similarity to class centers
        let mut best_visual_idx = 0usize;
        let mut best_sim = f64::NEG_INFINITY;
        for (class_idx, center) in class_centers.iter().enumerate() {
            let sim = cosine_similarity(&output_f64, center);
            if sim > best_sim {
                best_sim = sim;
                best_visual_idx = class_idx;
            }
        }

        // Propagate through bridge
        let lang_clusters = graph.propagate_visual_to_language(best_visual_idx);
        bridge_total += 1;

        if !lang_clusters.is_empty() {
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

    let bridge_accuracy = bridge_hits as f64 / bridge_total as f64;
    println!("\nBridge activation rate: {:.1}% ({}/{})",
        bridge_accuracy * 100.0, bridge_hits, bridge_total);

    // 5. Test bidirectional avalanche simulation
    println!("\n--- Bidirectional Avalanche Simulation ---");
    test_bidirectional_avalanche(&graph);

    // 6. Verify criticality is maintained
    println!("\n--- Criticality Check ---");
    verify_criticality(&graph);

    println!("\n=== Bridge Edges Complete ===");
}

/// Builds a multimodal ConceptGraph with visual and language nodes.
fn build_multimodal_graph() -> ConceptGraph {
    let mut graph = ConceptGraph::new(1.0);

    // Visual shell: digits at larger radius (outer shell)
    // These represent the MLP output clusters for digits 3, 4, 9
    let digits = [3u8, 4, 9];
    let visual_embeddings = vec![
        HyperbolicPoint::new(array![0.60, 0.10]).unwrap(),  // digit_3
        HyperbolicPoint::new(array![0.55, 0.50]).unwrap(),  // digit_4
        HyperbolicPoint::new(array![0.10, 0.60]).unwrap(),  // digit_9
    ];

    for (i, emb) in visual_embeddings.iter().enumerate() {
        graph.add_concept(&format!("digit_{}", digits[i]), emb.clone());
    }

    // Language shell: number words at smaller radius (inner shell)
    let lang_embeddings = vec![
        HyperbolicPoint::new(array![0.20, 0.03]).unwrap(),  // drei
        HyperbolicPoint::new(array![0.18, 0.16]).unwrap(),  // vier
        HyperbolicPoint::new(array![0.03, 0.20]).unwrap(),  // neun
    ];

    for (i, emb) in lang_embeddings.iter().enumerate() {
        let words = ["drei", "vier", "neun"];
        graph.add_concept(words[i], emb.clone());
    }

    // Add some intra-modal semantic edges (visual)
    let _ = graph.add_edge("digit_3", "digit_4", RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("digit_4", "digit_9", RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("digit_9", "digit_3", RelationType::RelatedTo, 0.3);

    // Add some intra-modal semantic edges (language)
    let _ = graph.add_edge("drei", "vier", RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("vier", "neun", RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("neun", "drei", RelationType::RelatedTo, 0.3);

    graph
}

/// Adds supervised bridge edges between digits and their German words.
fn add_digit_bridges(graph: &mut ConceptGraph) {
    // Find node IDs by label
    let visual_labels = ["digit_3", "digit_4", "digit_9"];
    let lang_labels = ["drei", "vier", "neun"];

    let visual_ids: Vec<usize> = visual_labels.iter()
        .filter_map(|label| {
            let id = graph.index.get(*label).copied();
            if id.is_none() { println!("  WARN: visual label '{}' not found in graph index", label); }
            id
        })
        .collect();

    let lang_ids: Vec<usize> = lang_labels.iter()
        .filter_map(|label| {
            let id = graph.index.get(*label).copied();
            if id.is_none() { println!("  WARN: lang label '{}' not found in graph index", label); }
            id
        })
        .collect();

    println!("  Visual IDs: {:?}", visual_ids);
    println!("  Language IDs: {:?}", lang_ids);

    for (v_id, l_id) in visual_ids.iter().zip(lang_ids.iter()) {
        graph.add_bridge(*v_id, *l_id);
    }
}

/// Creates train/test split from dataset.
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

/// Tests bidirectional avalanche propagation through bridges.
fn test_bidirectional_avalanche(graph: &ConceptGraph) {
    let mut rng = rand::thread_rng();

    // Forward: visual → language
    if let Some(&visual_id) = graph.index.get("digit_3") {
        let size = simulate_avalanche(graph, visual_id, 3, &mut rng);
        println!("  Avalanche from 'digit_3' (visual): size={}", size);
    }

    // Backward: language → visual
    if let Some(&lang_id) = graph.index.get("neun") {
        let size = simulate_avalanche(graph, lang_id, 3, &mut rng);
        println!("  Avalanche from 'neun' (language): size={}", size);
    }
}

/// Verifies criticality metrics (Tau, R²) are within expected bounds.
fn verify_criticality(graph: &ConceptGraph) {
    let mut observer = PowerLawObserver::new(100);

    // Record avalanches from multiple seeds
    let seeds: Vec<usize> = graph.nodes.iter().take(6).map(|n| n.id).collect();
    for &seed in &seeds {
        let mut rng = rand::thread_rng();
        let size = simulate_avalanche(graph, seed, 5, &mut rng);
        observer.record(size);
    }

    if let Some(fit) = observer.fit() {
        println!("  Tau = {:.2} (target ≈ -1.5)", fit.tau);
        println!("  R² = {:.3} (target > 0.8)", fit.r_squared);
        println!("  Status: {}", fit.status());
        println!("  Note: Small graphs (N<10) often show SUPER-CRITICAL due to high clustering");
    } else {
        println!("  Not enough data for power-law fit (need ≥4 samples)");
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
