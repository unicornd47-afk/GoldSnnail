//! Multi-Modal Generation — Visual DVS Input → Language Output
//!
//! End-to-end pipeline demonstrating the Bridge Edge architecture:
//! 1. Load N-MNIST DVS events for a digit
//! 2. Project to 8D hyperbolic space via trained MLP
//! 3. Find nearest visual cluster
//! 4. Propagate through BridgeEdge to language cluster
//! 5. Run avalanche simulation from language cluster
//! 6. Generate grammatical response via template filling
//!
//! This is the first SNN-based multi-modal generator: spike trains → words.

use goldsnnail::{
    ConceptGraph, NmnistDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_combined_features, AvalancheGuidedSelector,
    build_response_from_selection,
};
use rand::seq::SliceRandom;
use std::time::Instant;

const BINS: usize = 16;
const TAU_US: f32 = 50000.0;

fn main() {
    println!("=== GoldSnnail Multi-Modal Generation (Visual → Language) ===\n");

    // 1. Build multimodal ConceptGraph
    let graph = build_multimodal_graph();

    println!("ConceptGraph: {} nodes, {} edges, {} bridge edges",
        graph.nodes.len(), graph.edges.len(), graph.bridge_edges.len());

    for bridge in graph.bridges() {
        let v = &graph.nodes[bridge.visual_cluster].label;
        let l = &graph.nodes[bridge.language_cluster].label;
        println!("  Bridge: {} ↔ {} (w={:.2})", v, l, bridge.weight);
    }

    // 2. Train MLP projection layer
    println!("\nTraining MLP projection layer...");
    let dataset = NmnistDataset::load(200);
    let (train_set, test_set) = split_dataset(&dataset);

    let num_classes = dataset.available_digits.len();
    let class_centers = init_class_centers(num_classes, 8, 0.7);
    let mut mlp = ProjectionLayer::new(3 * BINS * BINS, 0.02, dataset.available_digits.clone(), 8);

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

    // 3. Build AvalancheGuidedSelector (text-only baseline)
    println!("\nBuilding AvalancheGuidedSelector...");
    let mut trainer = goldsnnail::SemanticTrainer::new(1.0);
    let mut encoder = goldsnnail::TokenSpikeEncoder::new(1.0, 5);
    let mut decoder = goldsnnail::SpikeTokenDecoder::new(1);
    encoder.register_lexicon(&trainer.lexicon);
    decoder.register_lexicon(&trainer.lexicon);

    // Register digit words so bridge seeds can be verbalized
    let digit_words = ["drei", "vier", "neun"];
    for (i, word) in digit_words.iter().enumerate() {
        let neuron_idx = 100 + i;
        encoder.register_word(word.to_string(), neuron_idx);
        decoder.register_word(word.to_string(), neuron_idx);
    }

    let mut observer = goldsnnail::PowerLawObserver::new(100);
    let mut selector = AvalancheGuidedSelector::new(
        &mut trainer, &mut encoder, &mut decoder, &mut observer,
    );

    // 4. End-to-end: Visual → Language pipeline
    println!("\n--- Visual → Language Generation ---");
    let mut generated = 0;
    let mut grammatical = 0;

    for sample in &test_set {
        let histogram = project_dvs_to_combined_features(&sample.events, BINS, TAU_US);
        let output = mlp.project(&histogram);
        let output_f64: Vec<f64> = output.iter().map(|&x| x as f64).collect();

        // Find nearest visual cluster
        let mut best_visual_idx = 0usize;
        let mut best_sim = f64::NEG_INFINITY;
        for (class_idx, center) in class_centers.iter().enumerate() {
            let sim = cosine_similarity(&output_f64, center);
            if sim > best_sim {
                best_sim = sim;
                best_visual_idx = class_idx;
            }
        }

        // Visual → Language via bridge
        let selection = selector.select_from_visual_input(&graph, best_visual_idx);
        let response = build_response_from_selection(&selection);

        generated += 1;
        if is_grammatical(&response) {
            grammatical += 1;
        }

        if generated <= 5 {
            let visual_label = &graph.nodes[best_visual_idx].label;
            println!("  Input digit={}: visual='{}' → response={:?}",
                sample.digit, visual_label, response);
        }
    }

    let gram_rate = grammatical as f64 / generated as f64;
    println!("\nGrammatical rate: {:.1}% ({}/{})", gram_rate * 100.0, grammatical, generated);

    // 5. Compare with text-only baseline
    println!("\n--- Text-Only Baseline ---");
    let text_inputs = ["drei", "vier", "neun"];
    for &word in &text_inputs {
        let selection = selector.select(word);
        let response = build_response_from_selection(&selection);
        println!("  Input '{}': response={:?}", word, response);
    }

    println!("\n=== Multi-Modal Generation Complete ===");
}

/// Builds a multimodal ConceptGraph with visual and language nodes + bridges.
fn build_multimodal_graph() -> ConceptGraph {
    let mut graph = ConceptGraph::new(1.0);

    let digits = [3u8, 4, 9];
    let visual_embeddings = vec![
        goldsnnail::HyperbolicPoint::new(ndarray::array![0.60, 0.10]).unwrap(),
        goldsnnail::HyperbolicPoint::new(ndarray::array![0.55, 0.50]).unwrap(),
        goldsnnail::HyperbolicPoint::new(ndarray::array![0.10, 0.60]).unwrap(),
    ];

    for (i, emb) in visual_embeddings.iter().enumerate() {
        graph.add_concept(&format!("digit_{}", digits[i]), emb.clone());
    }

    let lang_words = ["drei", "vier", "neun"];
    let lang_embeddings = vec![
        goldsnnail::HyperbolicPoint::new(ndarray::array![0.20, 0.03]).unwrap(),
        goldsnnail::HyperbolicPoint::new(ndarray::array![0.18, 0.16]).unwrap(),
        goldsnnail::HyperbolicPoint::new(ndarray::array![0.03, 0.20]).unwrap(),
    ];

    for (i, emb) in lang_embeddings.iter().enumerate() {
        graph.add_concept(lang_words[i], emb.clone());
    }

    // Intra-modal edges
    let _ = graph.add_edge("digit_3", "digit_4", goldsnnail::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("digit_4", "digit_9", goldsnnail::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("digit_9", "digit_3", goldsnnail::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("drei", "vier", goldsnnail::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("vier", "neun", goldsnnail::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("neun", "drei", goldsnnail::RelationType::RelatedTo, 0.3);

    // Bridge edges
    let visual_ids: Vec<usize> = digits.iter()
        .filter_map(|&d| graph.index.get(&format!("digit_{}", d)).copied())
        .collect();
    let lang_ids: Vec<usize> = lang_words.iter()
        .filter_map(|&w| graph.index.get(w).copied())
        .collect();

    for (&v, &l) in visual_ids.iter().zip(lang_ids.iter()) {
        graph.add_bridge(v, l);
    }

    graph
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

/// Simple grammaticality check for prototype: requires DET + at least one content word.
fn is_grammatical(response: &[String]) -> bool {
    let dets = ["der", "die", "das"];
    let content_words: Vec<&str> = vec![
        "hund", "katze", "vogel", "fisch", "tier", "tisch", "haus", "stein",
        "liebe", "freiheit", "idee", "wahrheit",
        "läuft", "springt", "ist", "sieht", "frisst", "denkt", "scheint", "bleibt",
        "groß", "klein", "schnell", "heiß", "kalt",
        "in", "auf", "unter", "mit",
        "AGENT", "PATIENT", "THEMA", "LOCATION",
        "drei", "vier", "neun",
    ];

    if response.is_empty() {
        return false;
    }

    let has_det = dets.contains(&response[0].as_str());
    let has_content = response.iter().any(|w| content_words.contains(&w.as_str()));

    has_det && has_content
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
