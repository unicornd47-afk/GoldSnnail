//! Cross-Modal Benchmark — Visual→Text Generation Evaluation
//!
//! Defines a reproducible test suite for measuring multi-modal generation quality:
//! - Digit Accuracy: Was die Ziffer korrekt erkannt?
//! - Semantic Relevance: Enthält die Antwort das korrekte Ziffernwort?
//! - Grammatical Rate: Ist die Antwort grammatikalisch?
//! - Bridge Fidelity: Kam die Information über die Bridge an?
//!
//! Run: cargo run --example cross_modal_benchmark --release

use goldworm::{
    ConceptGraph, NmnistDataset, ProjectionLayer, init_class_centers,
    project_dvs_to_combined_features, AvalancheGuidedSelector,
    build_response_from_selection,
};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

const BINS: usize = 16;
const TAU_US: f32 = 50000.0;

/// Gold-standard digit-word mappings
const DIGIT_WORDS: &[(&str, &str)] = &[
    ("3", "drei"),
    ("4", "vier"),
    ("9", "neun"),
];

/// Benchmark result aggregation
#[derive(Debug, Default, Clone)]
pub struct BenchmarkResult {
    pub total_samples: usize,
    pub digit_correct: usize,
    pub semantic_relevant: usize,
    pub grammatical: usize,
    pub bridge_fired: usize,
    pub responses: Vec<String>,
}

impl BenchmarkResult {
    pub fn digit_accuracy(&self) -> f64 {
        self.digit_correct as f64 / self.total_samples as f64
    }

    pub fn semantic_relevance(&self) -> f64 {
        self.semantic_relevant as f64 / self.total_samples as f64
    }

    pub fn grammatical_rate(&self) -> f64 {
        self.grammatical as f64 / self.total_samples as f64
    }

    pub fn bridge_fidelity(&self) -> f64 {
        self.bridge_fired as f64 / self.total_samples as f64
    }

    pub fn print_summary(&self, label: &str) {
        println!("\n=== {} ===", label);
        println!("  Samples: {}", self.total_samples);
        println!("  Digit Accuracy:    {:.1}% ({}/{})",
            self.digit_accuracy() * 100.0, self.digit_correct, self.total_samples);
        println!("  Semantic Relevance: {:.1}% ({}/{})",
            self.semantic_relevance() * 100.0, self.semantic_relevant, self.total_samples);
        println!("  Grammatical Rate:  {:.1}% ({}/{})",
            self.grammatical_rate() * 100.0, self.grammatical, self.total_samples);
        println!("  Bridge Fidelity:   {:.1}% ({}/{})",
            self.bridge_fidelity() * 100.0, self.bridge_fired, self.total_samples);
    }
}

fn main() {
    println!("=== GoldWorm Cross-Modal Benchmark ===\n");

    // 1. Setup multimodal graph
    let graph = build_multimodal_graph();

    // 2. Train MLP
    println!("Training MLP projection layer...");
    let mut dataset = NmnistDataset::load(200);

    // Filter to only digits represented in the multimodal graph (3, 4, 9)
    let target_digits = [3u8, 4, 9];
    dataset.train.retain(|s| target_digits.contains(&s.digit));
    dataset.test.retain(|s| target_digits.contains(&s.digit));
    dataset.available_digits.retain(|d| target_digits.contains(d));

    let (train_set, test_set) = split_dataset(&dataset);

    let num_classes = dataset.available_digits.len();
    let class_centers = init_class_centers(num_classes, 8, 0.7);
    let mut mlp = ProjectionLayer::new(3 * BINS * BINS, 0.02, dataset.available_digits.clone(), 8);

    for epoch in 0..300 {
        let lr = 0.002 + 0.018 * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / 100.0).cos());
        mlp.set_learning_rate(lr);
        for sample in &train_set {
            let histogram = project_dvs_to_combined_features(&sample.events, BINS, TAU_US);
            let target_index = dataset.available_digits.iter().position(|&d| d == sample.digit).unwrap_or(0);
            mlp.train_step(&histogram, sample.digit, target_index, num_classes, &class_centers);
        }
    }

    // 3. Build selector
    let mut trainer = goldworm::SemanticTrainer::new(1.0);
    let mut encoder = goldworm::TokenSpikeEncoder::new(1.0, 5);
    let mut decoder = goldworm::SpikeTokenDecoder::new(1);
    encoder.register_lexicon(&trainer.lexicon);
    decoder.register_lexicon(&trainer.lexicon);

    // Register digit words with the encoder so bridge seeds can be verbalized
    let digit_words = ["drei", "vier", "neun"];
    for (i, word) in digit_words.iter().enumerate() {
        let neuron_idx = 100 + i;
        encoder.register_word(word.to_string(), neuron_idx);
        decoder.register_word(word.to_string(), neuron_idx);
    }

    let mut observer = goldworm::PowerLawObserver::new(100);
    let mut selector = AvalancheGuidedSelector::new(
        &mut trainer, &mut encoder, &mut decoder, &mut observer,
    );

    // 4. Run benchmarks
    let multimodal = benchmark_multimodal(&test_set, &mlp, &graph, &mut selector, &class_centers, &dataset.available_digits);
    let text_only = benchmark_text_only(&test_set, &graph, &mut selector);
    let random_baseline = benchmark_random(&test_set);

    // 5. Compare
    multimodal.print_summary("Multi-Modal (Visual→Language)");
    text_only.print_summary("Text-Only Baseline");
    random_baseline.print_summary("Random Baseline");

    // 6. Relative improvements
    println!("\n=== Relative Improvements ===");
    let mm_sem = multimodal.semantic_relevance();
    let to_sem = text_only.semantic_relevance();
    let rnd_sem = random_baseline.semantic_relevance();

    if to_sem > 0.0 {
        println!("  Semantic vs Text-Only:  {:+.1}%", ((mm_sem - to_sem) / to_sem) * 100.0);
    }
    if rnd_sem > 0.0 {
        println!("  Semantic vs Random:     {:+.1}%", ((mm_sem - rnd_sem) / rnd_sem) * 100.0);
    }

    let mm_gram = multimodal.grammatical_rate();
    let to_gram = text_only.grammatical_rate();
    let rnd_gram = random_baseline.grammatical_rate();

    if to_gram > 0.0 {
        println!("  Grammatical vs Text-Only: {:+.1}%", ((mm_gram - to_gram) / to_gram) * 100.0);
    }
    if rnd_gram > 0.0 {
        println!("  Grammatical vs Random:    {:+.1}%", ((mm_gram - rnd_gram) / rnd_gram) * 100.0);
    }

    // 7. Export results
    let json = format!(
r#"{{
  "multimodal": {{
    "digit_accuracy": {:.4},
    "semantic_relevance": {:.4},
    "grammatical_rate": {:.4},
    "bridge_fidelity": {:.4}
  }},
  "text_only": {{
    "digit_accuracy": {:.4},
    "semantic_relevance": {:.4},
    "grammatical_rate": {:.4}
  }},
  "random": {{
    "digit_accuracy": {:.4},
    "semantic_relevance": {:.4},
    "grammatical_rate": {:.4}
  }},
  "dataset": "N-MNIST 3-digit",
  "encoding": "8d-radial",
  "samples": {}
}}"#,
        multimodal.digit_accuracy(),
        multimodal.semantic_relevance(),
        multimodal.grammatical_rate(),
        multimodal.bridge_fidelity(),
        text_only.digit_accuracy(),
        text_only.semantic_relevance(),
        text_only.grammatical_rate(),
        random_baseline.digit_accuracy(),
        random_baseline.semantic_relevance(),
        random_baseline.grammatical_rate(),
        test_set.len()
    );

    std::fs::write("docs/src/development/cross_modal_benchmark.json", json).unwrap();
    println!("\nResults exported to docs/src/development/cross_modal_benchmark.json");
    println!("\n=== Benchmark Complete ===");
}

/// Benchmark: Visual input → MLP → Bridge → Language → Response
fn benchmark_multimodal(
    test_set: &[goldworm::NmnistSample],
    mlp: &ProjectionLayer,
    graph: &ConceptGraph,
    selector: &mut AvalancheGuidedSelector,
    class_centers: &[Vec<f64>],
    available_digits: &[u8],
) -> BenchmarkResult {
    let mut result = BenchmarkResult::default();
    result.total_samples = test_set.len();

    let digit_word_map: HashMap<&str, &str> = DIGIT_WORDS.iter().cloned().collect();

    for sample in test_set {
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

        // Check digit accuracy
        let predicted_digit = available_digits[best_visual_idx];
        if predicted_digit == sample.digit {
            result.digit_correct += 1;
        }

        // Cross-modal generation
        let lang_clusters = graph.propagate_visual_to_language(best_visual_idx);
        if !lang_clusters.is_empty() {
            result.bridge_fired += 1;
        }

        let selection = selector.select_from_visual_input(graph, best_visual_idx);
        let response = build_response_from_selection(&selection);
        let response_str = response.join(" ");

        // Check semantic relevance
        let expected_word = digit_word_map.get(sample.digit.to_string().as_str()).copied();
        if let Some(expected) = expected_word {
            if response_str.contains(expected) {
                result.semantic_relevant += 1;
            }
        }

        // Check grammaticality
        if is_grammatical(&response) {
            result.grammatical += 1;
        }

        result.responses.push(response_str);
    }

    result
}

/// Benchmark: Text input → Language → Response (no visual component)
fn benchmark_text_only(
    test_set: &[goldworm::NmnistSample],
    graph: &ConceptGraph,
    selector: &mut AvalancheGuidedSelector,
) -> BenchmarkResult {
    let mut result = BenchmarkResult::default();
    result.total_samples = test_set.len();

    let digit_word_map: HashMap<&str, &str> = DIGIT_WORDS.iter().cloned().collect();

    for sample in test_set {
        let digit_str = sample.digit.to_string();
        let expected_word = digit_word_map.get(digit_str.as_str()).copied();

        // Text-only: use the digit word as input
        let input_word = expected_word.unwrap_or("???");
        let selection = selector.select(input_word);
        let response = build_response_from_selection(&selection);
        let response_str = response.join(" ");

        // For text-only, digit accuracy = whether we used the right input word
        if expected_word.is_some() {
            result.digit_correct += 1;
        }

        // Semantic relevance: response contains the digit word
        if let Some(expected) = expected_word {
            if response_str.contains(expected) {
                result.semantic_relevant += 1;
            }
        }

        // Grammaticality
        if is_grammatical(&response) {
            result.grammatical += 1;
        }

        result.responses.push(response_str);
    }

    result
}

/// Benchmark: Random word selection (lower bound)
fn benchmark_random(test_set: &[goldworm::NmnistSample]) -> BenchmarkResult {
    use rand::thread_rng;
    let mut rng = thread_rng();

    let mut result = BenchmarkResult::default();
    result.total_samples = test_set.len();

    let dets = ["der", "die", "das"];
    let all_words = vec![
        "hund", "katze", "vogel", "fisch", "tier",
        "tisch", "haus", "stein",
        "liebe", "freiheit", "idee", "wahrheit",
        "läuft", "springt", "ist", "sieht", "frisst", "denkt", "scheint", "bleibt",
        "groß", "klein", "schnell", "heiß", "kalt",
        "drei", "vier", "neun",
    ];

    let digit_word_map: HashMap<&str, &str> = DIGIT_WORDS.iter().cloned().collect();

    for sample in test_set {
        // Random response: 2-4 words
        let len = 2 + (rng.r#gen::<usize>() % 3);
        let mut response = Vec::with_capacity(len);
        response.push(dets[rng.r#gen::<usize>() % dets.len()].to_string());
        for _ in 1..len {
            response.push(all_words[rng.r#gen::<usize>() % all_words.len()].to_string());
        }

        let response_str = response.join(" ");

        // Random digit accuracy ~33% for 3 classes
        if rng.r#gen::<f64>() < 0.33 {
            result.digit_correct += 1;
        }

        // Semantic relevance: random chance ~1/len(all_words)
        let expected_word = digit_word_map.get(sample.digit.to_string().as_str()).copied();
        if let Some(expected) = expected_word {
            if response_str.contains(expected) {
                result.semantic_relevant += 1;
            }
        }

        if is_grammatical(&response) {
            result.grammatical += 1;
        }

        result.responses.push(response_str);
    }

    result
}

/// Builds multimodal ConceptGraph
fn build_multimodal_graph() -> ConceptGraph {
    let mut graph = ConceptGraph::new(1.0);

    let digits = [3u8, 4, 9];
    let visual_embeddings = vec![
        goldworm::HyperbolicPoint::new(ndarray::array![0.60, 0.10]).unwrap(),
        goldworm::HyperbolicPoint::new(ndarray::array![0.55, 0.50]).unwrap(),
        goldworm::HyperbolicPoint::new(ndarray::array![0.10, 0.60]).unwrap(),
    ];

    for (i, emb) in visual_embeddings.iter().enumerate() {
        graph.add_concept(&format!("digit_{}", digits[i]), emb.clone());
    }

    let lang_words = ["drei", "vier", "neun"];
    let lang_embeddings = vec![
        goldworm::HyperbolicPoint::new(ndarray::array![0.20, 0.03]).unwrap(),
        goldworm::HyperbolicPoint::new(ndarray::array![0.18, 0.16]).unwrap(),
        goldworm::HyperbolicPoint::new(ndarray::array![0.03, 0.20]).unwrap(),
    ];

    for (i, emb) in lang_embeddings.iter().enumerate() {
        graph.add_concept(lang_words[i], emb.clone());
    }

    let _ = graph.add_edge("digit_3", "digit_4", goldworm::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("digit_4", "digit_9", goldworm::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("digit_9", "digit_3", goldworm::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("drei", "vier", goldworm::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("vier", "neun", goldworm::RelationType::RelatedTo, 0.3);
    let _ = graph.add_edge("neun", "drei", goldworm::RelationType::RelatedTo, 0.3);

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

/// Train/test split
fn split_dataset(dataset: &NmnistDataset) -> (Vec<goldworm::NmnistSample>, Vec<goldworm::NmnistSample>) {
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

/// Grammaticality check: DET + at least one content word
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

/// Cosine similarity
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}
