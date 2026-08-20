//! Avalanche-Guided Response Selection
//!
//! Uses critical dynamics from the ConceptGraph to guide template selection.
//! Instead of purely random template choice, the avalanche simulation determines:
//! 1. Which semantic clusters are activated
//! 2. How many template slots should be filled
//! 3. Which specific tokens to select from each cluster
//!
//! This preserves grammatical structure (templates) while making content
//! selection emergent from the critical dynamics.

use crate::semantics::{SemanticTrainer, ConceptGraph};
use crate::telemetry::{power_law::PowerLawObserver, simulate_avalanche};
use crate::chat::{TokenSpikeEncoder, SpikeTokenDecoder};
use rand::Rng;

/// Result of an avalanche-guided selection process.
#[derive(Debug, Clone)]
pub struct AvalancheSelection {
    /// Activated cluster indices (0-6)
    pub active_clusters: Vec<usize>,
    /// Avalanche size (number of activated nodes)
    pub avalanche_size: usize,
    /// Selected words for each cluster
    pub selected_words: Vec<String>,
    /// Template pattern to use (e.g., "DET NOUN VERB")
    pub template_pattern: String,
    /// Clusters explicitly activated via bridge or text seed
    pub seed_clusters: Vec<usize>,
}

/// Selects response words using avalanche dynamics from the ConceptGraph.
///
/// This is the controlled experiment: templates provide the syntax,
/// avalanches provide the semantics.
pub struct AvalancheGuidedSelector {
    pub trainer: *mut SemanticTrainer,
    pub encoder: *mut TokenSpikeEncoder,
    pub decoder: *mut SpikeTokenDecoder,
    pub observer: *mut PowerLawObserver,
    pub max_avalanche_steps: usize,
}

impl AvalancheGuidedSelector {
    pub fn new(
        trainer: *mut SemanticTrainer,
        encoder: *mut TokenSpikeEncoder,
        decoder: *mut SpikeTokenDecoder,
        observer: *mut PowerLawObserver,
    ) -> Self {
        Self {
            trainer,
            encoder,
            decoder,
            observer,
            max_avalanche_steps: 5,
        }
    }

    /// Run an avalanche from the input word and select response content.
    pub fn select(&mut self, input: &str) -> AvalancheSelection {
        let trainer = unsafe { &mut *self.trainer };
        let encoder = unsafe { &mut *self.encoder };

        // Find the seed node for the input word
        let seed_id = trainer.lexicon.word_index.get(input)
            .copied()
            .unwrap_or(0);

        // Determine seed cluster from the word itself
        let seed_cluster = if let Some(seed_token) = trainer.lexicon.tokens.get(seed_id) {
            vec![seed_token.class as usize]
        } else {
            Vec::new()
        };

        self.select_from_seed(&trainer.concept_graph, seed_id, encoder, seed_cluster)
    }

    /// Runs the full avalanche pipeline from a given seed node in any ConceptGraph.
    ///
    /// This is the shared core used by both text-based and visual-based selection.
    pub fn select_from_seed(
        &mut self,
        graph: &ConceptGraph,
        seed_id: usize,
        encoder: &TokenSpikeEncoder,
        seed_clusters: Vec<usize>,
    ) -> AvalancheSelection {
        // Run avalanche simulation
        let mut rng = rand::thread_rng();
        let avalanche_size = simulate_avalanche(
            graph,
            seed_id,
            self.max_avalanche_steps,
            &mut rng,
        );

        // Determine template pattern based on avalanche size
        let template_pattern = if avalanche_size <= 3 {
            "DET NOUN VERB"
        } else if avalanche_size <= 10 {
            "DET NOUN ADJ VERB"
        } else {
            "DET NOUN ADJ VERB PREP LOC"
        };

        // Determine active clusters
        let active_clusters = self.determine_active_clusters_from_graph(graph, seed_id, avalanche_size);

        // Select words for each cluster
        let mut selected_words = self.select_words_for_clusters_from_graph(graph, encoder, &active_clusters);

        // Prepend seed words so they are prioritized in template slots
        // The seed node's label is the bridge-activated word (e.g., "drei", "vier", "neun")
        if let Some(seed_node) = graph.nodes.get(seed_id) {
            if encoder.neuron_for_word(&seed_node.label).is_some() {
                selected_words.insert(0, seed_node.label.clone());
            }
        }

        AvalancheSelection {
            active_clusters,
            avalanche_size,
            selected_words,
            template_pattern: template_pattern.to_string(),
            seed_clusters,
        }
    }

    /// Selects response content from visual DVS input via cross-modal bridge propagation.
    ///
    /// Pipeline:
    /// 1. Project DVS events to visual cluster index (caller must do MLP projection)
    /// 2. Propagate visual→language via BridgeEdge
    /// 3. Run avalanche from language cluster
    /// 4. Fill template slots with avalanche-selected words, prioritizing seed words
    pub fn select_from_visual_input(
        &mut self,
        graph: &ConceptGraph,
        visual_cluster_idx: usize,
    ) -> AvalancheSelection {
        let encoder = unsafe { &mut *self.encoder };

        // Cross-modal propagation: visual → language
        let lang_clusters = graph.propagate_visual_to_language(visual_cluster_idx);

        // Use the first language cluster as avalanche seed
        let seed_id = lang_clusters.first().copied().unwrap_or(0);

        // Track which clusters were activated by the bridge (for seed prioritization)
        let seed_clusters: Vec<usize> = if let Some(seed_node) = graph.nodes.get(seed_id) {
            vec![seed_node.id % 7]
        } else {
            Vec::new()
        };

        // Run avalanche from language seed
        self.select_from_seed(graph, seed_id, encoder, seed_clusters)
    }

    /// Graph-based cluster activation (no trainer dependency).
    fn determine_active_clusters_from_graph(
        &self,
        graph: &ConceptGraph,
        seed_id: usize,
        avalanche_size: usize,
    ) -> Vec<usize> {
        let mut clusters = Vec::new();
        let mut rng = rand::thread_rng();

        if let Some(seed_node) = graph.nodes.get(seed_id) {
            clusters.push(seed_node.id % 7);
        }

        let num_extra = (avalanche_size / 2).min(4);
        for _ in 0..num_extra {
            let random_class = rng.r#gen::<usize>() % 7;
            if !clusters.contains(&random_class) {
                clusters.push(random_class);
            }
        }

        clusters
    }

    /// Graph-based word selection (no trainer dependency).
    fn select_words_for_clusters_from_graph(
        &self,
        graph: &ConceptGraph,
        encoder: &TokenSpikeEncoder,
        clusters: &[usize],
    ) -> Vec<String> {
        let mut words = Vec::new();
        let mut rng = rand::thread_rng();

        for &cluster in clusters {
            let candidates: Vec<_> = graph.nodes.iter()
                .filter(|n| n.id % 7 == cluster)
                .collect();

            if candidates.is_empty() {
                continue;
            }

            let mut attempts = 0;
            while attempts < 10 {
                let candidate = candidates[rng.r#gen::<usize>() % candidates.len()];
                if encoder.neuron_for_word(&candidate.label).is_some() {
                    words.push(candidate.label.clone());
                    break;
                }
                attempts += 1;
            }
        }

        words
    }
}

/// Builds a response from an AvalancheSelection by filling template slots.
///
/// Seed clusters (e.g., digit words from bridge activation) are prioritized
/// in the template slots so that the cross-modal signal is preserved in the output.
pub fn build_response_from_selection(selection: &AvalancheSelection) -> Vec<String> {
    let mut response = Vec::new();
    let mut rng = rand::thread_rng();

    // Template slot fillers
    let dets = ["der", "die", "das"];
    let verbs = ["läuft", "springt", "ist", "sieht", "schläft", "fliegt", "scheint"];

    // Separate seed words from avalanche words
    // Seed words come from bridge activation or text input and should be prioritized
    let (seed_words, avalanche_words): (Vec<_>, Vec<_>) = selection.selected_words.iter()
        .enumerate()
        .partition(|(i, _)| {
            // Words from seed clusters appear first in selected_words
            // (select_from_visual_input ensures this ordering)
            *i < selection.seed_clusters.len()
        });

    let mut seed_iter = seed_words.into_iter().map(|(_, w)| w.clone());
    let mut avalanche_iter = avalanche_words.into_iter().map(|(_, w)| w.clone());

    match selection.template_pattern.as_str() {
        "DET NOUN VERB" => {
            response.push(dets[rng.r#gen::<usize>() % dets.len()].to_string());
            // Seed word first (NOUN slot), then avalanche fallback
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| "hund".to_string()));
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| verbs[rng.r#gen::<usize>() % verbs.len()].to_string()));
        }
        "DET NOUN ADJ VERB" => {
            response.push(dets[rng.r#gen::<usize>() % dets.len()].to_string());
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| "hund".to_string()));
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| "groß".to_string()));
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| verbs[rng.r#gen::<usize>() % verbs.len()].to_string()));
        }
        "DET NOUN ADJ VERB PREP LOC" => {
            response.push(dets[rng.r#gen::<usize>() % dets.len()].to_string());
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| "hund".to_string()));
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| "groß".to_string()));
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| verbs[rng.r#gen::<usize>() % verbs.len()].to_string()));
            response.push("in".to_string());
            response.push(seed_iter.next().or_else(|| avalanche_iter.next()).unwrap_or_else(|| "haus".to_string()));
        }
        _ => {}
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avalanche_guided_selector_creates_selection() {
        let mut trainer = SemanticTrainer::new(1.0);
        
        // Pre-populate concept graph
        for token in &trainer.lexicon.tokens.clone() {
            trainer.concept_graph.add_concept(&token.surface, token.hyperbolic.clone());
        }
        trainer.concept_graph.add_self_connections();
        trainer.concept_graph.add_random_edges(30);
        
        let mut encoder = TokenSpikeEncoder::new(1.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);
        encoder.register_lexicon(&trainer.lexicon);
        decoder.register_lexicon(&trainer.lexicon);
        
        let mut observer = PowerLawObserver::new(100);
        let mut selector = AvalancheGuidedSelector::new(
            &mut trainer, &mut encoder, &mut decoder, &mut observer,
        );
        
        let selection = selector.select("hund");
        assert!(!selection.selected_words.is_empty() || selection.avalanche_size > 0);
    }

    #[test]
    fn build_response_from_selection_produces_words() {
        let selection = AvalancheSelection {
            active_clusters: vec![4, 3],
            avalanche_size: 3,
            selected_words: vec!["hund".to_string(), "läuft".to_string()],
            template_pattern: "DET NOUN VERB".to_string(),
            seed_clusters: Vec::new(),
        };
        let response = build_response_from_selection(&selection);
        assert_eq!(response.len(), 3);
        let dets = ["der", "die", "das"];
        let verbs = ["läuft", "springt", "ist", "sieht", "schläft", "fliegt", "scheint"];
        assert!(dets.contains(&response[0].as_str()), "response[0] should be a determiner, got {}", response[0]);
        assert_eq!(response[1], "hund");
        assert!(verbs.contains(&response[2].as_str()), "response[2] should be a verb, got {}", response[2]);
    }
}
