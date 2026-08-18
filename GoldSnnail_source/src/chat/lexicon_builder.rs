//! Lexicon Builder — Extended multilingual chat vocabulary with systematic geometry
//!
//! Provides a deterministic, shared lexicon for chat and benchmark examples.
//! 200+ German and English words distributed across 7 semantic clusters
//! using golden-angle spiral placement in 2D hyperbolic space.

use crate::semantics::{ConceptGraph, RelationType, SemanticTrainer, TokenClass};
use crate::chat::spike_token_bridge::{TokenSpikeEncoder, SpikeTokenDecoder};
use crate::chat::config::WorldGeometry;
use crate::geometry::{HyperbolicPoint, Quaternion};
use crate::LexiconToken;
use ndarray::array;

// =============================================================================
// 1. CLUSTER CENTERS — 7 semantic regions in 2D hyperbolic space
// =============================================================================

pub fn cluster_centers() -> Vec<[f64; 2]> {
    vec![
        [0.35, 0.35],
        [-0.35, 0.35],
        [0.35, -0.35],
        [-0.35, -0.35],
        [0.0, 0.45],
        [0.0, -0.45],
        [0.0, 0.0],
    ]
}

// =============================================================================
// 2. SPIRAL OFFSET — Golden-angle spiral for intra-cluster distribution
// =============================================================================

pub fn spiral_offset(index: usize, total: usize) -> [f64; 2] {
    let golden_angle = std::f64::consts::PI * (3.0 - (5.0f64).sqrt());
    let angle = index as f64 * golden_angle;
    let scale = 0.08;
    let radius = scale * ((index + 1) as f64 / total.max(1) as f64).sqrt();
    [radius * angle.cos(), radius * angle.sin()]
}

// =============================================================================
// 3. COORDINATE COMPUTATION — Cluster center + spiral offset
// =============================================================================

pub fn compute_coordinates(cluster_idx: usize, word_idx: usize, cluster_size: usize) -> [f64; 2] {
    let centers = cluster_centers();
    let center = centers[cluster_idx];
    let offset = spiral_offset(word_idx, cluster_size);
    [center[0] + offset[0], center[1] + offset[1]]
}

// =============================================================================
// 4. WORD LIST — 200+ multilingual tokens across 7 semantic categories
// =============================================================================

pub fn generate_word_list() -> Vec<(&'static str, TokenClass, usize)> {
    let mut words = Vec::new();

    // Cluster 0: Greetings (~20)
    let greetings: &[(&str, TokenClass)] = &[
        ("hallo", TokenClass::Determiner),
        ("hi", TokenClass::Determiner),
        ("guten_tag", TokenClass::Determiner),
        ("moin", TokenClass::Determiner),
        ("servus", TokenClass::Determiner),
        ("hello", TokenClass::Determiner),
        ("hey", TokenClass::Determiner),
        ("good_morning", TokenClass::Determiner),
        ("good_evening", TokenClass::Determiner),
        ("greetings", TokenClass::Determiner),
        ("hallo_leute", TokenClass::Determiner),
        ("hi_da", TokenClass::Determiner),
        ("guten_tag_leute", TokenClass::Determiner),
        ("moin_moin", TokenClass::Determiner),
        ("servus_leute", TokenClass::Determiner),
        ("hey_there", TokenClass::Determiner),
        ("good_morning_all", TokenClass::Determiner),
        ("good_evening_all", TokenClass::Determiner),
        ("greetings_all", TokenClass::Determiner),
        ("hallo_world", TokenClass::Determiner),
    ];
    for (w, c) in greetings {
        words.push((*w, *c, 0));
    }

    // Cluster 1: Pronouns (~20)
    let pronouns: &[(&str, TokenClass)] = &[
        ("ich", TokenClass::Determiner),
        ("du", TokenClass::Determiner),
        ("er", TokenClass::Determiner),
        ("sie", TokenClass::Determiner),
        ("es", TokenClass::Determiner),
        ("wir", TokenClass::Determiner),
        ("ihr", TokenClass::Determiner),
        ("I", TokenClass::Determiner),
        ("you", TokenClass::Determiner),
        ("he", TokenClass::Determiner),
        ("she", TokenClass::Determiner),
        ("it", TokenClass::Determiner),
        ("we", TokenClass::Determiner),
        ("they", TokenClass::Determiner),
        ("me", TokenClass::Determiner),
        ("him", TokenClass::Determiner),
        ("her", TokenClass::Determiner),
        ("us", TokenClass::Determiner),
        ("them", TokenClass::Determiner),
        ("mich", TokenClass::Determiner),
    ];
    for (w, c) in pronouns {
        words.push((*w, *c, 1));
    }

    // Cluster 2: State Verbs (~20)
    let state_verbs: &[(&str, TokenClass)] = &[
        ("ist", TokenClass::VerbState),
        ("bin", TokenClass::VerbState),
        ("bist", TokenClass::VerbState),
        ("sind", TokenClass::VerbState),
        ("seid", TokenClass::VerbState),
        ("war", TokenClass::VerbState),
        ("waren", TokenClass::VerbState),
        ("is", TokenClass::VerbState),
        ("am", TokenClass::VerbState),
        ("are", TokenClass::VerbState),
        ("were", TokenClass::VerbState),
        ("be", TokenClass::VerbState),
        ("been", TokenClass::VerbState),
        ("being", TokenClass::VerbState),
        ("have", TokenClass::VerbState),
        ("has", TokenClass::VerbState),
        ("had", TokenClass::VerbState),
        ("do", TokenClass::VerbState),
        ("does", TokenClass::VerbState),
    ];
    for (w, c) in state_verbs {
        words.push((*w, *c, 2));
    }

    // Cluster 3: Action Verbs (German + English)
    let action_verbs: &[(&str, TokenClass)] = &[
        ("läuft", TokenClass::VerbAction),
        ("springt", TokenClass::VerbAction),
        ("schläft", TokenClass::VerbAction),
        ("fliegt", TokenClass::VerbAction),
        ("scheint", TokenClass::VerbAction),
        ("wächst", TokenClass::VerbAction),
        ("geht", TokenClass::VerbAction),
        ("kommt", TokenClass::VerbAction),
        ("sieht", TokenClass::VerbAction),
        ("sagt", TokenClass::VerbAction),
        ("run", TokenClass::VerbAction),
        ("jump", TokenClass::VerbAction),
        ("sleep", TokenClass::VerbAction),
        ("fly", TokenClass::VerbAction),
        ("shine", TokenClass::VerbAction),
        ("grow", TokenClass::VerbAction),
        ("walk", TokenClass::VerbAction),
        ("come", TokenClass::VerbAction),
        ("see", TokenClass::VerbAction),
        ("say", TokenClass::VerbAction),
        ("eat", TokenClass::VerbAction),
        ("drink", TokenClass::VerbAction),
        ("play", TokenClass::VerbAction),
        ("work", TokenClass::VerbAction),
        ("learn", TokenClass::VerbAction),
        ("think", TokenClass::VerbAction),
        ("know", TokenClass::VerbAction),
        ("love", TokenClass::VerbAction),
        ("hate", TokenClass::VerbAction),
        ("want", TokenClass::VerbAction),
        ("need", TokenClass::VerbAction),
        ("like", TokenClass::VerbAction),
        ("help", TokenClass::VerbAction),
        ("find", TokenClass::VerbAction),
        ("look", TokenClass::VerbAction),
        ("use", TokenClass::VerbAction),
        ("make", TokenClass::VerbAction),
        ("take", TokenClass::VerbAction),
        ("give", TokenClass::VerbAction),
        ("tell", TokenClass::VerbAction),
        ("read", TokenClass::VerbAction),
        ("write", TokenClass::VerbAction),
        ("speak", TokenClass::VerbAction),
        ("listen", TokenClass::VerbAction),
        ("understand", TokenClass::VerbAction),
        ("remember", TokenClass::VerbAction),
        ("forget", TokenClass::VerbAction),
        ("try", TokenClass::VerbAction),
        ("start", TokenClass::VerbAction),
        ("stop", TokenClass::VerbAction),
        ("continue", TokenClass::VerbAction),
        ("finish", TokenClass::VerbAction),
        ("win", TokenClass::VerbAction),
        ("lose", TokenClass::VerbAction),
    ];
    for (w, c) in action_verbs {
        words.push((*w, *c, 3));
    }

    // Cluster 4: Concrete Nouns (German + English, deduplicated)
    let concrete_nouns: &[(&str, TokenClass)] = &[
        ("hund", TokenClass::NounConcrete),
        ("katze", TokenClass::NounConcrete),
        ("vogel", TokenClass::NounConcrete),
        ("stern", TokenClass::NounConcrete),
        ("baum", TokenClass::NounConcrete),
        ("blume", TokenClass::NounConcrete),
        ("wasser", TokenClass::NounConcrete),
        ("feuer", TokenClass::NounConcrete),
        ("erde", TokenClass::NounConcrete),
        ("haus", TokenClass::NounConcrete),
        ("auto", TokenClass::NounConcrete),
        ("buch", TokenClass::NounConcrete),
        ("tisch", TokenClass::NounConcrete),
        ("stuhl", TokenClass::NounConcrete),
        ("fenster", TokenClass::NounConcrete),
        ("tür", TokenClass::NounConcrete),
        ("licht", TokenClass::NounConcrete),
        ("luft", TokenClass::NounConcrete),
        ("sonne", TokenClass::NounConcrete),
        ("mond", TokenClass::NounConcrete),
        ("wolke", TokenClass::NounConcrete),
        ("regen", TokenClass::NounConcrete),
        ("schnee", TokenClass::NounConcrete),
        ("wind", TokenClass::NounConcrete),
        ("meer", TokenClass::NounConcrete),
        ("berg", TokenClass::NounConcrete),
        ("wald", TokenClass::NounConcrete),
        ("feld", TokenClass::NounConcrete),
        ("stadt", TokenClass::NounConcrete),
        ("dorf", TokenClass::NounConcrete),
        ("straße", TokenClass::NounConcrete),
        ("brücke", TokenClass::NounConcrete),
        ("zug", TokenClass::NounConcrete),
        ("flugzeug", TokenClass::NounConcrete),
        ("schiff", TokenClass::NounConcrete),
        ("dog", TokenClass::NounConcrete),
        ("cat", TokenClass::NounConcrete),
        ("bird", TokenClass::NounConcrete),
        ("star", TokenClass::NounConcrete),
        ("tree", TokenClass::NounConcrete),
        ("flower", TokenClass::NounConcrete),
        ("water", TokenClass::NounConcrete),
        ("fire", TokenClass::NounConcrete),
        ("earth", TokenClass::NounConcrete),
        ("house", TokenClass::NounConcrete),
        ("car", TokenClass::NounConcrete),
        ("book", TokenClass::NounConcrete),
        ("table", TokenClass::NounConcrete),
        ("chair", TokenClass::NounConcrete),
        ("window", TokenClass::NounConcrete),
        ("door", TokenClass::NounConcrete),
        ("light", TokenClass::NounConcrete),
        ("air", TokenClass::NounConcrete),
        ("sun", TokenClass::NounConcrete),
        ("moon", TokenClass::NounConcrete),
        ("cloud", TokenClass::NounConcrete),
        ("rain", TokenClass::NounConcrete),
        ("snow", TokenClass::NounConcrete),
        ("sea", TokenClass::NounConcrete),
        ("mountain", TokenClass::NounConcrete),
        ("forest", TokenClass::NounConcrete),
        ("field", TokenClass::NounConcrete),
        ("city", TokenClass::NounConcrete),
        ("village", TokenClass::NounConcrete),
        ("road", TokenClass::NounConcrete),
        ("bridge", TokenClass::NounConcrete),
        ("train", TokenClass::NounConcrete),
        ("plane", TokenClass::NounConcrete),
        ("ship", TokenClass::NounConcrete),
        ("phone", TokenClass::NounConcrete),
        ("computer", TokenClass::NounConcrete),
        ("screen", TokenClass::NounConcrete),
        ("keyboard", TokenClass::NounConcrete),
        ("mouse", TokenClass::NounConcrete),
        ("network", TokenClass::NounConcrete),
        ("data", TokenClass::NounConcrete),
        ("code", TokenClass::NounConcrete),
        ("program", TokenClass::NounConcrete),
        ("file", TokenClass::NounConcrete),
        ("folder", TokenClass::NounConcrete),
        ("music", TokenClass::NounConcrete),
        ("movie", TokenClass::NounConcrete),
        ("game", TokenClass::NounConcrete),
        ("ball", TokenClass::NounConcrete),
        ("team", TokenClass::NounConcrete),
        ("player", TokenClass::NounConcrete),
        ("score", TokenClass::NounConcrete),
    ];
    for (w, c) in concrete_nouns {
        words.push((*w, *c, 4));
    }

    // Cluster 5: Adjectives (German + English, deduplicated)
    let adjectives: &[(&str, TokenClass)] = &[
        ("gut", TokenClass::Adjective),
        ("schlecht", TokenClass::Adjective),
        ("groß", TokenClass::Adjective),
        ("klein", TokenClass::Adjective),
        ("schnell", TokenClass::Adjective),
        ("langsam", TokenClass::Adjective),
        ("warm", TokenClass::Adjective),
        ("kalt", TokenClass::Adjective),
        ("hell", TokenClass::Adjective),
        ("dunkel", TokenClass::Adjective),
        ("good", TokenClass::Adjective),
        ("bad", TokenClass::Adjective),
        ("big", TokenClass::Adjective),
        ("small", TokenClass::Adjective),
        ("fast", TokenClass::Adjective),
        ("slow", TokenClass::Adjective),
        ("cold", TokenClass::Adjective),
        ("bright", TokenClass::Adjective),
        ("dark", TokenClass::Adjective),
        ("new", TokenClass::Adjective),
        ("old", TokenClass::Adjective),
        ("young", TokenClass::Adjective),
        ("strong", TokenClass::Adjective),
        ("weak", TokenClass::Adjective),
        ("hard", TokenClass::Adjective),
        ("soft", TokenClass::Adjective),
        ("heavy", TokenClass::Adjective),
        ("clean", TokenClass::Adjective),
        ("dirty", TokenClass::Adjective),
        ("rich", TokenClass::Adjective),
        ("poor", TokenClass::Adjective),
        ("happy", TokenClass::Adjective),
        ("sad", TokenClass::Adjective),
        ("beautiful", TokenClass::Adjective),
        ("ugly", TokenClass::Adjective),
        ("smart", TokenClass::Adjective),
        ("stupid", TokenClass::Adjective),
        ("brave", TokenClass::Adjective),
        ("scared", TokenClass::Adjective),
        ("friendly", TokenClass::Adjective),
        ("mean", TokenClass::Adjective),
        ("quiet", TokenClass::Adjective),
        ("loud", TokenClass::Adjective),
        ("sweet", TokenClass::Adjective),
        ("sour", TokenClass::Adjective),
        ("fresh", TokenClass::Adjective),
        ("stale", TokenClass::Adjective),
        ("safe", TokenClass::Adjective),
        ("dangerous", TokenClass::Adjective),
        ("easy", TokenClass::Adjective),
        ("clear", TokenClass::Adjective),
        ("confusing", TokenClass::Adjective),
        ("true", TokenClass::Adjective),
        ("false", TokenClass::Adjective),
        ("right", TokenClass::Adjective),
        ("wrong", TokenClass::Adjective),
        ("full", TokenClass::Adjective),
        ("empty", TokenClass::Adjective),
        ("open", TokenClass::Adjective),
        ("closed", TokenClass::Adjective),
        ("alive", TokenClass::Adjective),
        ("dead", TokenClass::Adjective),
        ("real", TokenClass::Adjective),
        ("fake", TokenClass::Adjective),
        ("simple", TokenClass::Adjective),
        ("complex", TokenClass::Adjective),
    ];
    for (w, c) in adjectives {
        words.push((*w, *c, 5));
    }

    // Cluster 6: Question Words & Connectors (~40)
    let questions: &[(&str, TokenClass)] = &[
        ("wer", TokenClass::Determiner),
        ("was", TokenClass::Determiner),
        ("wo", TokenClass::Determiner),
        ("wann", TokenClass::Determiner),
        ("warum", TokenClass::Determiner),
        ("wie", TokenClass::Determiner),
        ("wen", TokenClass::Determiner),
        ("wem", TokenClass::Determiner),
        ("wessen", TokenClass::Determiner),
        ("welcher", TokenClass::Determiner),
        ("who", TokenClass::Determiner),
        ("what", TokenClass::Determiner),
        ("where", TokenClass::Determiner),
        ("when", TokenClass::Determiner),
        ("why", TokenClass::Determiner),
        ("how", TokenClass::Determiner),
        ("which", TokenClass::Determiner),
        ("whose", TokenClass::Determiner),
        ("whom", TokenClass::Determiner),
        ("und", TokenClass::Preposition),
        ("oder", TokenClass::Preposition),
        ("aber", TokenClass::Preposition),
        ("weil", TokenClass::Preposition),
        ("denn", TokenClass::Preposition),
        ("sodass", TokenClass::Preposition),
        ("and", TokenClass::Preposition),
        ("or", TokenClass::Preposition),
        ("but", TokenClass::Preposition),
        ("because", TokenClass::Preposition),
        ("since", TokenClass::Preposition),
        ("although", TokenClass::Preposition),
        ("though", TokenClass::Preposition),
        ("however", TokenClass::Preposition),
        ("therefore", TokenClass::Preposition),
        ("thus", TokenClass::Preposition),
        ("so", TokenClass::Preposition),
        ("yet", TokenClass::Preposition),
        ("still", TokenClass::Preposition),
        ("also", TokenClass::Preposition),
        ("plus", TokenClass::Preposition),
        ("moreover", TokenClass::Preposition),
        ("furthermore", TokenClass::Preposition),
        ("nevertheless", TokenClass::Preposition),
        ("nonetheless", TokenClass::Preposition),
    ];
    for (w, c) in questions {
        words.push((*w, *c, 6));
    }

    words
}

// =============================================================================
// 5. LEXICON BUILDER — Extended multilingual vocabulary
// =============================================================================

pub fn build_extended_lexicon(
    trainer: &mut SemanticTrainer,
    encoder: &mut TokenSpikeEncoder,
    decoder: &mut SpikeTokenDecoder,
) {
    let words = generate_word_list();

    let mut cluster_counts: [usize; 7] = [0; 7];
    for (_, _, cluster_idx) in &words {
        cluster_counts[*cluster_idx] += 1;
    }

    let mut cluster_positions: [usize; 7] = [0; 7];

    for (word, class, cluster_idx) in &words {
        let word_idx = cluster_positions[*cluster_idx];
        let coords = compute_coordinates(*cluster_idx, word_idx, cluster_counts[*cluster_idx]);
        cluster_positions[*cluster_idx] += 1;

        let id = trainer.lexicon.tokens.len();
        let hp = HyperbolicPoint::new(array![coords[0], coords[1]]).unwrap();
        let q = Quaternion::new(coords[0] as f32, coords[1] as f32, 0.0, 0.0).normalize();
        trainer.lexicon.tokens.push(LexiconToken {
            id,
            surface: word.to_string(),
            class: *class,
            embedding: q,
            hyperbolic: hp,
            salience: 0.5,
        });
        trainer.lexicon.word_index.insert(word.to_string(), id);
        trainer.lexicon.class_index.entry(*class).or_default().push(id);
    }

    for token in &trainer.lexicon.tokens {
        trainer.concept_graph.add_concept(&token.surface, token.hyperbolic.clone());
    }

    add_concept_graph_edges(&mut trainer.concept_graph, &words);

    encoder.register_lexicon(&trainer.lexicon);
    decoder.register_lexicon(&trainer.lexicon);
}

fn add_concept_graph_edges(graph: &mut ConceptGraph, words: &[(&str, TokenClass, usize)]) {
    let surfaces: Vec<String> = words.iter().map(|(w, _, _)| w.to_string()).collect();

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); 7];
    for (global_idx, (_, _, cluster_idx)) in words.iter().enumerate() {
        clusters[*cluster_idx].push(global_idx);
    }

    for cluster in &clusters {
        for i in 0..cluster.len() {
            let next = (i + 1) % cluster.len();
            let _ = graph.add_edge(&surfaces[cluster[i]], &surfaces[cluster[next]], RelationType::RelatedTo, 0.5);
            if i + 2 < cluster.len() {
                let _ = graph.add_edge(&surfaces[cluster[i]], &surfaces[cluster[i + 2]], RelationType::RelatedTo, 0.3);
            }
        }
    }

    for i in 0..clusters.len() {
        let next = (i + 1) % clusters.len();
        if !clusters[i].is_empty() && !clusters[next].is_empty() {
            let _ = graph.add_edge(
                &surfaces[clusters[i][0]],
                &surfaces[clusters[next][0]],
                RelationType::RelatedTo,
                0.2,
            );
        }
    }

    let semantic_pairs = vec![
        ("hund", "läuft", RelationType::Causes),
        ("katze", "schläft", RelationType::Causes),
        ("vogel", "fliegt", RelationType::Causes),
        ("stern", "scheint", RelationType::Causes),
        ("baum", "wächst", RelationType::Causes),
        ("dog", "run", RelationType::Causes),
        ("cat", "sleep", RelationType::Causes),
        ("bird", "fly", RelationType::Causes),
        ("sun", "shine", RelationType::Causes),
        ("fire", "warm", RelationType::Causes),
        ("water", "flow", RelationType::Causes),
    ];

    for (src, dst, rel) in semantic_pairs {
        if graph.index.contains_key(src) && graph.index.contains_key(dst) {
            let _ = graph.add_edge(src, dst, rel, 0.8);
        }
    }
}

// =============================================================================
// 6. GEOMETRY — Updated for extended lexicon
// =============================================================================

pub fn standard_geometry() -> WorldGeometry {
    WorldGeometry::new(2, 8, 1.0)
}

// =============================================================================
// 7. COHERENCE TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantics::SemanticTrainer;
    use crate::chat::spike_token_bridge::{TokenSpikeEncoder, SpikeTokenDecoder};

    #[test]
    fn extended_lexicon_size() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut encoder = TokenSpikeEncoder::new(3.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);
        build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
        assert!(trainer.lexicon.tokens.len() >= 200, "Lexicon size {} < 200", trainer.lexicon.tokens.len());
    }

    #[test]
    fn all_points_inside_ball() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut encoder = TokenSpikeEncoder::new(3.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);
        build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
        for token in &trainer.lexicon.tokens {
            let norm = token.hyperbolic.coords.iter().map(|c| c * c).sum::<f64>().sqrt();
            assert!(norm < 1.0, "Token '{}' has norm {} >= 1.0", token.surface, norm);
        }
    }

    #[test]
    fn english_words_present() {
        let english_count = generate_word_list()
            .iter()
            .filter(|(w, _, _)| {
                w.chars().all(|c| c.is_ascii_alphabetic() && c.is_ascii_lowercase())
                    || w.chars().all(|c| c.is_ascii_alphabetic() && c.is_ascii_uppercase())
            })
            .count();
        assert!(english_count >= 75, "English word count {} < 75", english_count);
    }

    #[test]
    fn no_duplicate_surfaces() {
        let words = generate_word_list();
        let mut seen = std::collections::HashSet::new();
        for (word, _, _) in &words {
            assert!(seen.insert(*word), "Duplicate surface: {}", word);
        }
    }

    #[test]
    fn concept_graph_connected_components() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut encoder = TokenSpikeEncoder::new(3.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);
        build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
        let graph = &trainer.concept_graph;

        assert!(graph.nodes.len() >= 200, "Concept graph has {} nodes < 200", graph.nodes.len());

        let mut visited = vec![false; graph.nodes.len()];
        let mut component_sizes = Vec::new();

        for start in 0..graph.nodes.len() {
            if !visited[start] {
                let mut stack = vec![start];
                let mut size = 0;
                visited[start] = true;
                while let Some(node) = stack.pop() {
                    size += 1;
                    for edge in &graph.edges {
                        if edge.source == node && !visited[edge.target] {
                            visited[edge.target] = true;
                            stack.push(edge.target);
                        }
                        if edge.target == node && !visited[edge.source] {
                            visited[edge.source] = true;
                            stack.push(edge.source);
                        }
                    }
                }
                component_sizes.push(size);
            }
        }

        component_sizes.sort_unstable();
        let largest = component_sizes.last().copied().unwrap_or(0);
        let total = graph.nodes.len();
        let connectivity = largest as f64 / total as f64;
        assert!(connectivity > 0.5, "Connectivity {:.2} <= 0.5, components: {:?}", connectivity, component_sizes);
    }

    #[test]
    fn no_nan_in_embeddings() {
        let mut trainer = SemanticTrainer::new(1.0);
        let mut encoder = TokenSpikeEncoder::new(3.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);
        build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
        for token in &trainer.lexicon.tokens {
            for (i, &c) in [token.embedding.w, token.embedding.x, token.embedding.y, token.embedding.z].iter().enumerate() {
                assert!(c.is_finite(), "Token '{}' quaternion component {} is NaN/Inf", token.surface, i);
            }
            for (i, &c) in token.hyperbolic.coords.iter().enumerate() {
                assert!(c.is_finite(), "Token '{}' hyperbolic coord {} is NaN/Inf", token.surface, i);
            }
        }
    }
}
