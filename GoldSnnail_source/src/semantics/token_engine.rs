//! Token Engine — Semantic Lexicon, Noise Injection, and Multi-Objective Reward
//!
//! This module bridges raw spike trains and abstract knowledge:
//! - `Lexicon`: Hierarchical token embeddings in hyperbolic space
//! - `NoiseInjector`: Controlled exploration via token corruption
//! - `SemanticRewardEngine`: 6-dimensional reward (coherence, syntax, prediction, novelty, robustness, compression)
//! - `TokenComposer`: Sentence generator with grammatical patterns
//! - `SemanticTrainer`: Connects everything to the AGI stack

use crate::geometry::{HyperbolicPoint, PoincareBall, Quaternion};
use crate::semantics::{ConceptGraph, RelationType, HyperbolicContrastive};
use ndarray::array;
use std::collections::{HashMap, HashSet};

// =============================================================================
// 1. THE LEXICON — All building blocks of language
// =============================================================================

/// Token category determines grammatical/semantic role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenClass {
    Determiner,      // der, die, das, ein, kein
    NounConcrete,    // Hund, Tisch, Stein
    NounAbstract,    // Liebe, Idee, Freiheit
    VerbAction,      // läuft, springt, denkt
    VerbState,       // ist, scheint, bleibt
    Adjective,       // groß, schnell, heiß
    Preposition,     // in, auf, unter, mit
    SemanticRole,    // AGENT, PATIENT, THEMA, LOCATION
    GrammarMarker,   // NOM, AKK, DAT, GEN, SG, PL
    Punctuation,     // . , ; !
    Noise,           // Noise token for exploration
}

/// A token in the lexicon = a neural building block
#[derive(Debug, Clone)]
pub struct LexiconToken {
    pub id: usize,
    pub surface: String,
    pub class: TokenClass,
    pub embedding: Quaternion,
    pub hyperbolic: HyperbolicPoint,
    /// Intrinsic salience: how "important" is this token? (0..1)
    pub salience: f64,
}

/// The complete lexicon — DOD: flat Vecs, no nested maps
#[derive(Debug, Clone)]
pub struct Lexicon {
    pub tokens: Vec<LexiconToken>,
    /// Index: surface → id
    pub word_index: HashMap<String, usize>,
    /// Index: class → [ids]
    pub class_index: HashMap<TokenClass, Vec<usize>>,
    pub ball: PoincareBall,
}

impl Lexicon {
    pub fn new(curvature: f64) -> Self {
        let mut lex = Self {
            tokens: Vec::new(),
            word_index: HashMap::new(),
            class_index: HashMap::new(),
            ball: PoincareBall::new(curvature),
        };
        lex.build_universal_tokens();
        lex
    }

    /// Builds the universal token vocabulary
    fn build_universal_tokens(&mut self) {
        // --- Determiners (near center = frequent, generic) ---
        self.add("der", TokenClass::Determiner, array![0.02, 0.0], 0.3);
        self.add("die", TokenClass::Determiner, array![0.02, 0.01], 0.3);
        self.add("das", TokenClass::Determiner, array![0.02, -0.01], 0.3);
        self.add("ein", TokenClass::Determiner, array![0.03, 0.0], 0.25);
        self.add("kein", TokenClass::Determiner, array![0.03, 0.02], 0.25);

        // --- Nouns: Concrete (middle radius) ---
        // Animals (cluster in hyperbolic space)
        self.add("hund", TokenClass::NounConcrete, array![0.15, 0.05], 0.6);
        self.add("katze", TokenClass::NounConcrete, array![0.15, 0.08], 0.6);
        self.add("vogel", TokenClass::NounConcrete, array![0.14, 0.12], 0.55);
        self.add("fisch", TokenClass::NounConcrete, array![0.16, 0.02], 0.55);
        self.add("tier", TokenClass::NounConcrete, array![0.10, 0.07], 0.7); // Hypernym

        // Objects
        self.add("tisch", TokenClass::NounConcrete, array![-0.10, 0.05], 0.5);
        self.add("haus", TokenClass::NounConcrete, array![-0.08, 0.10], 0.55);
        self.add("stein", TokenClass::NounConcrete, array![-0.12, -0.05], 0.4);

        // --- Nouns: Abstract (nearer boundary = specific, complex) ---
        self.add("liebe", TokenClass::NounAbstract, array![0.40, 0.10], 0.8);
        self.add("freiheit", TokenClass::NounAbstract, array![0.42, 0.15], 0.8);
        self.add("idee", TokenClass::NounAbstract, array![0.38, 0.05], 0.75);
        self.add("wahrheit", TokenClass::NounAbstract, array![0.45, 0.0], 0.85);

        // --- Verbs: Action ---
        self.add("läuft", TokenClass::VerbAction, array![0.05, 0.20], 0.6);
        self.add("springt", TokenClass::VerbAction, array![0.06, 0.22], 0.6);
        self.add("denkt", TokenClass::VerbAction, array![0.20, 0.25], 0.7);
        self.add("sieht", TokenClass::VerbAction, array![0.04, 0.18], 0.6);
        self.add("frisst", TokenClass::VerbAction, array![0.08, 0.15], 0.55);

        // --- Verbs: State ---
        self.add("ist", TokenClass::VerbState, array![0.0, 0.15], 0.5);
        self.add("scheint", TokenClass::VerbState, array![0.02, 0.17], 0.5);
        self.add("bleibt", TokenClass::VerbState, array![-0.01, 0.14], 0.5);

        // --- Adjectives ---
        self.add("groß", TokenClass::Adjective, array![0.08, -0.10], 0.45);
        self.add("klein", TokenClass::Adjective, array![0.06, -0.12], 0.45);
        self.add("schnell", TokenClass::Adjective, array![0.10, -0.08], 0.5);
        self.add("heiß", TokenClass::Adjective, array![0.25, -0.15], 0.55);
        self.add("kalt", TokenClass::Adjective, array![0.22, -0.18], 0.55);

        // --- Prepositions ---
        self.add("in", TokenClass::Preposition, array![-0.05, 0.05], 0.3);
        self.add("auf", TokenClass::Preposition, array![-0.04, 0.06], 0.3);
        self.add("unter", TokenClass::Preposition, array![-0.06, 0.04], 0.3);
        self.add("mit", TokenClass::Preposition, array![-0.03, 0.07], 0.35);

        // --- Semantic Roles (control argument structure) ---
        self.add("AGENT", TokenClass::SemanticRole, array![0.30, 0.30], 0.9);
        self.add("PATIENT", TokenClass::SemanticRole, array![0.30, 0.25], 0.9);
        self.add("THEMA", TokenClass::SemanticRole, array![0.28, 0.28], 0.85);
        self.add("LOCATION", TokenClass::SemanticRole, array![0.25, 0.32], 0.8);

        // --- Grammar Markers ---
        self.add("NOM", TokenClass::GrammarMarker, array![0.0, 0.0], 0.2);
        self.add("AKK", TokenClass::GrammarMarker, array![0.01, 0.0], 0.2);
        self.add("DAT", TokenClass::GrammarMarker, array![0.0, 0.01], 0.2);
        self.add("SG", TokenClass::GrammarMarker, array![-0.01, 0.0], 0.15);
        self.add("PL", TokenClass::GrammarMarker, array![-0.01, 0.01], 0.15);

        // --- Noise tokens (for exploration) ---
        self.add("???", TokenClass::Noise, array![0.0, 0.0], 0.1);
        self.add("###", TokenClass::Noise, array![0.0, 0.0], 0.1);
    }

    fn add(&mut self, surface: &str, class: TokenClass, coords: ndarray::Array1<f64>, salience: f64) {
        let id = self.tokens.len();
        let hp = HyperbolicPoint::new(array![coords[0] * 0.9, coords[1] * 0.9]).unwrap();
        let q = Quaternion::new(coords[0] as f32, coords[1] as f32, 0.0, 0.0).normalize();

        self.tokens.push(LexiconToken {
            id,
            surface: surface.to_string(),
            class,
            embedding: q,
            hyperbolic: hp,
            salience,
        });
        self.word_index.insert(surface.to_string(), id);
        self.class_index.entry(class).or_default().push(id);
    }

    pub fn get(&self, word: &str) -> Option<&LexiconToken> {
        self.word_index.get(word).and_then(|&id| self.tokens.get(id))
    }

    pub fn random_from_class(&self, class: TokenClass) -> Option<&LexiconToken> {
        use rand::seq::SliceRandom;
        let ids = self.class_index.get(&class)?;
        let mut rng = rand::thread_rng();
        ids.choose(&mut rng).and_then(|&id| self.tokens.get(id))
    }
}

// =============================================================================
// 2. NOISE INJECTOR — Controlled exploration
// =============================================================================

pub struct NoiseInjector {
    /// Probability to replace a token with noise
    pub token_noise_prob: f64,
    /// Probability to insert an extra noise token
    pub insertion_prob: f64,
    /// Strength of hyperbolic jitter (0 = no noise)
    pub hyperbolic_jitter: f64,
    /// Enabled/disabled
    pub enabled: bool,
}

impl NoiseInjector {
    pub fn new(token_noise: f64, insertion: f64, jitter: f64) -> Self {
        Self {
            token_noise_prob: token_noise,
            insertion_prob: insertion,
            hyperbolic_jitter: jitter,
            enabled: true,
        }
    }

    /// Injects noise into a token sequence
    pub fn corrupt_sequence(
        &self,
        sequence: &[String],
        _lexicon: &Lexicon,
    ) -> Vec<String> {
        if !self.enabled { return sequence.to_vec(); }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut corrupted = Vec::with_capacity(sequence.len() + 2);

        for token in sequence {
            // Insertion: insert noise token before
            if rng.r#gen::<f64>() < self.insertion_prob {
                corrupted.push("???".to_string());
            }

            // Substitution: replace token with noise
            if rng.r#gen::<f64>() < self.token_noise_prob {
                corrupted.push("###".to_string());
            } else {
                corrupted.push(token.clone());
            }
        }
        corrupted
    }

    /// Hyperbolic jitter: shifts a point slightly in the Poincaré ball
    pub fn jitter(&self, point: &HyperbolicPoint) -> HyperbolicPoint {
        if !self.enabled || self.hyperbolic_jitter < 1e-12 {
            return point.clone();
        }
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut new_coords = point.coords.clone();
        for i in 0..new_coords.len() {
            let noise = (rng.r#gen::<f64>() - 0.5) * 2.0 * self.hyperbolic_jitter;
            new_coords[i] += noise;
        }
        // Project back into ball
        let norm = new_coords.iter().map(|x| x*x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm;
            for x in &mut new_coords {
                *x *= scale;
            }
        }
        HyperbolicPoint::new(ndarray::Array1::from(new_coords)).unwrap_or_else(|_| point.clone())
    }
}

// =============================================================================
// 3. SEMANTIC REWARD SYSTEM — Multi-Objective
// =============================================================================

/// Reward engine with multiple complementary objectives
pub struct SemanticRewardEngine {
    pub ball: PoincareBall,
    pub contrastive: HyperbolicContrastive,
    /// Weighting of reward components
    pub weights: RewardWeights,
    /// Known valid sequences (for syntax check)
    pub valid_patterns: HashSet<Vec<TokenClass>>,
}

#[derive(Debug, Clone, Copy)]
pub struct RewardWeights {
    pub semantic_coherence: f64,   // Hyperbolic proximity of related tokens
    pub syntactic_wellformed: f64, // Grammatical correctness
    pub prediction_accuracy: f64,  // World model error
    pub novelty: f64,              // New but meaningful combinations
    pub noise_tolerance: f64,      // Robustness against noise
    pub compression_quality: f64,  // Delta encoder efficiency
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            semantic_coherence: 1.0,
            syntactic_wellformed: 0.8,
            prediction_accuracy: 1.2,
            novelty: 0.5,
            noise_tolerance: 0.6,
            compression_quality: 0.4,
        }
    }
}

/// A single reward batch
#[derive(Debug, Clone, Default)]
pub struct RewardSignal {
    pub total: f64,
    pub semantic: f64,
    pub syntactic: f64,
    pub prediction: f64,
    pub novelty: f64,
    pub noise_robust: f64,
    pub compression: f64,
}

impl SemanticRewardEngine {
    pub fn new(curvature: f64, weights: RewardWeights) -> Self {
        Self {
            ball: PoincareBall::new(curvature),
            contrastive: HyperbolicContrastive::new(curvature, 0.1, 0.5),
            weights,
            valid_patterns: HashSet::new(),
        }
    }

    /// Register a syntactically valid pattern
    pub fn learn_pattern(&mut self, pattern: Vec<TokenClass>) {
        self.valid_patterns.insert(pattern);
    }

    /// MAIN FUNCTION: Computes total reward for a sequence
    pub fn compute(
        &mut self,
        sequence: &[LexiconToken],
        predicted_next: Option<&HyperbolicPoint>,
        actual_next: Option<&HyperbolicPoint>,
        is_noisy: bool,
        concept_graph: &ConceptGraph,
    ) -> RewardSignal {
        let mut signal = RewardSignal::default();

        // 1. SEMANTIC COHERENCE
        // Similar tokens should be close in hyperbolic space
        if sequence.len() >= 2 {
            let mut coherence_sum = 0.0;
            let mut pairs = 0;
            for window in sequence.windows(2) {
                let a = &window[0].hyperbolic;
                let b = &window[1].hyperbolic;
                if let Ok(dist) = self.ball.distance(a, b) {
                    // Low distance = high coherence
                    coherence_sum += (-dist).exp();
                    pairs += 1;
                }
            }
            if pairs > 0 {
                signal.semantic = coherence_sum / pairs as f64;
            }
        }

        // 2. SYNTACTIC WELLFORMEDNESS
        let pattern: Vec<TokenClass> = sequence.iter().map(|t| t.class).collect();
        if self.valid_patterns.contains(&pattern) {
            signal.syntactic = 1.0;
        } else {
            // Partial reward for sub-patterns
            let mut best_match: f64 = 0.0;
            for valid in &self.valid_patterns {
                if valid.len() <= pattern.len() {
                    let overlap = pattern.windows(valid.len()).any(|w| w == valid.as_slice());
                    if overlap {
                        best_match = best_match.max(0.5f64);
                    }
                }
            }
            signal.syntactic = best_match;
        }

        // 3. PREDICTION ACCURACY (World Model)
        if let (Some(pred), Some(actual)) = (predicted_next, actual_next) {
            if let Ok(dist) = self.ball.distance(pred, actual) {
                // Low distance = good prediction
                signal.prediction = (-dist).exp();
            }
        }

        // 4. NOVELTY — Reward new but coherent combinations
        if sequence.len() >= 2 {
            let mut novel_edges = 0;
            let mut total_edges = 0;
            for window in sequence.windows(2) {
                let a_id = window[0].id;
                let b_id = window[1].id;
                total_edges += 1;
                // Check if this edge exists in ConceptGraph
                let exists = concept_graph.edges.iter().any(|e| {
                    (e.source == a_id && e.target == b_id) ||
                    (e.source == b_id && e.target == a_id)
                });
                if !exists {
                    novel_edges += 1;
                }
            }
            if total_edges > 0 {
                // Reward moderate novelty (not all new, not all known)
                let ratio = novel_edges as f64 / total_edges as f64;
                signal.novelty = 1.0 - (ratio - 0.3).abs() * 2.0;
                signal.novelty = signal.novelty.max(0.0);
            }
        }

        // 5. NOISE ROBUSTNESS
        if is_noisy {
            // If sequence is still coherent despite noise → high reward!
            signal.noise_robust = signal.semantic * 1.5;
        } else {
            signal.noise_robust = 0.0;
        }

        // 6. COMPRESSION QUALITY
        // Implicitly: if sequence is compressible (repeated patterns), it's good
        if sequence.len() >= 4 {
            let mut repeats = 0;
            for i in 0..sequence.len()-2 {
                if sequence[i].surface == sequence[i+2].surface {
                    repeats += 1;
                }
            }
            signal.compression = (repeats as f64 / (sequence.len() as f64 - 2.0)).min(1.0);
        }

        // Weighted sum
        signal.total = 
            signal.semantic * self.weights.semantic_coherence +
            signal.syntactic * self.weights.syntactic_wellformed +
            signal.prediction * self.weights.prediction_accuracy +
            signal.novelty * self.weights.novelty +
            signal.noise_robust * self.weights.noise_tolerance +
            signal.compression * self.weights.compression_quality;

        signal
    }

    /// Intrinsic curiosity reward: rewards moderate prediction error
    pub fn curiosity_reward(&self, prediction_error: f64) -> f64 {
        // Too small = boring, too large = chaotic
        // Optimum at moderate error
        let optimal = 0.3;
        1.0 - (prediction_error - optimal).abs() * 2.0
    }
}

// =============================================================================
// 4. TOKEN COMPOSER — Builds sequences from the lexicon
// =============================================================================

pub struct TokenComposer<'a> {
    pub lexicon: &'a Lexicon,
    pub noise: NoiseInjector,
}

impl<'a> TokenComposer<'a> {
    pub fn new(lexicon: &'a Lexicon) -> Self {
        Self {
            lexicon,
            noise: NoiseInjector::new(0.05, 0.02, 0.01),
        }
    }

    /// Builds a simple sentence: [DET] [NOUN] [VERB]
    pub fn build_sentence_simple(&self, noun: &str, verb: &str) -> Vec<String> {
        let det = if noun == "hund" || noun == "vogel" { "der" } else { "die" };
        vec![det.to_string(), noun.to_string(), verb.to_string()]
    }

    /// Builds a complex sentence with adjective and preposition
    pub fn build_sentence_complex(
        &self,
        det: &str,
        adj: &str,
        noun: &str,
        verb: &str,
        prep: &str,
        location: &str,
    ) -> Vec<String> {
        vec![
            det.to_string(),
            adj.to_string(),
            noun.to_string(),
            verb.to_string(),
            prep.to_string(),
            location.to_string(),
        ]
    }

    /// Generates a batch of training sequences
    pub fn generate_training_batch(&self, count: usize) -> Vec<Vec<String>> {
        let nouns = ["hund", "katze", "vogel", "tisch", "haus"];
        let verbs = ["läuft", "springt", "ist", "sieht"];
        let adjs = ["groß", "klein", "schnell"];
        let preps = ["in", "auf", "unter"];

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut batch = Vec::with_capacity(count);

        for _ in 0..count {
            let pattern = rng.r#gen::<u32>() % 3;
            let seq = match pattern {
                0 => self.build_sentence_simple(
                    nouns[rng.r#gen::<usize>() % nouns.len()],
                    verbs[rng.r#gen::<usize>() % verbs.len()],
                ),
                1 => {
                    let n = nouns[rng.r#gen::<usize>() % nouns.len()];
                    let v = verbs[rng.r#gen::<usize>() % verbs.len()];
                    let a = adjs[rng.r#gen::<usize>() % adjs.len()];
                    let det = if n == "hund" || n == "vogel" { "der" } else { "die" };
                    vec![det.to_string(), a.to_string(), n.to_string(), v.to_string()]
                },
                _ => {
                    let n = nouns[rng.r#gen::<usize>() % nouns.len()];
                    let v = verbs[rng.r#gen::<usize>() % verbs.len()];
                    let p = preps[rng.r#gen::<usize>() % preps.len()];
                    let loc = "haus";
                    let det = if n == "hund" || n == "vogel" { "der" } else { "die" };
                    vec![det.to_string(), n.to_string(), v.to_string(), p.to_string(), loc.to_string()]
                },
            };
            batch.push(seq);
        }
        batch
    }

    /// Converts String sequence to LexiconToken sequence
    pub fn resolve(&self, words: &[String]) -> Vec<LexiconToken> {
        words.iter()
            .filter_map(|w| self.lexicon.get(w).cloned())
            .collect()
    }
}

// =============================================================================
// 5. TRAINING LOOP INTEGRATION
// =============================================================================

/// Connects token engine to the AGI stack
pub struct SemanticTrainer {
    pub lexicon: Lexicon,
    pub composer: TokenComposer<'static>,
    pub reward_engine: SemanticRewardEngine,
    pub concept_graph: ConceptGraph,
}

impl SemanticTrainer {
    pub fn new(curvature: f64) -> Self {
        let lexicon = Lexicon::new(curvature);
        // SAFETY: In real code, use Arc<Lexicon> to avoid lifetime issues
        // For now, we leak the reference to get a 'static lifetime
        let leaked: &'static Lexicon = Box::leak(Box::new(lexicon));
        let composer = TokenComposer::new(leaked);
        let mut engine = SemanticRewardEngine::new(curvature, RewardWeights::default());
        let mut graph = ConceptGraph::new(curvature);

        // Load initial taxonomy into graph
        Self::seed_taxonomy(&mut graph, leaked);

        // Learn syntactic patterns
        engine.learn_pattern(vec![TokenClass::Determiner, TokenClass::NounConcrete, TokenClass::VerbAction]);
        engine.learn_pattern(vec![TokenClass::Determiner, TokenClass::Adjective, TokenClass::NounConcrete, TokenClass::VerbAction]);

        Self {
            lexicon: leaked.clone(),
            composer,
            reward_engine: engine,
            concept_graph: graph,
        }
    }

    fn seed_taxonomy(graph: &mut ConceptGraph, lexicon: &Lexicon) {
        // Add lexicon tokens as concept nodes
        for token in &lexicon.tokens {
            graph.add_concept(&token.surface, token.hyperbolic.clone());
        }
        // Taxonomy edges
        let _ = graph.add_edge("hund", "tier", RelationType::IsA, 0.9);
        let _ = graph.add_edge("katze", "tier", RelationType::IsA, 0.9);
        let _ = graph.add_edge("vogel", "tier", RelationType::IsA, 0.9);
        let _ = graph.add_edge("fisch", "tier", RelationType::IsA, 0.9);
    }

    /// One training step: sequence → reward → feedback
    pub fn train_step(&mut self, sequence_words: &[String], is_noisy: bool) -> RewardSignal {
        let tokens = self.composer.resolve(sequence_words);
        if tokens.is_empty() {
            return RewardSignal::default();
        }

        // Simulate World Model prediction (in real code: from WorldModel)
        let predicted = tokens.last().map(|t| t.hyperbolic.clone());
        let actual = predicted.clone(); // Placeholder

        let reward = self.reward_engine.compute(
            &tokens,
            predicted.as_ref(),
            actual.as_ref(),
            is_noisy,
            &self.concept_graph,
        );

        // Here RSTDP update would happen:
        // reward.total → modulates weight change

        reward
    }

    /// Train with noise injection
    pub fn train_with_noise(&mut self, clean_sequence: &[String]) -> (RewardSignal, RewardSignal) {
        // Clean
        let clean_reward = self.train_step(clean_sequence, false);

        // Noisy
        let noisy_words = self.composer.noise.corrupt_sequence(clean_sequence, &self.lexicon);
        let noisy_reward = self.train_step(&noisy_words, true);

        (clean_reward, noisy_reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexicon_has_all_classes() {
        let lex = Lexicon::new(1.0);
        assert!(lex.class_index.contains_key(&TokenClass::Determiner));
        assert!(lex.class_index.contains_key(&TokenClass::NounConcrete));
        assert!(lex.class_index.contains_key(&TokenClass::VerbAction));
        assert!(lex.class_index.contains_key(&TokenClass::Noise));
        assert!(lex.tokens.len() > 30);
    }

    #[test]
    fn noise_corrupts_sequence() {
        let lex = Lexicon::new(1.0);
        let noise = NoiseInjector::new(0.99, 0.99, 0.1);
        let seq = vec!["der".into(), "hund".into(), "läuft".into()];
        let corrupted = noise.corrupt_sequence(&seq, &lex);
        assert_ne!(corrupted, seq); // With high probability changed
    }

    #[test]
    fn reward_for_valid_sentence() {
        let mut trainer = SemanticTrainer::new(1.0);
        let seq = vec!["der".into(), "hund".into(), "läuft".into()];
        let reward = trainer.train_step(&seq, false);
        assert!(reward.total > 0.0, "Valid sentence should get positive reward");
        assert!(reward.syntactic > 0.0, "Should be recognized as syntactic");
    }

    #[test]
    fn reward_higher_for_coherent_than_random() {
        let mut trainer = SemanticTrainer::new(1.0);
        let coherent = vec!["der".into(), "hund".into(), "läuft".into()];
        let random = vec!["freiheit".into(), "tisch".into(), "heiß".into()];

        let r_coherent = trainer.train_step(&coherent, false).total;
        let r_random = trainer.train_step(&random, false).total;

        assert!(r_coherent > r_random, 
            "Coherent sentence should get more reward than random: {} vs {}", 
            r_coherent, r_random);
    }

    #[test]
    fn noise_tolerance_reward() {
        let mut trainer = SemanticTrainer::new(1.0);
        let clean = vec!["der".into(), "hund".into(), "läuft".into()];
        let (r_clean, r_noisy) = trainer.train_with_noise(&clean);
        
        // Noisy should still have positive components
        assert!(r_noisy.noise_robust >= 0.0);
        println!("Clean: {:.3}, Noisy: {:.3}", r_clean.total, r_noisy.total);
    }

    #[test]
    fn composer_generates_batch() {
        let lex = Lexicon::new(1.0);
        let composer = TokenComposer::new(&lex);
        let batch = composer.generate_training_batch(10);
        assert_eq!(batch.len(), 10);
        assert!(batch.iter().all(|s| !s.is_empty()));
    }
}
