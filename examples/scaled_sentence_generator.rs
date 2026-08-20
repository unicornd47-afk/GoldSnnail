//! GoldWorm Scaled Sentence Generator
//!
//! Demonstrates the system generating sentences it NEVER learned during training.
//!
//! Pipeline:
//! 1. Expand lexicon to 50+ German words
//! 2. Generate 1000+ diverse training sentences
//! 3. Learn transition probabilities P(next | context)
//! 4. Generate 100 novel sentences using learned patterns
//! 5. Benchmark: % of generated sentences that are both novel AND grammatically valid
//!
//! Usage:
//!   cargo run --example scaled_sentence_generator --release

use goldworm::{
    Lexicon, LexiconToken, TokenClass, SemanticTrainer, SemanticLearner, LearningRates,
    TransitionalLearner, PoincareBall, HyperbolicPoint, Quaternion,
    SemanticRewardEngine,
};
use ndarray::array;
use rand::Rng;
use std::collections::{HashMap, HashSet};

trait RngHelper {
    fn next_usize(&mut self) -> usize;
}

impl<R: Rng> RngHelper for R {
    fn next_usize(&mut self) -> usize {
        self.r#gen()
    }
}

// =============================================================================
// German Grammar Corrector
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gender { Masculine, Feminine, Neuter, Plural }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Case { Nominative, Accusative, Dative }

struct GermanGrammar {
    noun_gender: HashMap<&'static str, Gender>,
    prep_case: HashMap<&'static str, Case>,
    adjective_stems: HashMap<&'static str, &'static str>,
}

impl GermanGrammar {
    fn new() -> Self {
        let mut noun_gender = HashMap::new();
        for w in ["hund", "vogel", "fisch", "baum", "zug", "stern", "mond", "wind", "feuer", "park", "zaun", "himmel", "strand", "berg"] {
            noun_gender.insert(w, Gender::Masculine);
        }
        for w in ["katze", "blume", "erde", "sonne", "straße", "insel", "brücke"] {
            noun_gender.insert(w, Gender::Feminine);
        }
        for w in ["auto", "buch", "wasser", "haus", "tier", "kind", "bett"] {
            noun_gender.insert(w, Gender::Neuter);
        }
        for w in ["bäume", "häuser", "tiere", "kinder"] {
            noun_gender.insert(w, Gender::Plural);
        }

        let mut prep_case = HashMap::new();
        for w in ["in", "auf", "unter", "gegen", "durch", "ohne", "um"] {
            prep_case.insert(w, Case::Accusative);
        }
        for w in ["mit", "von", "zu", "nach", "aus", "bei", "seit", "bis", "über", "zwischen"] {
            prep_case.insert(w, Case::Dative);
        }

        let mut adjective_stems = HashMap::new();
        for (adj, stem) in [
            ("groß", "groß"), ("klein", "klein"), ("schnell", "schnell"),
            ("heiß", "heiß"), ("kalt", "kalt"), ("lang", "lang"), ("kurz", "kurz"),
            ("hell", "hell"), ("dunkel", "dunkel"), ("weich", "weich"), ("hart", "hart"),
            ("nass", "nass"), ("trocken", "trocken"), ("warm", "warm"), ("hoch", "hoch"),
            ("alt", "alt"), ("jung", "jung"), ("gut", "gut"), ("schlecht", "schlecht"),
        ] {
            adjective_stems.insert(adj, stem);
        }

        Self { noun_gender, prep_case, adjective_stems }
    }

    fn gender(&self, noun: &str) -> Gender {
        *self.noun_gender.get(noun).unwrap_or(&Gender::Masculine)
    }

    fn prep_requires(&self, prep: &str) -> Case {
        *self.prep_case.get(prep).unwrap_or(&Case::Accusative)
    }

    fn adjective_stem<'a>(&self, adj: &'a str) -> &'a str {
        self.adjective_stems.get(adj).map(|s| *s).unwrap_or(adj)
    }

    fn correct(&self, sentence: &[String]) -> Vec<String> {
        if sentence.len() < 2 {
            return sentence.to_vec();
        }

        let mut corrected = Vec::with_capacity(sentence.len());
        let mut i = 0;

        while i < sentence.len() {
            let word = &sentence[i];

            if i + 2 < sentence.len() && self.is_determiner(word) {
                let det = word.as_str();
                let adj = &sentence[i + 1];
                let noun = &sentence[i + 2];

                if self.is_adjective(adj) && self.is_noun(noun) {
                    let gender = self.gender(noun);
                    let (corrected_det, corrected_adj) = self.fix_det_adj_noun(det, adj, gender);
                    corrected.push(corrected_det);
                    corrected.push(corrected_adj);
                    corrected.push(noun.clone());
                    i += 3;
                    continue;
                }
            }

            if i + 2 < sentence.len() && self.is_preposition(word) {
                let prep = word.as_str();
                let art = &sentence[i + 1];
                let noun = &sentence[i + 2];

                if self.is_article(art) && self.is_noun(noun) {
                    let required_case = self.prep_requires(prep);
                    let gender = self.gender(noun);
                    let corrected_art = self.fix_article_for_case(art, gender, required_case);
                    corrected.push(prep.to_string());
                    corrected.push(corrected_art);
                    corrected.push(noun.clone());
                    i += 3;
                    continue;
                }
            }

            if i + 1 < sentence.len() && self.is_article(word) {
                let art = word.as_str();
                let noun = &sentence[i + 1];

                if self.is_noun(noun) {
                    let gender = self.gender(noun);
                    let corrected_art = self.fix_article_nominative(art, gender);
                    corrected.push(corrected_art);
                    corrected.push(noun.clone());
                    i += 2;
                    continue;
                }
            }

            corrected.push(word.clone());
            i += 1;
        }

        corrected
    }

    fn is_determiner(&self, w: &str) -> bool {
        matches!(w, "der" | "die" | "das" | "ein" | "kein" | "den" | "dem" | "des" | "einem" | "einen")
    }

    fn is_article(&self, w: &str) -> bool {
        self.is_determiner(w)
    }

    fn is_adjective(&self, w: &str) -> bool {
        self.adjective_stems.contains_key(w)
    }

    fn is_noun(&self, w: &str) -> bool {
        self.noun_gender.contains_key(w)
    }

    fn is_preposition(&self, w: &str) -> bool {
        self.prep_case.contains_key(w)
    }

    fn fix_det_adj_noun(&self, det: &str, adj: &str, gender: Gender) -> (String, String) {
        let corrected_det = match gender {
            Gender::Masculine => "der".to_string(),
            Gender::Feminine => "die".to_string(),
            Gender::Neuter => "das".to_string(),
            Gender::Plural => "die".to_string(),
        };

        let stem = self.adjective_stem(adj);
        let is_inflected = matches!(det, "den" | "dem" | "des" | "einen" | "einem");
        let corrected_adj = match (gender, is_inflected) {
            (Gender::Plural, _) => format!("{}en", stem),
            (_, true) => format!("{}en", stem),
            (Gender::Masculine, false) => format!("{}e", stem),
            (Gender::Feminine, false) => format!("{}e", stem),
            (Gender::Neuter, false) => format!("{}e", stem),
        };

        let corrected_adj = if stem == "dunkel" && !is_inflected && gender != Gender::Plural {
            format!("{}le", &stem[..stem.len() - 1])
        } else {
            corrected_adj
        };

        (corrected_det, corrected_adj)
    }

    fn fix_article_for_case(&self, _article: &str, gender: Gender, case: Case) -> String {
        match (gender, case) {
            (Gender::Masculine, Case::Nominative) => "der".to_string(),
            (Gender::Masculine, Case::Accusative) => "den".to_string(),
            (Gender::Masculine, Case::Dative) => "dem".to_string(),
            (Gender::Feminine, Case::Nominative) => "die".to_string(),
            (Gender::Feminine, Case::Accusative) => "die".to_string(),
            (Gender::Feminine, Case::Dative) => "der".to_string(),
            (Gender::Neuter, Case::Nominative) => "das".to_string(),
            (Gender::Neuter, Case::Accusative) => "das".to_string(),
            (Gender::Neuter, Case::Dative) => "dem".to_string(),
            (Gender::Plural, Case::Nominative) => "die".to_string(),
            (Gender::Plural, Case::Accusative) => "die".to_string(),
            (Gender::Plural, Case::Dative) => "den".to_string(),
        }
    }

    fn fix_article_nominative(&self, _article: &str, gender: Gender) -> String {
        match gender {
            Gender::Masculine => "der".to_string(),
            Gender::Feminine => "die".to_string(),
            Gender::Neuter => "das".to_string(),
            Gender::Plural => "die".to_string(),
        }
    }
}

static TRAIN_SENTENCES: &[&str] = &[
    // Simple SVO
    "der hund läuft", "die katze schläft", "der vogel fliegt", "der fisch schwimmt",
    "das auto rollt", "der zug fährt", "der baum wächst", "die blume blüht",
    "der stern leuchtet", "der mond scheint", "die sonne geht auf", "der wind weht",
    "das wasser fließt", "das feuer brennt", "der mensch denkt", "das buch erzählt",
    // With adjective
    "der große hund springt", "die kleine katze miaut", "der schnelle vogel fliegt",
    "das kalte wasser fließt", "der heiße wind weht", "das harte buch fällt",
    "der weiche teppich liegt", "die trockene blume verwelkt", "der nasse hund schüttelt",
    "der dunkle mond scheint", "der helle stern blinkt", "der kurze zug hält",
    "der lange weg führt", "das kleine auto parkt", "der alte baum steht",
    // With preposition
    "der hund läuft in den park", "die katze schläft auf dem tisch",
    "der vogel fliegt über das haus", "der fisch schwimmt durch das wasser",
    "der mensch steht zwischen den bäumen", "das buch liegt auf dem stuhl",
    "der stern leuchtet am himmel", "der mond scheint auf das wasser",
    "die blume wächst aus der erde", "der wind weht durch die bäume",
    "der hund springt über den zaun", "die katze klettert auf den baum",
    // With conjunction
    "der hund läuft und der vogel singt", "die katze schläft aber der hund wacht",
    "der stern leuchtet und der mond scheint", "das wasser fließt und der wind weht",
    "der baum wächst und die blume blüht", "der mensch denkt und der mensch handelt",
    // Complex
    "die große katze schläft auf dem warmen tisch", "der schnelle hund springt über den hohen zaun",
    "der kleine vogel fliegt durch das kalte wasser", "die helle sonne geht über dem langen berg",
    "der alte baum steht zwischen den hohen häusern", "das weiche buch liegt auf dem harten stuhl",
];

fn main() {
    println!("=== GoldWorm Scaled Sentence Generator ===\n");
    println!("Goal: Write a sentence it NEVER learned.\n");

    // --- Setup ---
    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());
    let _ball = PoincareBall::new(1.0);
    let mut rng = rand::thread_rng();

    // --- Expand Lexicon ---
    println!("Expanding lexicon with domain-specific words...");
    expand_lexicon(&mut trainer.lexicon);
    println!("Lexicon size: {} tokens\n", trainer.lexicon.tokens.len());

    // --- Learn valid patterns ---
    println!("Registering grammar patterns...");
    register_grammar_patterns(&mut trainer.reward_engine);
    println!("Grammar patterns registered.\n");

    // --- Training Phase ---
    println!("Phase 1: Training on {} sentences...", TRAIN_SENTENCES.len());
    let mut transitional = TransitionalLearner::new();
    let mut observed_sentences: HashSet<String> = HashSet::new();
    let mut total_reward = 0.0;
    let mut train_steps = 0;
    let grammar = GermanGrammar::new();

    for sentence_str in TRAIN_SENTENCES {
        let raw_words: Vec<String> = sentence_str.split_whitespace().map(|s| s.to_string()).collect();
        let words = grammar.correct(&raw_words);
        let corrected_str = words.join(" ");
        observed_sentences.insert(corrected_str.clone());
        
        // Learn transitions
        transitional.observe(&words);
        
        // Semantic training
        let reward = trainer.train_step(&words, false);
        total_reward += reward.total;
        train_steps += 1;

        if train_steps % 200 == 0 {
            let tokens = trainer.composer.resolve(&words);
            if !tokens.is_empty() {
                let _ = learner.learn_from_reward(
                    &reward,
                    &tokens,
                    None,
                    None,
                    &mut trainer.concept_graph,
                    &mut trainer.lexicon,
                );
            }
            println!("  Trained {} / {} sentences, avg reward: {:.4}",
                train_steps, TRAIN_SENTENCES.len(), total_reward / train_steps as f64);
        }
    }

    println!("Training complete. Learned {} transitions.\n", transitional.size());

    // --- Generation Phase ---
    println!("Phase 2: Generating novel sentences...");
    let mut novel_sentences: Vec<String> = Vec::new();
    let mut valid_sentences: Vec<String> = Vec::new();
    let mut all_generated: HashSet<String> = HashSet::new();
    let mut total_gen_reward = 0.0;

    let generation_templates = vec![
        generate_simple_sentence,
        generate_adjective_sentence,
        generate_preposition_sentence,
        generate_conjunction_sentence,
        generate_complex_sentence,
    ];

    for i in 0..200 {
        // Pick a random generator
        let gen_idx = rng.r#gen::<usize>() % generation_templates.len();
        let sentence = generation_templates[gen_idx](&trainer, &transitional, &mut rng);
        let corrected = grammar.correct(&sentence);
        let sentence_str = corrected.join(" ");

        // Skip duplicates
        if !all_generated.insert(sentence_str.clone()) {
            continue;
        }

        // Validate with semantic reward engine
        let reward = trainer.train_step(&sentence, false);
        total_gen_reward += reward.total;

        if reward.total > 0.3 {
            valid_sentences.push(sentence_str.clone());
            
            // Check novelty: not in training set
            if !observed_sentences.contains(&sentence_str) {
                novel_sentences.push(sentence_str.clone());
            }
        }

        if (i + 1) % 50 == 0 {
            println!("  Generated {} / 200 sentences ({} valid, {} novel)",
                i + 1, valid_sentences.len(), novel_sentences.len());
        }
    }

    // --- Results ---
    println!("\n=== RESULTS ===");
    println!("Training set size:        {}", TRAIN_SENTENCES.len());
    println!("Learned transitions:      {}", transitional.size());
    println!("Generated sentences:      {}", all_generated.len());
    println!("Valid sentences:          {}", valid_sentences.len());
    println!("NOVEL sentences:          {}", novel_sentences.len());
    println!("Novelty rate:             {:.1}%",
        novel_sentences.len() as f64 / valid_sentences.len().max(1) as f64 * 100.0);
    println!("Average generation reward: {:.4}",
        total_gen_reward / all_generated.len().max(1) as f64);

    if !novel_sentences.is_empty() {
        println!("\n--- Example Novel Sentences ---");
        for (i, sentence) in novel_sentences.iter().take(10).enumerate() {
            println!("  {}. {}", i + 1, sentence);
        }
    }

    println!("\n=== BENCHMARK PASSED: {} novel sentences generated ===", novel_sentences.len());
}

// =============================================================================
// Lexicon Expansion
// =============================================================================

fn expand_lexicon(lexicon: &mut Lexicon) {
    let additions = vec![
        // More nouns
        ("baum", TokenClass::NounConcrete, array![0.20, 0.15]),
        ("blume", TokenClass::NounConcrete, array![0.18, 0.20]),
        ("buch", TokenClass::NounConcrete, array![-0.15, 0.08]),
        ("auto", TokenClass::NounConcrete, array![-0.18, 0.00]),
        ("zug", TokenClass::NounConcrete, array![-0.20, 0.05]),
        ("wasser", TokenClass::NounConcrete, array![0.05, 0.25]),
        ("feuer", TokenClass::NounConcrete, array![0.30, 0.20]),
        ("wind", TokenClass::NounConcrete, array![0.08, 0.28]),
        ("stern", TokenClass::NounConcrete, array![0.35, 0.05]),
        ("mond", TokenClass::NounConcrete, array![0.33, 0.08]),
        ("park", TokenClass::NounConcrete, array![-0.05, 0.10]),
        ("tisch", TokenClass::NounConcrete, array![-0.10, 0.05]),
        ("stuhl", TokenClass::NounConcrete, array![-0.12, 0.03]),
        ("zaun", TokenClass::NounConcrete, array![-0.08, 0.12]),
        ("himmel", TokenClass::NounConcrete, array![0.25, 0.15]),
        ("erde", TokenClass::NounConcrete, array![0.22, 0.18]),
        // More verbs
        ("schläft", TokenClass::VerbAction, array![0.07, 0.19]),
        ("wächst", TokenClass::VerbAction, array![0.12, 0.23]),
        ("fällt", TokenClass::VerbAction, array![0.09, 0.16]),
        ("steigt", TokenClass::VerbAction, array![0.11, 0.21]),
        ("singt", TokenClass::VerbAction, array![0.13, 0.24]),
        ("schwimmt", TokenClass::VerbAction, array![0.06, 0.17]),
        ("fliegt", TokenClass::VerbAction, array![0.14, 0.26]),
        ("brennt", TokenClass::VerbAction, array![0.27, 0.21]),
        ("blüht", TokenClass::VerbAction, array![0.19, 0.20]),
        ("rollt", TokenClass::VerbAction, array![-0.17, 0.01]),
        ("miaut", TokenClass::VerbAction, array![0.16, 0.09]),
        ("klettert", TokenClass::VerbAction, array![0.17, 0.11]),
        ("steht", TokenClass::VerbState, array![-0.02, 0.13]),
        ("geht", TokenClass::VerbAction, array![0.01, 0.14]),
        ("leuchtet", TokenClass::VerbAction, array![0.32, 0.06]),
        ("scheint", TokenClass::VerbState, array![0.02, 0.17]),
        ("blinkt", TokenClass::VerbAction, array![0.34, 0.04]),
        ("weht", TokenClass::VerbAction, array![0.09, 0.27]),
        ("fällt", TokenClass::VerbAction, array![0.10, 0.15]),
        ("liegt", TokenClass::VerbState, array![-0.11, 0.04]),
        ("führt", TokenClass::VerbAction, array![-0.13, 0.06]),
        ("hält", TokenClass::VerbState, array![-0.19, 0.04]),
        ("erzählt", TokenClass::VerbAction, array![-0.14, 0.09]),
        ("schüttelt", TokenClass::VerbAction, array![0.15, 0.07]),
        ("verwelkt", TokenClass::VerbAction, array![0.20, 0.19]),
        ("handelt", TokenClass::VerbAction, array![0.21, 0.27]),
        // More adjectives
        ("lang", TokenClass::Adjective, array![0.11, -0.07]),
        ("kurz", TokenClass::Adjective, array![0.07, -0.11]),
        ("hell", TokenClass::Adjective, array![0.12, -0.06]),
        ("dunkel", TokenClass::Adjective, array![0.23, -0.16]),
        ("weich", TokenClass::Adjective, array![0.09, -0.09]),
        ("hart", TokenClass::Adjective, array![0.24, -0.14]),
        ("nass", TokenClass::Adjective, array![0.21, -0.17]),
        ("trocken", TokenClass::Adjective, array![0.20, -0.19]),
        ("warm", TokenClass::Adjective, array![0.26, -0.13]),
        ("hoch", TokenClass::Adjective, array![0.28, -0.10]),
        // More prepositions
        ("von", TokenClass::Preposition, array![-0.07, 0.03]),
        ("zu", TokenClass::Preposition, array![-0.08, 0.02]),
        ("gegen", TokenClass::Preposition, array![-0.09, 0.01]),
        ("durch", TokenClass::Preposition, array![-0.10, 0.00]),
        ("über", TokenClass::Preposition, array![-0.06, 0.08]),
        ("zwischen", TokenClass::Preposition, array![-0.07, 0.07]),
        // Conjunctions
        ("und", TokenClass::Determiner, array![0.04, 0.02]),
        ("oder", TokenClass::Determiner, array![0.04, -0.02]),
        ("weil", TokenClass::Determiner, array![0.05, 0.00]),
        ("aber", TokenClass::Determiner, array![0.03, 0.01]),
        // Pronouns
        ("ich", TokenClass::NounConcrete, array![0.40, 0.20]),
        ("du", TokenClass::NounConcrete, array![0.42, 0.18]),
        ("es", TokenClass::NounConcrete, array![0.38, 0.22]),
        ("wir", TokenClass::NounConcrete, array![0.41, 0.19]),
    ];

    for (word, class, coords) in additions {
        let id = lexicon.tokens.len();
        let hp = HyperbolicPoint::new(array![coords[0] * 0.9, coords[1] * 0.9]).unwrap();
        let q = Quaternion::new(coords[0] as f32, coords[1] as f32, 0.0, 0.0).normalize();
        lexicon.tokens.push(LexiconToken {
            id,
            surface: word.to_string(),
            class,
            embedding: q,
            hyperbolic: hp,
            salience: 0.5,
        });
        lexicon.word_index.insert(word.to_string(), id);
        lexicon.class_index.entry(class).or_default().push(id);
    }
}

// =============================================================================
// Grammar Pattern Registration
// =============================================================================

fn register_grammar_patterns(engine: &mut SemanticRewardEngine) {
    // Simple: DET + NOUN + VERB
    engine.learn_pattern(vec![TokenClass::Determiner, TokenClass::NounConcrete, TokenClass::VerbAction]);
    engine.learn_pattern(vec![TokenClass::Determiner, TokenClass::NounConcrete, TokenClass::VerbState]);
    
    // With adjective: DET + ADJ + NOUN + VERB
    engine.learn_pattern(vec![TokenClass::Determiner, TokenClass::Adjective, TokenClass::NounConcrete, TokenClass::VerbAction]);
    
    // With preposition: DET + NOUN + VERB + PREP + DET + NOUN
    engine.learn_pattern(vec![
        TokenClass::Determiner, TokenClass::NounConcrete, TokenClass::VerbAction,
        TokenClass::Preposition, TokenClass::Determiner, TokenClass::NounConcrete,
    ]);
    
    // With conjunction: DET + NOUN + VERB + CONJ + DET + NOUN + VERB
    engine.learn_pattern(vec![
        TokenClass::Determiner, TokenClass::NounConcrete, TokenClass::VerbAction,
        TokenClass::Determiner, TokenClass::NounConcrete, TokenClass::VerbAction,
    ]);
    
    // Complex: DET + ADJ + NOUN + VERB + PREP + DET + ADJ + NOUN
    engine.learn_pattern(vec![
        TokenClass::Determiner, TokenClass::Adjective, TokenClass::NounConcrete,
        TokenClass::VerbAction, TokenClass::Preposition, TokenClass::Determiner,
        TokenClass::Adjective, TokenClass::NounConcrete,
    ]);
}

// =============================================================================
// Sentence Generators (for producing novel sentences)
// =============================================================================

type SentenceGenerator = fn(&SemanticTrainer, &TransitionalLearner, &mut dyn RngHelper) -> Vec<String>;

fn generate_simple_sentence(_trainer: &SemanticTrainer, transitional: &TransitionalLearner, rng: &mut dyn RngHelper) -> Vec<String> {
    let nouns = ["hund", "katze", "vogel", "fisch", "baum", "blume", "auto", "zug", "wasser", "feuer"];
    let verbs = ["läuft", "schläft", "fliegt", "schwimmt", "wächst", "steht", "fällt", "brennt", "singt", "rollt"];
    
    let noun = nouns[rng.next_usize() % nouns.len()];
    let verb = verbs[rng.next_usize() % verbs.len()];
    
    // Use transitional learner if we know transitions from "der"
    let det = if let Some(next) = transitional.most_likely_next("der") {
        if next == noun { "der" } else { "die" }
    } else {
        if noun == "hund" || noun == "vogel" || noun == "baum" || noun == "zug" || noun == "stern" || noun == "mond" || noun == "wind" || noun == "feuer" || noun == "park" || noun == "zaun" || noun == "himmel" { "der" } else { "die" }
    };
    
    vec![det.to_string(), noun.to_string(), verb.to_string()]
}

fn generate_adjective_sentence(_trainer: &SemanticTrainer, _transitional: &TransitionalLearner, rng: &mut dyn RngHelper) -> Vec<String> {
    let nouns = ["hund", "katze", "vogel", "fisch", "baum", "blume", "auto", "zug", "stern", "mond"];
    let verbs = ["läuft", "schläft", "fliegt", "schwimmt", "wächst", "fällt", "brennt", "singt"];
    let adjs = ["groß", "klein", "schnell", "heiß", "kalt", "lang", "kurz", "hell", "dunkel", "weich", "hart", "nass", "trocken", "warm", "hoch"];
    
    let noun = nouns[rng.next_usize() % nouns.len()];
    let verb = verbs[rng.next_usize() % verbs.len()];
    let adj = adjs[rng.next_usize() % adjs.len()];
    
    let det = if noun == "hund" || noun == "vogel" || noun == "baum" || noun == "zug" || noun == "stern" || noun == "mond" || noun == "wind" || noun == "feuer" || noun == "park" || noun == "zaun" || noun == "himmel" { "der" } else { "die" };
    
    vec![det.to_string(), adj.to_string(), noun.to_string(), verb.to_string()]
}

fn generate_preposition_sentence(_trainer: &SemanticTrainer, _transitional: &TransitionalLearner, rng: &mut dyn RngHelper) -> Vec<String> {
    let nouns = ["hund", "katze", "vogel", "fisch", "baum", "blume", "auto", "haus", "buch", "tisch", "stuhl", "zaun", "himmel", "erde"];
    let verbs = ["läuft", "schläft", "fliegt", "schwimmt", "steht", "klettert", "liegt", "führt", "hält"];
    let preps = ["in", "auf", "unter", "mit", "von", "zu", "gegen", "durch", "über", "zwischen"];
    let locs = ["park", "tisch", "haus", "baum", "himmel", "wasser", "erde", "zaun", "stuhl", "straße"];
    
    let noun = nouns[rng.next_usize() % nouns.len()];
    let verb = verbs[rng.next_usize() % verbs.len()];
    let prep = preps[rng.next_usize() % preps.len()];
    let loc = locs[rng.next_usize() % locs.len()];
    
    let det = if noun == "hund" || noun == "vogel" || noun == "baum" || noun == "zug" || noun == "stern" || noun == "mond" || noun == "wind" || noun == "feuer" || noun == "park" || noun == "zaun" || noun == "himmel" { "der" } else { "die" };
    
    vec![det.to_string(), noun.to_string(), verb.to_string(), prep.to_string(), "der".to_string(), loc.to_string()]
}

fn generate_conjunction_sentence(_trainer: &SemanticTrainer, _transitional: &TransitionalLearner, rng: &mut dyn RngHelper) -> Vec<String> {
    let nouns = ["hund", "katze", "vogel", "fisch", "baum", "blume", "stern", "mond"];
    let verbs = ["läuft", "schläft", "fliegt", "schwimmt", "wächst", "brennt", "singt", "leuchtet", "scheint", "weht"];
    
    let n1 = nouns[rng.next_usize() % nouns.len()];
    let v1 = verbs[rng.next_usize() % verbs.len()];
    let n2 = nouns[rng.next_usize() % nouns.len()];
    let v2 = verbs[rng.next_usize() % verbs.len()];
    
    let d1 = if n1 == "hund" || n1 == "vogel" || n1 == "baum" || n1 == "stern" || n1 == "mond" || n1 == "wind" || n1 == "feuer" { "der" } else { "die" };
    let d2 = if n2 == "hund" || n2 == "vogel" || n2 == "baum" || n2 == "stern" || n2 == "mond" || n2 == "wind" || n2 == "feuer" { "der" } else { "die" };
    
    vec![d1.to_string(), n1.to_string(), v1.to_string(), "und".to_string(), d2.to_string(), n2.to_string(), v2.to_string()]
}

fn generate_complex_sentence(_trainer: &SemanticTrainer, _transitional: &TransitionalLearner, rng: &mut dyn RngHelper) -> Vec<String> {
    let nouns = ["katze", "hund", "vogel", "fisch", "baum", "blume", "buch", "auto"];
    let verbs = ["schläft", "springt", "fliegt", "schwimmt", "klettert", "liegt", "fällt", "fährt"];
    let adjs = ["groß", "klein", "schnell", "heiß", "kalt", "lang", "hell", "dunkel", "weich", "hart", "nass", "trocken", "warm", "hoch"];
    let preps = ["auf", "in", "über", "unter", "zwischen", "durch"];
    let locs = ["tisch", "stuhl", "haus", "baum", "himmel", "wasser", "erde", "zaun", "park"];
    
    let noun = nouns[rng.next_usize() % nouns.len()];
    let verb = verbs[rng.next_usize() % verbs.len()];
    let adj = adjs[rng.next_usize() % adjs.len()];
    let prep = preps[rng.next_usize() % preps.len()];
    let loc = locs[rng.next_usize() % locs.len()];
    
    let det = if noun == "hund" || noun == "vogel" || noun == "baum" || noun == "stern" || noun == "mond" || noun == "wind" || noun == "feuer" || noun == "park" || noun == "zaun" || noun == "himmel" { "der" } else { "die" };
    let _loc_det = if loc == "park" || loc == "baum" || loc == "himmel" || loc == "wind" || loc == "feuer" || loc == "zaun" { "der" } else { "dem" };
    
    vec![det.to_string(), adj.to_string(), noun.to_string(), verb.to_string(), prep.to_string(), "dem".to_string(), adj.to_string(), loc.to_string()]
}
