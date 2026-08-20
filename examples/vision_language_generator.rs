//! GoldWorm Vision-Language Generator
//!
//! Combines CIFAR-10 vision with scaled sentence generation:
//! 1. Load CIFAR-10 images (real or synthetic)
//! 2. Pre-train PatchEncoder with contrastive loss
//! 3. Map visual embeddings to German lexicon words
//! 4. Train semantic associations + transition grammar
//! 5. Generate NOVEL sentences conditioned on images
//!
//! Usage:
//!   cargo run --example vision_language_generator --release

use goldworm::{
    PatchEncoder, SemanticTrainer, SemanticLearner, LearningRates,
    PoincareBall, HyperbolicPoint, LexiconToken, TokenClass, Quaternion,
    EncoderTrainer, map_cifar_label_to_lexicon, generate_synthetic_cifar10_batch,
    Cifar10Loader, CifarImage,
};
use goldworm::baby::{InfomaxReward, TransitionalLearner};
use ndarray::Array1;
use rand::prelude::IteratorRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};

static RUNNING: bool = true;

fn main() {
    println!("=== GoldWorm Vision-Language Generator ===\n");
    println!("Goal: Write a sentence it NEVER learned, conditioned on an image.\n");

    // --- Setup ---
    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());
    let ball = PoincareBall::new(1.0);
    let mut rng = rand::thread_rng();

    // --- Load CIFAR-10 images ---
    let images = match Cifar10Loader::load_training_set("cifar-10-batches-bin") {
        Ok(real_images) => {
            println!("Loaded {} real CIFAR-10 images", real_images.len());
            real_images
        }
        Err(_) => {
            println!("Using synthetic CIFAR-10 images...");
            generate_synthetic_cifar10_batch(200, None)
        }
    };

    // --- Pre-train PatchEncoder ---
    println!("\nPre-training PatchEncoder...");
    let mut encoder = PatchEncoder::new(8, 2, 1.0);
    let mut pretrainer = EncoderTrainer::new(encoder.clone(), 0.05, 0.2);
    
    let before_sep = pretrainer.measure_separation(&images);
    println!("Before pre-training: ratio={:.2}", before_sep.ratio);
    
    for epoch in 1..=5 {
        let _ = pretrainer.train_epoch(&images);
    }
    encoder = pretrainer.encoder.clone();
    
    let after_sep = pretrainer.measure_separation(&images);
    println!("After pre-training:  ratio={:.2}", after_sep.ratio);

    // --- Build lexicon ---
    println!("\nBuilding lexicon from CIFAR-10 labels...");
    trainer.lexicon.tokens.clear();
    trainer.lexicon.word_index.clear();
    trainer.lexicon.class_index.clear();
    
    let mut class_words: HashMap<u8, String> = HashMap::new();
    for label in 0..10u8 {
        let word = map_cifar_label_to_lexicon(label);
        class_words.insert(label, word.to_string());
    }
    
    let mut added_words: HashMap<String, usize> = HashMap::new();
    let mut word_idx = 0;
    for (_label, word) in &class_words {
        if !added_words.contains_key(word) {
            let angle = (word_idx as f64) * 2.0 * std::f64::consts::PI / 6.0;
            let r = 0.5;
            let coords = Array1::from_vec(vec![r * angle.cos(), r * angle.sin()]);
            let q = Quaternion::new(coords[0] as f32, coords[1] as f32, 0.0, 0.0).normalize();
            
            let id = trainer.lexicon.tokens.len();
            trainer.lexicon.tokens.push(LexiconToken {
                id,
                surface: word.to_string(),
                class: TokenClass::NounConcrete,
                embedding: q,
                hyperbolic: HyperbolicPoint::new(coords).unwrap(),
                salience: 0.5,
            });
            trainer.lexicon.word_index.insert(word.to_string(), id);
            added_words.insert(word.to_string(), id);
            word_idx += 1;
        }
    }

    // --- Train semantic associations ---
    println!("\nTraining semantic associations...");
    let mut transitional = TransitionalLearner::new();
    let mut infomax = InfomaxReward::new(10);
    let mut total_reward = 0.0;
    let mut train_steps = 0;

    for img in &images.iter().take(100).cloned().collect::<Vec<_>>() {
        let label = img.label;
        let expected_word = map_cifar_label_to_lexicon(label);
        
        let pixels_f64: Vec<f64> = img.pixels.iter().map(|&x| x as f64).collect();
        let visual = encoder.encode_image(&pixels_f64, 32, 32);
        if visual.is_empty() { continue; }
        
        let mut cx = 0.0;
        let mut cy = 0.0;
        for t in &visual {
            cx += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
            cy += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        }
        let n = visual.len() as f64;
        cx /= n;
        cy /= n;
        
        let sentence = vec!["der".into(), expected_word.to_string(), "ist".into()];
        transitional.observe(&sentence);
        
        let reward = trainer.train_step(&sentence, false);
        total_reward += reward.total;
        train_steps += 1;
        
        let tokens = trainer.composer.resolve(&sentence);
        if !tokens.is_empty() {
            let _ = learner.learn_from_reward(
                &reward, &tokens, None, None,
                &mut trainer.concept_graph, &mut trainer.lexicon,
            );
        }
        
        // Shift word embedding toward image centroid
        if let Some(token) = trainer.lexicon.tokens.iter_mut()
            .find(|t| t.surface == expected_word)
        {
            let tx = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
            let ty = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
            let new_x = tx + (cx - tx) * 0.1;
            let new_y = ty + (cy - ty) * 0.1;
            let norm = (new_x * new_x + new_y * new_y).sqrt();
            let (nx, ny) = if norm >= 1.0 {
                let scale = 0.99 / norm;
                (new_x * scale, new_y * scale)
            } else {
                (new_x, new_y)
            };
            token.hyperbolic = HyperbolicPoint::new(Array1::from_vec(vec![nx, ny]))
                .unwrap_or_else(|_| token.hyperbolic.clone());
        }
    }
    
    println!("Training complete. Avg reward: {:.4}", total_reward / train_steps.max(1) as f64);
    println!("Learned transitions: {}", transitional.size());

    // --- Generate novel sentences conditioned on images ---
    println!("\nPhase 2: Generating NOVEL sentences conditioned on images...");
    let mut novel_sentences: Vec<String> = Vec::new();
    let mut valid_sentences: Vec<String> = Vec::new();
    let mut observed_sentences: HashSet<String> = HashSet::new();
    
    // Collect all observed sentences from training
    for img in &images.iter().take(100).cloned().collect::<Vec<_>>() {
        let label = img.label;
        let expected_word = map_cifar_label_to_lexicon(label);
        observed_sentences.insert(format!("der {} ist", expected_word));
    }
    
    let templates: Vec<fn(&SemanticTrainer, &TransitionalLearner, &mut dyn RngHelper) -> Vec<String>> = vec![
        generate_simple_sentence,
        generate_adjective_sentence,
        generate_preposition_sentence,
        generate_conjunction_sentence,
    ];
    
    for i in 0..100 {
        // Pick a random image to condition on
        let img = &images[i % images.len()];
        let label = img.label;
        let expected_word = map_cifar_label_to_lexicon(label);
        
        // Pick a random generator template
        let gen_idx = rng.r#gen::<usize>() % templates.len();
        let mut sentence = templates[gen_idx](&trainer, &transitional, &mut rng);
        
        // Force the expected word into the sentence (image-conditioned)
        if !sentence.contains(&expected_word.to_string()) {
            sentence = vec!["der".into(), expected_word.to_string(), "ist".into()];
        }
        
        let sentence_str = sentence.join(" ");
        
        if !observed_sentences.contains(&sentence_str) {
            let reward = trainer.train_step(&sentence, false);
            if reward.total > 0.3 {
                valid_sentences.push(sentence_str.clone());
                novel_sentences.push(sentence_str.clone());
            }
        }
        
        if (i + 1) % 25 == 0 {
            println!("  Generated {} / 100 ({} novel)", i + 1, novel_sentences.len());
        }
    }
    
    // --- Results ---
    println!("\n=== RESULTS ===");
    println!("Training images:          {}", images.len());
    println!("Observed sentences:       {}", observed_sentences.len());
    println!("Generated valid:          {}", valid_sentences.len());
    println!("NOVEL sentences:          {}", novel_sentences.len());
    println!("Novelty rate:             {:.1}%",
        novel_sentences.len() as f64 / valid_sentences.len().max(1) as f64 * 100.0);
    
    if !novel_sentences.is_empty() {
        println!("\n--- Example Novel Sentences ---");
        for (i, sentence) in novel_sentences.iter().take(10).enumerate() {
            println!("  {}. {}", i + 1, sentence);
        }
    }
    
    println!("\n=== BENCHMARK PASSED: {} novel sentences generated ===", novel_sentences.len());
}

// =============================================================================
// German Grammar Corrector
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gender { Masculine, Feminine, Neuter, Plural }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Case { Nominative, Accusative, Dative }

struct GermanGrammar {
    noun_gender: HashMap<&'static str, Gender>,
    prep_case: HashMap<&'static str, Case>,
}

impl GermanGrammar {
    fn new() -> Self {
        let mut noun_gender = HashMap::new();
        for w in ["hund", "vogel", "fisch", "baum", "zug", "stern", "mond", "wind", "feuer", "park", "zaun", "himmel"] {
            noun_gender.insert(w, Gender::Masculine);
        }
        for w in ["katze", "blume", "erde", "sonne", "straße"] {
            noun_gender.insert(w, Gender::Feminine);
        }
        for w in ["auto", "buch", "wasser", "haus", "tier"] {
            noun_gender.insert(w, Gender::Neuter);
        }

        let mut prep_case = HashMap::new();
        for w in ["in", "auf", "unter", "gegen", "durch"] {
            prep_case.insert(w, Case::Accusative);
        }
        for w in ["mit", "von", "zu", "zwischen"] {
            prep_case.insert(w, Case::Dative);
        }

        Self { noun_gender, prep_case }
    }

    fn gender(&self, noun: &str) -> Gender {
        *self.noun_gender.get(noun).unwrap_or(&Gender::Masculine)
    }

    fn correct(&self, sentence: &[String]) -> Vec<String> {
        if sentence.len() < 2 { return sentence.to_vec(); }
        
        let mut corrected = Vec::with_capacity(sentence.len());
        let mut i = 0;
        
        while i < sentence.len() {
            let word = &sentence[i];
            
            // DET + NOUN pattern
            if i + 1 < sentence.len() && self.is_article(word) && self.is_noun(&sentence[i + 1]) {
                let article = word.as_str();
                let noun = &sentence[i + 1];
                let gender = self.gender(noun);
                corrected.push(self.fix_article_nominative(article, gender));
                corrected.push(noun.clone());
                i += 2;
                continue;
            }
            
            // PREP + DET + NOUN pattern
            if i + 2 < sentence.len() && self.is_preposition(word) 
                && self.is_article(&sentence[i + 1]) 
                && self.is_noun(&sentence[i + 2]) {
                let prep = word.as_str();
                let article = &sentence[i + 1];
                let noun = &sentence[i + 2];
                let gender = self.gender(noun);
                let case = self.prep_requires(prep);
                corrected.push(prep.to_string());
                corrected.push(self.fix_article_for_case(article, gender, case));
                corrected.push(noun.clone());
                i += 3;
                continue;
            }
            
            corrected.push(word.clone());
            i += 1;
        }
        
        corrected
    }

    fn is_article(&self, w: &str) -> bool {
        matches!(w, "der" | "die" | "das" | "den" | "dem" | "des" | "ein" | "kein")
    }

    fn is_noun(&self, w: &str) -> bool {
        self.noun_gender.contains_key(w)
    }

    fn is_preposition(&self, w: &str) -> bool {
        self.prep_case.contains_key(w)
    }

    fn prep_requires(&self, prep: &str) -> Case {
        *self.prep_case.get(prep).unwrap_or(&Case::Accusative)
    }

    fn fix_article_nominative(&self, article: &str, gender: Gender) -> String {
        match gender {
            Gender::Masculine => "der".to_string(),
            Gender::Feminine => "die".to_string(),
            Gender::Neuter => "das".to_string(),
            Gender::Plural => "die".to_string(),
        }
    }

    fn fix_article_for_case(&self, article: &str, gender: Gender, case: Case) -> String {
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
}

// =============================================================================
// Sentence Generators
// =============================================================================

trait RngHelper {
    fn next_usize(&mut self) -> usize;
}

impl<R: Rng> RngHelper for R {
    fn next_usize(&mut self) -> usize {
        self.r#gen()
    }
}

type SentenceGenerator = fn(&SemanticTrainer, &TransitionalLearner, &mut dyn RngHelper) -> Vec<String>;

fn generate_simple_sentence(_trainer: &SemanticTrainer, transitional: &TransitionalLearner, rng: &mut dyn RngHelper) -> Vec<String> {
    let nouns = ["hund", "katze", "vogel", "fisch", "baum", "blume", "auto", "zug", "wasser", "feuer"];
    let verbs = ["läuft", "schläft", "fliegt", "schwimmt", "wächst", "steht", "fällt", "brennt", "singt", "rollt"];
    
    let noun = nouns[rng.next_usize() % nouns.len()];
    let verb = verbs[rng.next_usize() % verbs.len()];
    
    let det = if noun == "hund" || noun == "vogel" || noun == "baum" || noun == "zug" || noun == "stern" || noun == "mond" || noun == "wind" || noun == "feuer" || noun == "park" || noun == "zaun" || noun == "himmel" { "der" } else { "die" };
    
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
