//! GoldSnnail Chat Benchmark Suite
//!
//! Measures SNN-LLM chat engine performance across 6 dimensions:
//! 1. SNN Encoding Efficiency
//! 2. Response Quality
//! 3. Learning Speed
//! 4. Reasoning Quality
//! 5. Online Learning
//! 6. World Model Prediction
//!
//! Usage:
//!   cargo run --example benchmark --release

use goldsnnail::{
    LexiconToken, TokenClass, SemanticTrainer, TransitionalLearner,
    TokenSpikeEncoder, SpikeTokenDecoder, ConversationBuffer, ConversationTurn,
    ReasoningEngine, OnlineLearner, WorldChat, SpikeBuffer, ChatArena,
    HyperbolicPoint, Quaternion, WorldGeometry, build_extended_lexicon,
    PowerLawObserver, AvalancheGuidedSelector, build_response_from_selection,
};
use ndarray::array;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

fn main() {
    println!("=== GoldSnnail Chat Benchmark Suite ===\n");

    let mut results = HashMap::new();

    // Suite 1: SNN Encoding Efficiency
    let snn_results = bench_snn_encoding();
    results.insert("snn_encoding", snn_results);

    // Suite 2: Response Quality
    let response_results = bench_response_quality();
    results.insert("response_quality", response_results);

    // Suite 3: Learning Speed
    let learning_results = bench_learning_speed();
    results.insert("learning_speed", learning_results);

    // Suite 4: Reasoning Quality
    let reasoning_results = bench_reasoning();
    results.insert("reasoning", reasoning_results);

    // Suite 5: Online Learning
    let online_results = bench_online_learning();
    results.insert("online_learning", online_results);

    // Suite 6: World Model
    let world_results = bench_world_model();
    results.insert("world_model", world_results);

    // Suite 7: Criticality
    let criticality_results = bench_criticality();
    results.insert("criticality", criticality_results);

    // Suite 8: Avalanche-Guided
    let avalanche_results = bench_avalanche_guided();
    results.insert("avalanche_guided", avalanche_results);

// Suite 9: Comparison
    let comparison_results = bench_template_vs_avalanche();
    results.insert("template_vs_avalanche", comparison_results);

    // Export results
    export_results(&results);

    println!("\n=== Benchmark Complete ===");
    println!("Results exported to docs/src/development/benchmark_results.json");
}

// =============================================================================
// Suite 1: SNN Encoding Efficiency
// =============================================================================

fn bench_snn_encoding() -> HashMap<String, f64> {
    println!("[Suite 1] SNN Encoding Efficiency...");
    let mut rng = rand::thread_rng();

    let mut trainer = SemanticTrainer::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);

    // Build 20-word lexicon
    let words = ["hallo", "ich", "bin", "goldsnnail", "der", "die", "das", "hund", "katze", "vogel",
                 "läuft", "springt", "ist", "gut", "schlecht", "danke", "bitte", "wie", "was", "stern"];
    for (i, &word) in words.iter().enumerate() {
        let id = trainer.lexicon.tokens.len();
        let coords = array![(i as f64 * 0.04) % 0.9, ((i as f64 * 0.07) % 0.9)];
        let hp = HyperbolicPoint::new(array![coords[0] * 0.9, coords[1] * 0.9]).unwrap();
        let q = Quaternion::new(coords[0] as f32, coords[1] as f32, 0.0, 0.0).normalize();
        trainer.lexicon.tokens.push(LexiconToken {
            id,
            surface: word.to_string(),
            class: TokenClass::NounConcrete,
            embedding: q,
            hyperbolic: hp,
            salience: 0.5,
        });
        trainer.lexicon.word_index.insert(word.to_string(), id);
    }
    encoder.register_lexicon(&trainer.lexicon);
    decoder.register_lexicon(&trainer.lexicon);

    let mut total_spikes = 0usize;
    let mut total_words = 0usize;
    let mut roundtrip_hits = 0usize;
    let mut total_encode_time_us = 0u64;
    let num_sentences = 50;

    for _ in 0..num_sentences {
        let len = 3 + (rng.r#gen::<usize>() % 6);
        let sentence: Vec<String> = (0..len)
            .map(|_| words[rng.r#gen::<usize>() % words.len()].to_string())
            .collect();

        let start = Instant::now();
        let spikes = encoder.encode_sentence(&sentence);
        let elapsed = start.elapsed().as_micros() as u64;
        total_encode_time_us += elapsed;

        total_spikes += spikes.len();
        total_words += sentence.len();

        // Roundtrip check
        let decoded = decoder.decode_spikes(&SpikeBuffer {
            indices: spikes.iter().map(|e| e.src.0 as u32).collect(),
            count: spikes.len(),
        });
        let decoded_set: std::collections::HashSet<_> = decoded.iter().collect();
        for w in &sentence {
            if decoded_set.contains(w) {
                roundtrip_hits += 1;
            }
        }
    }

    let mut results = HashMap::new();
    results.insert("avg_spikes_per_word".to_string(), total_spikes as f64 / total_words.max(1) as f64);
    results.insert("roundtrip_accuracy".to_string(), roundtrip_hits as f64 / total_words.max(1) as f64);
    results.insert("avg_encode_time_us".to_string(), total_encode_time_us as f64 / num_sentences as f64);

    println!("  Avg spikes/word: {:.2}", results["avg_spikes_per_word"]);
    println!("  Roundtrip accuracy: {:.1}%", results["roundtrip_accuracy"] * 100.0);
    println!("  Avg encode time: {:.0} us\n", results["avg_encode_time_us"]);

    results
}

// =============================================================================
// Suite 2: Response Quality
// =============================================================================

fn bench_response_quality() -> HashMap<String, f64> {
    println!("[Suite 2] Response Quality...");
    let mut rng = rand::thread_rng();

    let mut trainer = SemanticTrainer::new(1.0);
    let mut transitional = TransitionalLearner::new();
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    let mut conv = ConversationBuffer::new(50);

    setup_lexicon(&mut trainer, &mut encoder, &mut decoder);

    let mut chat_arena = ChatArena::new();
    let trainer_idx = chat_arena.push(trainer, encoder, decoder);

    let inputs = [
        "hallo", "hi", "wie gehts", "was ist das", "wer bist du",
        "danke", "bitte", "gut", "schlecht", "ja",
        "nein", "warum", "wo", "wann", "wie",
    ];

    let templates = vec![
        vec!["ich", "bin", "goldsnnail"],
        vec!["der", "hund", "läuft"],
        vec!["die", "katze", "schläft"],
        vec!["der", "vogel", "fliegt"],
        vec!["der", "stern", "scheint"],
    ];

    let mut total_length = 0usize;
    let mut total_vocab_hits = 0usize;
    let mut total_words = 0usize;
    let mut novel_responses = 0usize;
    let mut grammatical = 0usize;

    for input in &inputs {
        let input_vec: Vec<String> = input.split_whitespace().map(|s| s.to_string()).collect();

        let trainer = &mut chat_arena.trainers[trainer_idx];
        let encoder = &mut chat_arena.encoders[trainer_idx];
        let decoder = &mut chat_arena.decoders[trainer_idx];

        let response = generate_response(&input_vec, trainer, &transitional, encoder, decoder, &conv);
        let response_text: Vec<String> = response.clone();
        let len = response_text.len();
        total_length += len;

        for w in &response_text {
            total_words += 1;
            if chat_arena.encoders[trainer_idx].neuron_for_word(w).is_some() {
                total_vocab_hits += 1;
            }
        }

        // Novelty: not matching a template exactly
        let is_template = templates.iter().any(|t| t == &response_text.as_slice());
        if !is_template && len > 0 {
            novel_responses += 1;
        }

        // Grammatical: DET NOUN VERB or DET NOUN ADJ VERB
        if len >= 3 {
            let dets = ["der", "die", "das"];
            let verbs = ["läuft", "springt", "ist", "sieht", "schläft", "fliegt", "scheint", "wächst", "blüht", "fließt"];
            if dets.contains(&response_text[0].as_str()) && verbs.contains(&response_text[len-1].as_str()) {
                grammatical += 1;
            }
        }

        conv.push(ConversationTurn::new_user(input.to_string()));
        conv.push(ConversationTurn::new_assistant(response_text.join(" "), response_text));
    }

    let n = inputs.len() as f64;
    let mut results = HashMap::new();
    results.insert("avg_length".to_string(), total_length as f64 / n);
    results.insert("vocab_coverage".to_string(), total_vocab_hits as f64 / total_words.max(1) as f64);
    results.insert("novelty_rate".to_string(), novel_responses as f64 / n);
    results.insert("grammatical_rate".to_string(), grammatical as f64 / n);

    println!("  Avg response length: {:.1} words", results["avg_length"]);
    println!("  Vocab coverage: {:.1}%", results["vocab_coverage"] * 100.0);
    println!("  Novelty rate: {:.1}%", results["novelty_rate"] * 100.0);
    println!("  Grammatical rate: {:.1}%\n", results["grammatical_rate"] * 100.0);

    results
}

// =============================================================================
// Suite 3: Learning Speed
// =============================================================================

fn bench_learning_speed() -> HashMap<String, f64> {
    println!("[Suite 3] Learning Speed...");
    let mut rng = rand::thread_rng();

    let mut trainer = SemanticTrainer::new(1.0);
    let mut transitional = TransitionalLearner::new();
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    let mut conv = ConversationBuffer::new(50);

    setup_lexicon(&mut trainer, &mut encoder, &mut decoder);

    let mut chat_arena = ChatArena::new();
    let trainer_idx = chat_arena.push(trainer, encoder, decoder);

    let sentences = vec![
        vec!["der", "hund", "läuft"],
        vec!["die", "katze", "schläft"],
        vec!["der", "vogel", "fliegt"],
        vec!["der", "stern", "scheint"],
        vec!["der", "baum", "wächst"],
    ];

    let initial_transitions = transitional.size();
    let initial_concepts = chat_arena.trainers[trainer_idx].concept_graph.nodes.len();
    let initial_lexicon = chat_arena.trainers[trainer_idx].lexicon.tokens.len();

    for turn in 0..30 {
        let sentence = &sentences[turn % sentences.len()];
        let sentence_strings: Vec<String> = sentence.iter().map(|s| s.to_string()).collect();

        let trainer = &mut chat_arena.trainers[trainer_idx];
        let encoder = &mut chat_arena.encoders[trainer_idx];
        let decoder = &mut chat_arena.decoders[trainer_idx];

        let response = generate_response(&sentence_strings, trainer, &transitional, encoder, decoder, &conv);

        if !response.is_empty() {
            transitional.observe(&sentence_strings);
            let _ = chat_arena.trainers[trainer_idx].train_step(&sentence_strings, false);
        }

        conv.push(ConversationTurn::new_user(sentence_strings.join(" ")));
        conv.push(ConversationTurn::new_assistant(response.join(" "), response));
    }

    let final_transitions = transitional.size();
    let final_concepts = chat_arena.trainers[trainer_idx].concept_graph.nodes.len();
    let final_lexicon = chat_arena.trainers[trainer_idx].lexicon.tokens.len();

    let mut results = HashMap::new();
    results.insert("final_transitions".to_string(), final_transitions as f64);
    results.insert("final_concepts".to_string(), final_concepts as f64);
    results.insert("final_lexicon".to_string(), final_lexicon as f64);
    results.insert("transition_growth".to_string(), (final_transitions - initial_transitions) as f64);
    results.insert("concept_growth".to_string(), (final_concepts - initial_concepts) as f64);
    results.insert("lexicon_growth".to_string(), (final_lexicon - initial_lexicon) as f64);

    println!("  Transitions: {} -> {} (+{})", initial_transitions, final_transitions, final_transitions - initial_transitions);
    println!("  Concepts: {} -> {} (+{})", initial_concepts, final_concepts, final_concepts - initial_concepts);
    println!("  Lexicon: {} -> {} (+{})\n", initial_lexicon, final_lexicon, final_lexicon - initial_lexicon);

    results
}

// =============================================================================
// Suite 4: Reasoning Quality
// =============================================================================

fn bench_reasoning() -> HashMap<String, f64> {
    println!("[Suite 4] Reasoning Quality...");
    let mut rng = rand::thread_rng();

    let mut trainer = SemanticTrainer::new(1.0);
    let mut transitional = TransitionalLearner::new();
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    let mut conv = ConversationBuffer::new(50);

    setup_lexicon(&mut trainer, &mut encoder, &mut decoder);

    let mut chat_arena = ChatArena::new();
    let trainer_idx = chat_arena.push(trainer, encoder, decoder);

    let queries = [
        "hallo goldsnnail",
        "wie ist das wetter",
        "was denkst du über hunde",
        "erzähl mir etwas",
        "wer bist du",
        "danke für die hilfe",
        "warum läuft der hund",
        "wo ist die katze",
        "wann kommt der vogel",
        "wie geht es dir",
    ];

    let mut total_chain_len = 0usize;
    let mut total_confidence = 0.0;
    let mut total_spikes = 0usize;
    let mut synthesis_count = 0usize;

    for query in &queries {
        conv.push(ConversationTurn::new_user(query.to_string()));

        let trainer = &mut chat_arena.trainers[trainer_idx];
        let encoder = &mut chat_arena.encoders[trainer_idx];

        let mut reasoner = ReasoningEngine::new(trainer, &mut transitional, encoder);
        let chain = reasoner.reason(query, &conv);

        total_chain_len += chain.len();
        for thought in &chain.thoughts {
            total_confidence += thought.confidence;
            total_spikes += thought.spikes;
        }
        if chain.len() >= 3 {
            synthesis_count += 1;
        }
    }

    let n = queries.len() as f64;
    let mut results = HashMap::new();
    results.insert("avg_chain_length".to_string(), total_chain_len as f64 / n);
    results.insert("avg_confidence".to_string(), total_confidence / total_chain_len.max(1) as f64);
    results.insert("avg_spikes_per_chain".to_string(), total_spikes as f64 / n);
    results.insert("synthesis_rate".to_string(), synthesis_count as f64 / n);

    println!("  Avg chain length: {:.1}", results["avg_chain_length"]);
    println!("  Avg confidence: {:.3}", results["avg_confidence"]);
    println!("  Avg spikes/chain: {:.0}", results["avg_spikes_per_chain"]);
    println!("  Synthesis rate: {:.1}%\n", results["synthesis_rate"] * 100.0);

    results
}

// =============================================================================
// Suite 5: Online Learning
// =============================================================================

fn bench_online_learning() -> HashMap<String, f64> {
    println!("[Suite 5] Online Learning...");
    let mut rng = rand::thread_rng();

    let mut trainer = SemanticTrainer::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);

    // Small seed lexicon
    let seed_words = ["hund", "katze", "vogel", "laufen", "schlafen"];
    for (i, &word) in seed_words.iter().enumerate() {
        let id = trainer.lexicon.tokens.len();
        let coords = array![(i as f64 * 0.1) % 0.9, ((i as f64 * 0.15) % 0.9)];
        let hp = HyperbolicPoint::new(array![coords[0] * 0.9, coords[1] * 0.9]).unwrap();
        let q = Quaternion::new(coords[0] as f32, coords[1] as f32, 0.0, 0.0).normalize();
        trainer.lexicon.tokens.push(LexiconToken {
            id,
            surface: word.to_string(),
            class: TokenClass::NounConcrete,
            embedding: q,
            hyperbolic: hp,
            salience: 0.5,
        });
        trainer.lexicon.word_index.insert(word.to_string(), id);
    }
    // Pre-populate concept graph
    for token in &trainer.lexicon.tokens.clone() {
        trainer.concept_graph.add_concept(&token.surface, token.hyperbolic.clone());
    }
    encoder.register_lexicon(&trainer.lexicon);
    decoder.register_lexicon(&trainer.lexicon);

    let mut chat_arena = ChatArena::new();
    let trainer_idx = chat_arena.push(trainer, encoder, decoder);
    let mut online_learner = OnlineLearner::new(trainer_idx, trainer_idx, trainer_idx);
    let mut conv = ConversationBuffer::new(50);

    let unknown_words = ["banane", "auto", "haus", "baum", "wolke", "blume", "wasser", "feuer", "erde", "luft",
                         "licht", "ton", "farbe", "form", "bewegung"];
    let initial_size = chat_arena.trainers[trainer_idx].lexicon.tokens.len();

    let mut learned_count = 0usize;
    for word in &unknown_words {
        conv.push(ConversationTurn::new_user(format!("hund {} katze", word)));
        let learned = online_learner.process_unknown(&mut chat_arena, &[word.to_string()], &conv).unwrap_or_default();
        learned_count += learned.len();
    }

    let final_size = chat_arena.trainers[trainer_idx].lexicon.tokens.len();

    let mut results = HashMap::new();
    results.insert("words_presented".to_string(), unknown_words.len() as f64);
    results.insert("words_learned".to_string(), learned_count as f64);
    results.insert("learn_rate".to_string(), learned_count as f64 / unknown_words.len() as f64);
    results.insert("lexicon_expansion".to_string(), (final_size - initial_size) as f64);

    println!("  Presented: {}", unknown_words.len());
    println!("  Learned: {}", learned_count);
    println!("  Learn rate: {:.1}%", results["learn_rate"] * 100.0);
    println!("  Lexicon expansion: +{}\n", final_size - initial_size);

    results
}

// =============================================================================
// Suite 6: World Model Prediction
// =============================================================================

fn bench_world_model() -> HashMap<String, f64> {
    println!("[Suite 6] World Model Prediction...");
    let mut rng = rand::thread_rng();

    let mut trainer = SemanticTrainer::new(1.0);
    let mut transitional = TransitionalLearner::new();
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    let mut conv = ConversationBuffer::new(50);

    setup_lexicon(&mut trainer, &mut encoder, &mut decoder);

    let mut chat_arena = ChatArena::new();
    let trainer_idx = chat_arena.push(trainer, encoder, decoder);

    let geom = WorldGeometry::new(2, 4, 1.0);
    geom.validate(chat_arena.trainers[trainer_idx].lexicon.tokens[0].hyperbolic.coords.len()).expect("WorldGeometry dimension must match lexicon");
    let mut world_chat = WorldChat::from_config(geom);

    let test_sentences = [
        vec!["hallo".to_string(), "goldsnnail".to_string()],
        vec!["der".to_string(), "hund".to_string(), "läuft".to_string()],
        vec!["die".to_string(), "katze".to_string()],
        vec!["wie".to_string(), "geht".to_string(), "es".to_string()],
        vec!["der".to_string(), "vogel".to_string(), "fliegt".to_string()],
    ];

    let mut stable_count = 0usize;
    let mut match_count = 0usize;
    let total = test_sentences.len();

    for sentence in &test_sentences {
        let trainer = &chat_arena.trainers[trainer_idx];
        let encoder = &chat_arena.encoders[trainer_idx];

        if let Some(predicted) = world_chat.predict_response_word(trainer, sentence) {
            if encoder.neuron_for_word(&predicted).is_some() {
                match_count += 1;
            }
        }

        // Check stability: encode state, predict, verify inside ball
        let state = world_chat.encode_sentence_state(trainer, sentence);
        if let Ok(predicted) = world_chat.predict_next(&state) {
            if predicted.euclidean_norm() < 1.0 {
                stable_count += 1;
            }
        }
    }

    let mut results = HashMap::new();
    results.insert("stability_rate".to_string(), stable_count as f64 / total as f64);
    results.insert("token_match_rate".to_string(), match_count as f64 / total as f64);

    println!("  Stability rate: {:.1}%", results["stability_rate"] * 100.0);
    println!("  Token match rate: {:.1}%\n", results["token_match_rate"] * 100.0);

    results
}

// =============================================================================
// Suite 7: Criticality / Power-Law
// =============================================================================

fn bench_criticality() -> HashMap<String, f64> {
    println!("[Suite 7] Criticality (Power-Law)...");
    let mut rng = rand::thread_rng();

    let mut trainer = SemanticTrainer::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);

    build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);

    trainer.concept_graph.add_self_connections();
    trainer.concept_graph.add_preferential_attachment(800);

    let mut observer = PowerLawObserver::new(1000);
    observer.record_graph_avalanches(&trainer.concept_graph, 1000);

    let test_sentences = vec![
        vec!["hallo".to_string(), "ich".to_string(), "bin".to_string(), "goldsnnail".to_string()],
        vec!["hello".to_string(), "I".to_string(), "am".to_string(), "goldsnnail".to_string()],
        vec!["der".to_string(), "hund".to_string(), "läuft".to_string()],
        vec!["the".to_string(), "dog".to_string(), "run".to_string()],
        vec!["die".to_string(), "katze".to_string(), "schläft".to_string()],
        vec!["the".to_string(), "cat".to_string(), "sleep".to_string()],
        vec!["hallo".to_string(), "wie".to_string(), "geht".to_string()],
        vec!["hello".to_string(), "how".to_string(), "are".to_string(), "you".to_string()],
        vec!["danke".to_string(), "und".to_string(), "bitte".to_string()],
        vec!["thanks".to_string(), "and".to_string(), "please".to_string()],
    ];

    for sentence in &test_sentences {
        let spikes = encoder.encode_sentence(sentence);
        let mut raster = vec![0u8; 64];
        for spike in &spikes {
            let idx = spike.src.0 % raster.len();
            raster[idx] = raster[idx].saturating_add(1);
        }
        observer.record_raster(&raster);
    }

    let is_critical = observer.is_critical();
    let tau = observer.fit().map(|f| f.tau).unwrap_or(0.0);
    let r2 = observer.fit().map(|f| f.r_squared).unwrap_or(0.0);

    println!("  Tau: {:.3}, R²: {:.3}, Critical: {}", tau, r2, is_critical);

    let mut results = HashMap::new();
    results.insert("tau".to_string(), tau as f64);
    results.insert("r_squared".to_string(), r2 as f64);
    results.insert("is_critical".to_string(), if is_critical { 1.0 } else { 0.0 });

    results
}

// =============================================================================
// Suite 8: Avalanche-Guided Response Generation
// =============================================================================

fn bench_avalanche_guided() -> HashMap<String, f64> {
    println!("[Suite 8] Avalanche-Guided Response Generation...");
    
    let mut trainer = SemanticTrainer::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
    
    // Add recurrent connections for criticality
    trainer.concept_graph.add_self_connections();
    trainer.concept_graph.add_preferential_attachment(800);
    
    let mut observer = PowerLawObserver::new(100);
    let mut selector = AvalancheGuidedSelector::new(
        &mut trainer, &mut encoder, &mut decoder, &mut observer,
    );
    
    let base_inputs = vec![
        "hund", "katze", "vogel", "stern", "baum",
        "hallo", "gut", "danke", "wie", "was",
    ];
    let test_inputs: Vec<String> = base_inputs.iter()
        .cycle()
        .take(100)
        .map(|s| s.to_string())
        .collect();
    
    let mut total_length = 0usize;
    let mut grammatical_count = 0usize;
    let mut vocab_hits = 0usize;
    let mut total_words = 0usize;
    
    for input in &test_inputs {
        let selection = selector.select(input);
        let response = build_response_from_selection(&selection);
        let len = response.len();
        total_length += len;
        
        for w in &response {
            total_words += 1;
            if encoder.neuron_for_word(w).is_some() {
                vocab_hits += 1;
            }
        }
        
        // Grammatical check: DET NOUN VERB or DET NOUN ADJ VERB
        if len >= 3 {
            let dets = ["der", "die", "das"];
            let verbs = ["läuft", "springt", "ist", "sieht", "schläft", "fliegt", "scheint"];
            if dets.contains(&response[0].as_str()) && verbs.contains(&response[len-1].as_str()) {
                grammatical_count += 1;
            }
        }
    }
    
    let n = test_inputs.len() as f64;
    let mut results = HashMap::new();
    results.insert("avg_length".to_string(), total_length as f64 / n);
    results.insert("vocab_coverage".to_string(), vocab_hits as f64 / total_words.max(1) as f64);
    results.insert("grammatical_rate".to_string(), grammatical_count as f64 / n);
    results.insert("avg_avalanche_size".to_string(), 0.0);
    
    println!("  Avg response length: {:.1}", results["avg_length"]);
    println!("  Vocab coverage: {:.1}%", results["vocab_coverage"] * 100.0);
    println!("  Grammatical rate: {:.1}%", results["grammatical_rate"] * 100.0);
    
    results
}

// =============================================================================
// Suite 9: Template vs Avalanche-Guided Comparison
// =============================================================================

fn bench_template_vs_avalanche() -> HashMap<String, f64> {
    println!("[Suite 9] Template vs Avalanche-Guided Comparison...");
    
    let mut trainer = SemanticTrainer::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);
    
    trainer.concept_graph.add_self_connections();
    trainer.concept_graph.add_preferential_attachment(800);
    
    let mut observer = PowerLawObserver::new(100);
    let mut selector = AvalancheGuidedSelector::new(
        &mut trainer, &mut encoder, &mut decoder, &mut observer,
    );
    
    let test_inputs = vec![
        "hund", "katze", "vogel", "stern", "baum",
        "hallo", "gut", "danke", "wie", "was",
    ];
    
    let mut template_grammatical = 0usize;
    let mut template_lengths = Vec::new();
    for input in &test_inputs {
        let response = generate_response(
            &[input.to_string()],
            &mut trainer,
            &TransitionalLearner::new(),
            &mut encoder,
            &mut decoder,
            &ConversationBuffer::new(10),
        );
        template_lengths.push(response.len());
        if response.len() >= 3 {
            let verbs = ["läuft", "springt", "ist", "sieht", "schläft", "fliegt", "scheint"];
            let dets = ["der", "die", "das"];
            if dets.contains(&response[0].as_str()) && verbs.contains(&response[response.len()-1].as_str()) {
                template_grammatical += 1;
            }
        }
    }
    
    let mut avalanche_grammatical = 0usize;
    let mut avalanche_lengths = Vec::new();
    for input in &test_inputs {
        let selection = selector.select(input);
        let response = build_response_from_selection(&selection);
        avalanche_lengths.push(response.len());
        if response.len() >= 3 {
            let verbs = ["läuft", "springt", "ist", "sieht", "schläft", "fliegt", "scheint"];
            let dets = ["der", "die", "das"];
            if dets.contains(&response[0].as_str()) && verbs.contains(&response[response.len()-1].as_str()) {
                avalanche_grammatical += 1;
            }
        }
    }
    
    let n = test_inputs.len() as f64;
    let template_gram_rate = template_grammatical as f64 / n;
    let avalanche_gram_rate = avalanche_grammatical as f64 / n;
    let template_avg_len = template_lengths.iter().sum::<usize>() as f64 / n;
    let avalanche_avg_len = avalanche_lengths.iter().sum::<usize>() as f64 / n;
    
    println!("  Template:    grammatical={:.1}%, avg_len={:.1}", template_gram_rate * 100.0, template_avg_len);
    println!("  Avalanche:   grammatical={:.1}%, avg_len={:.1}", avalanche_gram_rate * 100.0, avalanche_avg_len);
    println!("  Improvement: {:+.1}% grammatical", (avalanche_gram_rate - template_gram_rate) * 100.0);
    
    let mut results = HashMap::new();
    results.insert("template_grammatical_rate".to_string(), template_gram_rate);
    results.insert("avalanche_grammatical_rate".to_string(), avalanche_gram_rate);
    results.insert("template_avg_length".to_string(), template_avg_len);
    results.insert("avalanche_avg_length".to_string(), avalanche_avg_len);
    results.insert("improvement".to_string(), avalanche_gram_rate - template_gram_rate);
    
    results
}

// =============================================================================
// Helpers
// =============================================================================

fn setup_lexicon(
    trainer: &mut SemanticTrainer,
    encoder: &mut TokenSpikeEncoder,
    decoder: &mut SpikeTokenDecoder,
) {
    build_extended_lexicon(trainer, encoder, decoder);
}

fn generate_response(
    user_input: &[String],
    trainer: &mut SemanticTrainer,
    transitional: &TransitionalLearner,
    encoder: &mut TokenSpikeEncoder,
    _decoder: &mut SpikeTokenDecoder,
    _conv: &ConversationBuffer,
) -> Vec<String> {
    let mut rng = rand::thread_rng();

    let greetings = ["hallo", "hi", "guten tag", "moin", "servus"];
    if user_input.iter().any(|w| greetings.contains(&w.as_str())) {
        let sentence = vec!["hallo".to_string(), "ich".to_string(), "bin".to_string(), "goldsnnail".to_string()];
        let filtered: Vec<String> = sentence.into_iter().filter(|w| encoder.neuron_for_word(w).is_some()).collect();
        if !filtered.is_empty() { return filtered; }
    }

    let farewells = ["tschüss", "auf wiedersehen", "bye", "ciao"];
    if user_input.iter().any(|w| farewells.contains(&w.as_str())) {
        return vec!["auf".to_string(), "wiedersehen".to_string()];
    }

    let questions = ["wie", "was", "wer", "wo", "wann", "warum"];
    if user_input.iter().any(|w| questions.contains(&w.as_str())) {
        let answers = vec![
            vec!["ich".to_string(), "bin".to_string(), "goldsnnail".to_string()],
            vec!["der".to_string(), "hund".to_string(), "läuft".to_string()],
            vec!["die".to_string(), "katze".to_string(), "schläft".to_string()],
            vec!["der".to_string(), "stern".to_string(), "scheint".to_string()],
        ];
        let answer = answers[rng.r#gen::<usize>() % answers.len()].clone();
        let filtered: Vec<String> = answer.into_iter().filter(|w| encoder.neuron_for_word(w).is_some()).collect();
        if !filtered.is_empty() { return filtered; }
    }

    if user_input.contains(&"danke".to_string()) || user_input.contains(&"thanks".to_string()) {
        let sentence = vec!["bitte".to_string(), "gern".to_string(), "geschehen".to_string()];
        let filtered: Vec<String> = sentence.into_iter().filter(|w| encoder.neuron_for_word(w).is_some()).collect();
        if !filtered.is_empty() { return filtered; }
    }

    if transitional.size() > 0 {
        if let Some(last_word) = user_input.last() {
            if encoder.neuron_for_word(last_word).is_some() {
                let generated = transitional.generate(last_word, 4, &mut rng);
                if !generated.contains(&"???".to_string()) {
                    let filtered: Vec<String> = generated
                        .into_iter()
                        .filter(|w| encoder.neuron_for_word(w).is_some())
                        .collect();
                    if filtered.len() >= 2 {
                        return filtered;
                    }
                }
            }
        }
    }

    let templates = vec![
        vec!["ich".to_string(), "bin".to_string(), "goldsnnail".to_string()],
        vec!["der".to_string(), "hund".to_string(), "läuft".to_string()],
        vec!["die".to_string(), "katze".to_string(), "schläft".to_string()],
        vec!["der".to_string(), "vogel".to_string(), "fliegt".to_string()],
        vec!["der".to_string(), "baum".to_string(), "wächst".to_string()],
        vec!["die".to_string(), "blume".to_string(), "blüht".to_string()],
        vec!["das".to_string(), "wasser".to_string(), "fließt".to_string()],
        vec!["der".to_string(), "stern".to_string(), "scheint".to_string()],
    ];

    for template in &templates {
        let filtered: Vec<String> = template.iter().cloned().filter(|w| encoder.neuron_for_word(w).is_some()).collect();
        if !filtered.is_empty() { return filtered; }
    }
    let fallback = fallback_noun(encoder, &mut rng);
    vec!["der".to_string(), fallback, "ist".to_string()]
}

fn fallback_noun(encoder: &TokenSpikeEncoder, rng: &mut impl Rng) -> String {
    let nouns = ["hund", "katze", "vogel", "stern", "baum", "blume", "wasser", "feuer", "erde", "haus"];
    let available: Vec<_> = nouns.iter().filter(|&&w| encoder.neuron_for_word(w).is_some()).cloned().collect();
    if available.is_empty() {
        "hund".to_string()
    } else {
        available[rng.r#gen::<usize>() % available.len()].to_string()
    }
}

fn export_results(results: &HashMap<&str, HashMap<String, f64>>) {
    fs::create_dir_all("docs/src/development").unwrap();

    let mut json = String::from("{\n");
    json.push_str(&format!("  \"timestamp\": \"{}\",\n", chrono_now()));
    json.push_str("  \"suites\": {\n");

    let suite_names = ["snn_encoding", "response_quality", "learning_speed", "reasoning", "online_learning", "world_model", "criticality", "avalanche_guided", "template_vs_avalanche"];
    for (i, suite) in suite_names.iter().enumerate() {
        if let Some(data) = results.get(suite) {
            json.push_str(&format!("    \"{}\": {{\n", suite));
            for (j, (key, value)) in data.iter().enumerate() {
                let formatted = if *value == value.round() {
                    format!("{}", *value as i64)
                } else {
                    format!("{:.4}", value)
                };
                json.push_str(&format!("      \"{}\": {}", key, formatted));
                if j < data.len() - 1 {
                    json.push_str(",\n");
                } else {
                    json.push('\n');
                }
            }
            json.push('}');
            if i < suite_names.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
    }
    json.push_str("  }\n}\n");

    let path = "docs/src/development/benchmark_results.json";
    let mut file = fs::File::create(path).unwrap();
    use std::io::Write;
    file.write_all(json.as_bytes()).unwrap();
}

fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{}", now.as_secs())
}
