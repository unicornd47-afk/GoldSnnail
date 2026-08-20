//! GoldSnnail CLI Chat Interface
//!
//! Real-time conversation with the GoldSnnail SNN-LLM engine.
//!
//! Usage:
//!   cargo run --example chat_cli --release

use goldsnnail::{
    TokenSpikeEncoder, SpikeTokenDecoder,
    ConversationBuffer, ConversationTurn, SemanticTrainer,
    TransitionalLearner,
    ReasoningEngine, ThoughtChain, OnlineLearner,
    WorldChat, WorldGeometry, ChatArena,
    PatchEncoder, generate_synthetic_cifar10_batch, map_cifar_label_to_lexicon,
    build_extended_lexicon,
};
use rand::Rng;
use std::io::Write;

fn main() {
    println!("=== GoldSnnail SNN-LLM Chat Engine ===\n");
    println!("Initializing neural substrate...\n");

    // --- Setup ---
    let mut trainer = SemanticTrainer::new(1.0);
    let mut transitional = TransitionalLearner::new();
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);
    let patch_encoder = PatchEncoder::new(8, 8, 1.0);
    let mut conv = ConversationBuffer::new(50);

    encoder.register_lexicon(&trainer.lexicon);
    decoder.register_lexicon(&trainer.lexicon);

    let geom = WorldGeometry::new(2, 4, 1.0);
    geom.validate(trainer.lexicon.tokens[0].hyperbolic.coords.len()).expect("WorldGeometry dimension must match lexicon");
    let mut world_chat = WorldChat::from_config(geom);

    build_extended_lexicon(&mut trainer, &mut encoder, &mut decoder);

    let mut chat_arena = ChatArena::new();
    let trainer_idx = chat_arena.push(trainer, encoder, decoder);
    let mut online_learner = OnlineLearner::new(trainer_idx, trainer_idx, trainer_idx);

    println!("  Lexicon: {} words", chat_arena.trainers[trainer_idx].lexicon.tokens.len());
    println!("  SNN-LLM bridge: {} neurons", chat_arena.encoders[trainer_idx].vocab_size());
    println!("  Conversation buffer: 50 turns\n");

    // Register some basic grammar patterns
    chat_arena.trainers[trainer_idx].reward_engine.learn_pattern(vec![
        goldsnnail::TokenClass::Determiner,
        goldsnnail::TokenClass::NounConcrete,
        goldsnnail::TokenClass::VerbAction,
    ]);
    chat_arena.trainers[trainer_idx].reward_engine.learn_pattern(vec![
        goldsnnail::TokenClass::Determiner,
        goldsnnail::TokenClass::Adjective,
        goldsnnail::TokenClass::NounConcrete,
        goldsnnail::TokenClass::VerbAction,
    ]);

    println!("{}", "=".repeat(50));
    println!("GoldSnnail: Hallo! Ich bin GoldSnnail, ein neuronales System.");
    println!("         Sprich mit mir. (Tippe 'quit' zum Beenden)");
    println!("{}\n", "=".repeat(50));

    // --- Conversation Loop ---
    let mut total_spikes: usize = 0;
    let mut total_reasoning_steps: usize = 0;
    let mut total_learned: usize = 0;

    loop {
        // Get user input
        print!("Du: ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("\nGoldSnnail: Auf Wiedersehen!");
            break;
        }

        // --- Special Commands ---
        if input.eq_ignore_ascii_case("/image") || input.eq_ignore_ascii_case("/bild") {
            let batch = generate_synthetic_cifar10_batch(1, None);
            let img = &batch[0];
            let label = map_cifar_label_to_lexicon(img.label);

            let pixels_f64: Vec<f64> = img.pixels.iter().map(|&p| p as f64).collect();
            let tokens = patch_encoder.encode_image(&pixels_f64, 32, 32);

            println!("GoldSnnail: Ich sehe ein Bild. Es enthält {} visuelle Patches.", tokens.len());
            println!("         Das Bild scheint ein {} zu sein.", label);

            let _ = std::fs::write(
                "docs/src/development/chat_state.json",
                conv.to_json(),
            );
            continue;
        }

        // Tokenize user input
        let user_tokens: Vec<String> = input.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        // Add to conversation buffer
        conv.push(ConversationTurn::new_user(input.to_string()));

        // Encode to spikes
        let mut encoded_words = Vec::new();
        let mut unknown_words = Vec::new();
        for token in &user_tokens {
            if chat_arena.encoders[trainer_idx].neuron_for_word(token).is_some() {
                encoded_words.push(token.clone());
            } else {
                unknown_words.push(token.clone());
            }
        }
        let user_spikes = chat_arena.encoders[trainer_idx].encode_sentence(&encoded_words);
        total_spikes += user_spikes.len();
        println!("  [SNN: {} known, {} unknown → {} spikes]",
            encoded_words.len(), unknown_words.len(), user_spikes.len());
        if !unknown_words.is_empty() {
            println!("  [Unknown: {:?}]", unknown_words);
        }

        // --- Online Learning ---
        if !unknown_words.is_empty() {
            let learned = online_learner.process_unknown(&mut chat_arena, &unknown_words, &conv).unwrap_or_default();
            total_learned += learned.len();
            if !learned.is_empty() {
                println!("  [Learned: {:?}]", learned);
            }
        }

        // --- Generate Response ---
        let trainer = &mut chat_arena.trainers[trainer_idx];
        let encoder = &mut chat_arena.encoders[trainer_idx];
        let decoder = &mut chat_arena.decoders[trainer_idx];

        let response = generate_response(
            &user_tokens,
            trainer,
            &transitional,
            encoder,
            decoder,
            &conv,
        );

        // --- Reason about the response ---
        {
            let mut reasoner = ReasoningEngine::new(trainer, &mut transitional, encoder);
            let chain: ThoughtChain = reasoner.reason(&input, &conv);
            if chain.len() > 1 {
                println!("  [Reasoning chain]:");
                for thought in chain.thoughts.iter() {
                    println!("    Step {}: {} (confidence: {:.2}, spikes: {})",
                        thought.step, thought.output, thought.confidence, thought.spikes);
                }
                println!();
            }
            total_reasoning_steps += chain.len();
        }

        // --- World Model Prediction ---
        {
            let trainer = &chat_arena.trainers[trainer_idx];
            if let Some(predicted_word) = world_chat.predict_response_word(trainer, &user_tokens) {
                if chat_arena.encoders[trainer_idx].neuron_for_word(&predicted_word).is_some() {
                    println!("  [World model predicts: {}]", predicted_word);
                }
            }
        }

        // Decode response for display
        let _response_tokens: Vec<String> = response.clone();
        let response_text = response.join(" ");

        // Display response
        println!("GoldSnnail: {}\n", response_text);

        // Add to conversation buffer
        conv.push(ConversationTurn::new_assistant(response_text.clone(), response.clone()));

        // Learn from this exchange
        if !response.is_empty() {
            let full_sentence = vec!["der".to_string(), response[0].clone(), "ist".to_string()];
            transitional.observe(&full_sentence);

            let reward = chat_arena.trainers[trainer_idx].train_step(&full_sentence, false);
            if reward.total > 0.3 {
                println!("  [SNN learned: reward={:.3}]\n", reward.total);
            }
        }

        // Export state periodically
        if conv.len() % 5 == 0 {
            let _ = std::fs::write(
                "docs/src/development/chat_state.json",
                conv.to_json(),
            );
        }
    }

    // --- Final Export ---
    println!("\n=== Conversation Ended ===");
    println!("Total turns: {}", conv.len());
    println!("Learned transitions: {}", transitional.size());
    println!("Total spike events: {}", total_spikes);

    let _ = std::fs::write(
        "docs/src/development/chat_state.json",
        conv.to_json(),
    );
    println!("Conversation exported to docs/src/development/chat_state.json");

    let stats_json = format!(
        "{{\"transitions\": {}, \"spikes\": {}, \"turns\": {}, \"reasoning_steps\": {}, \"learned_words\": {}}}\n",
        transitional.size(),
        total_spikes,
        conv.len(),
        total_reasoning_steps,
        total_learned
    );
    let _ = std::fs::write(
        "docs/src/development/chat_stats.json",
        stats_json,
    );
    println!("Stats exported to docs/src/development/chat_stats.json");
}

// =============================================================================
// Response Generation
// =============================================================================

fn generate_response(
    user_input: &[String],
    trainer: &mut SemanticTrainer,
    transitional: &TransitionalLearner,
    encoder: &mut TokenSpikeEncoder,
    _decoder: &mut SpikeTokenDecoder,
    conv: &ConversationBuffer,
) -> Vec<String> {
    let mut rng = rand::thread_rng();

    // Check for greetings
    let greetings = ["hallo", "hi", "guten tag", "moin", "servus"];
    if user_input.iter().any(|w| greetings.contains(&w.as_str())) {
        let sentence = vec!["hallo".to_string(), "ich".to_string(), "bin".to_string(), "goldsnnail".to_string()];
        let filtered: Vec<String> = sentence.into_iter().filter(|w| encoder.neuron_for_word(w).is_some()).collect();
        if !filtered.is_empty() { return filtered; }
    }

    // Check for farewells
    let farewells = ["tschüss", "auf wiedersehen", "bye", "ciao"];
    if user_input.iter().any(|w| farewells.contains(&w.as_str())) {
        let sentence = vec!["auf".to_string(), "wiedersehen".to_string()];
        let filtered: Vec<String> = sentence.into_iter().filter(|w| encoder.neuron_for_word(w).is_some()).collect();
        if !filtered.is_empty() { return filtered; }
    }

    // Check for questions
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

    // Check for thanks
    if user_input.contains(&"danke".to_string()) || user_input.contains(&"thanks".to_string()) {
        let sentence = vec!["bitte".to_string(), "gern".to_string(), "geschehen".to_string()];
        let filtered: Vec<String> = sentence.into_iter().filter(|w| encoder.neuron_for_word(w).is_some()).collect();
        if !filtered.is_empty() { return filtered; }
    }

    // Strategy 1: Use transitional learner to compose a sentence from learned transitions
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

    // Strategy 2: Use concept graph to find related concepts and build a sentence
    if let Some(last_turn) = conv.last_user_turn() {
        let tokens = trainer.composer.resolve(
            &last_turn.text.split_whitespace().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        if !tokens.is_empty() {
            let token = &tokens[0];
            if let Ok(neighbors) = trainer.concept_graph.nearest_neighbors(&token.hyperbolic, 3) {
                for &(node_id, dist) in &neighbors {
                    if dist < 0.5 && dist > 0.01 {
                        if let Some(node) = trainer.concept_graph.nodes.get(node_id) {
                            if let Some(lex_token) = trainer.lexicon.get(&node.label) {
                                let nouns = ["hund", "katze", "vogel", "stern", "baum"];
                                let noun = if nouns.contains(&lex_token.surface.as_str()) {
                                    lex_token.surface.clone()
                                } else {
                                    nouns[rng.r#gen::<usize>() % nouns.len()].to_string()
                                };
                                let verbs = ["läuft", "springt", "ist", "sieht", "schläft"];
                                let verb = verbs[rng.r#gen::<usize>() % verbs.len()];
                                let sentence = trainer.composer.build_sentence_simple(&noun, verb);
                                if sentence.iter().all(|w| encoder.neuron_for_word(w).is_some()) {
                                    return sentence;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Strategy 3: Use conversation context to build a concept-aware sentence
    if let Some(last_turn) = conv.last_user_turn() {
        let tokens = trainer.composer.resolve(
            &last_turn.text.split_whitespace().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        if let Some(token) = tokens.first() {
            if let Ok(neighbors) = trainer.concept_graph.nearest_neighbors(&token.hyperbolic, 5) {
                for &(node_id, _) in &neighbors {
                    if let Some(node) = trainer.concept_graph.nodes.get(node_id) {
                        if let Some(lex_token) = trainer.lexicon.get(&node.label) {
                            if matches!(lex_token.class, goldsnnail::TokenClass::NounConcrete) {
                                let verbs = ["läuft", "springt", "ist", "sieht", "schläft"];
                                let verb = verbs[rng.r#gen::<usize>() % verbs.len()];
                                let sentence = trainer.composer.build_sentence_simple(&lex_token.surface, verb);
                                if sentence.iter().all(|w| encoder.neuron_for_word(w).is_some()) {
                                    return sentence;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Strategy 4: Random template (fallback)
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
