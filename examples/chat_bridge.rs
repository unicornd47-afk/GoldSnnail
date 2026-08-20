//! GoldSnnail Chat Bridge — Proof of Concept
//!
//! Demonstrates the SNN-LLM bridge:
//! 1. Register lexicon words as neurons
//! 2. Encode sentences into spike trains
//! 3. Decode spikes back into words
//! 4. Simple conversation loop
//!
//! Usage:
//!   cargo run --example chat_bridge --release

use goldsnnail::{
    Lexicon, TokenSpikeEncoder, SpikeTokenDecoder,
    ConversationBuffer, ConversationTurn,
};
use rand::Rng;

fn main() {
    println!("=== GoldSnnail Chat Bridge ===\n");

    // --- Setup ---
    let lexicon = Lexicon::new(1.0);
    let mut encoder = TokenSpikeEncoder::new(3.0, 5);
    let mut decoder = SpikeTokenDecoder::new(1);

    encoder.register_lexicon(&lexicon);
    decoder.register_lexicon(&lexicon);

    println!("Registered {} words in SNN-LLM bridge", encoder.vocab_size());

    // --- Test Spike Encoding ---
    println!("\n--- Spike Encoding ---");
    let test_sentences = vec![
        vec!["der".to_string(), "hund".to_string(), "läuft".to_string()],
        vec!["die".to_string(), "katze".to_string(), "schläft".to_string()],
        vec!["der".to_string(), "vogel".to_string(), "fliegt".to_string()],
    ];

    for sentence in &test_sentences {
        let spikes = encoder.encode_sentence(sentence);
        println!("  '{}' → {} spike events", sentence.join(" "), spikes.len());
    }

    // --- Test Spike Decoding ---
    println!("\n--- Spike Decoding ---");
    let test_spikes = vec![0u32, 0u32, 8u32, 8u32, 8u32, 18u32];
    let spike_buffer = goldsnnail::SpikeBuffer {
        indices: test_spikes.clone(),
        count: 100,
    };
    let decoded = decoder.decode_spikes(&spike_buffer);
    println!("  Spikes {:?} → decoded: {:?}", test_spikes, decoded);

    // --- Conversation Buffer Test ---
    println!("\n--- Conversation Buffer ---");
    let mut conv = ConversationBuffer::new(10);
    conv.push(ConversationTurn::new_user("Hallo!".to_string()));
    conv.push(ConversationTurn::new_assistant(
        "Ich bin GoldSnnail.".to_string(),
        vec!["ich".to_string(), "bin".to_string(), "goldsnnail".to_string()],
    ));
    conv.push(ConversationTurn::new_user("Was bist du?".to_string()));

    println!("  Buffer size: {}", conv.len());
    println!("  Last user: '{}'", conv.last_user_turn().unwrap().text);
    println!("  Last assistant: '{}'", conv.last_assistant_turn().unwrap().text);

    // --- Simple Chat Simulation ---
    println!("\n--- Simple Chat Simulation ---");
    let mut rng = rand::thread_rng();
    let greetings = vec!["hallo", "hi", "guten tag", "moin"];
    let responses = vec![
        vec!["ich".to_string(), "bin".to_string(), "goldsnnail".to_string()],
        vec!["der".to_string(), "hund".to_string(), "läuft".to_string()],
        vec!["die".to_string(), "katze".to_string(), "schläft".to_string()],
        vec!["der".to_string(), "vogel".to_string(), "fliegt".to_string()],
    ];

    for _ in 0..5 {
        let greeting = greetings[rng.r#gen::<usize>() % greetings.len()];
        println!("  User: {}", greeting);
        
        let response = &responses[rng.r#gen::<usize>() % responses.len()];
        let response_str = response.join(" ");
        println!("  GoldSnnail: {}", response_str);
        
        let spikes = encoder.encode_sentence(response);
        let decoded = decoder.decode_spikes(&goldsnnail::SpikeBuffer {
            indices: spikes.iter().map(|e| e.src.0 as u32).collect(),
            count: 100,
        });
        println!("  Decoded: {:?}", decoded);
    }

    // --- Export Conversation ---
    println!("\n--- Export ---");
    let json = conv.to_json();
    println!("  Conversation JSON ({} bytes)", json.len());
    println!("  First 200 chars: {}", &json[..200.min(json.len())]);

    println!("\n=== Chat Bridge Ready ===");
    println!("Next steps:");
    println!("  1. Add WorldModel for next-token prediction");
    println!("  2. Add Curiosity module for active learning");
    println!("  3. Build CLI chat interface");
    println!("  4. Add web frontend (chat.html)");
}
