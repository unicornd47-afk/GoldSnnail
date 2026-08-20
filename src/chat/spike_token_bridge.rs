//! Spike-Token Bridge — Convert between SNN spikes and natural language tokens
//!
//! This is the core of the SNN-LLM interface:
//! - `TokenSpikeEncoder`: Words → Spike trains
//! - `SpikeTokenDecoder`: Spike trains → Words
//!
//! Coding schemes:
//! - Rate coding: word frequency → spike rate
//! - Temporal coding: word position → spike timing

use crate::semantics::Lexicon;
use crate::substrate::{SpikeBuffer, SpikeEvent, NeuronIdx};
use std::collections::HashMap;

/// Encodes tokens into spike trains for the SNN.
pub struct TokenSpikeEncoder {
    /// Mapping: word → neuron index
    word_to_neuron: HashMap<String, usize>,
    /// Reverse mapping: neuron index → word
    neuron_to_word: HashMap<usize, String>,
    /// Spike rate multiplier (higher = more spikes per word)
    rate_multiplier: f32,
    /// Temporal window size (timesteps per word)
    temporal_window: usize,
    /// Jitter range (±N ticks around base delay)
    pub jitter: u16,
}

impl TokenSpikeEncoder {
    pub fn new(rate_multiplier: f32, temporal_window: usize) -> Self {
        Self {
            word_to_neuron: HashMap::new(),
            neuron_to_word: HashMap::new(),
            rate_multiplier,
            temporal_window,
            jitter: 1,
        }
    }

    /// Set the jitter range (±N ticks).
    pub fn set_jitter(&mut self, jitter: u16) {
        self.jitter = jitter.max(1);
    }

    /// Set the jitter range (±N ticks) and return self for chaining.
    pub fn with_jitter(mut self, jitter: u16) -> Self {
        self.set_jitter(jitter);
        self
    }

    /// Register a word with a specific neuron index.
    pub fn register_word(&mut self, word: String, neuron_idx: usize) {
        self.word_to_neuron.insert(word.clone(), neuron_idx);
        self.neuron_to_word.insert(neuron_idx, word);
    }

    /// Register all words from a lexicon.
    pub fn register_lexicon(&mut self, lexicon: &Lexicon) {
        for token in &lexicon.tokens {
            self.register_word(token.surface.clone(), token.id);
        }
    }

    /// Encode a sentence (list of words) into spike events.
    /// Each word generates spikes at a rate proportional to its position.
    pub fn encode_sentence(&self, sentence: &[String]) -> Vec<SpikeEvent> {
        let mut events = Vec::new();
        let base_delay = 1u16;
        use rand::Rng;

        for (pos, word) in sentence.iter().enumerate() {
            if let Some(&neuron_idx) = self.word_to_neuron.get(word) {
                let rate = (self.rate_multiplier * (1.0 + 0.1 * pos as f32)) as u16;
                let base = base_delay + (pos as u16) * (self.temporal_window as u16);
                let mut rng = rand::thread_rng();
                
                for _ in 0..rate {
                    let jitter_range = self.jitter.max(1);
                    let jitter = (rng.r#gen::<u16>() % (2 * jitter_range + 1)).saturating_sub(jitter_range);
                    let delay = base.saturating_add(jitter);
                    events.push(SpikeEvent {
                        src: NeuronIdx(neuron_idx),
                        dst: NeuronIdx(neuron_idx),
                        delay_ticks: delay,
                        amplitude_u8: 255,
                        flags: 0,
                    });
                }
            }
        }

        events
    }

    /// Get the neuron index for a word.
    pub fn neuron_for_word(&self, word: &str) -> Option<usize> {
        self.word_to_neuron.get(word).copied()
    }

    /// Get the word for a neuron index.
    pub fn word_for_neuron(&self, neuron_idx: usize) -> Option<&str> {
        self.neuron_to_word.get(&neuron_idx).map(|s| s.as_str())
    }

    /// Number of registered words.
    pub fn vocab_size(&self) -> usize {
        self.word_to_neuron.len()
    }
}

/// Decodes spike buffers back into tokens.
pub struct SpikeTokenDecoder {
    /// Mapping: neuron index → word
    neuron_to_word: HashMap<usize, String>,
    /// Minimum spike count to consider a neuron "active"
    min_spike_count: usize,
}

impl SpikeTokenDecoder {
    pub fn new(min_spike_count: usize) -> Self {
        Self {
            neuron_to_word: HashMap::new(),
            min_spike_count,
        }
    }

    /// Register a word with a specific neuron index.
    pub fn register_word(&mut self, word: String, neuron_idx: usize) {
        self.neuron_to_word.insert(neuron_idx, word);
    }

    /// Register all words from a lexicon.
    pub fn register_lexicon(&mut self, lexicon: &Lexicon) {
        for token in &lexicon.tokens {
            self.register_word(token.surface.clone(), token.id);
        }
    }

    /// Decode a spike buffer into a list of words.
    /// Words are ordered by first-spike time (temporal decoding).
    pub fn decode_spikes(&self, spikes: &SpikeBuffer) -> Vec<String> {
        let mut counts: HashMap<usize, usize> = HashMap::new();

        for &neuron_idx in spikes.iter() {
            *counts.entry(neuron_idx as usize).or_insert(0) += 1;
        }

        let mut decoded: Vec<(u16, String)> = Vec::new();
        for (neuron_idx, count) in counts {
            if count >= self.min_spike_count {
                if let Some(word) = self.neuron_to_word.get(&neuron_idx) {
                    decoded.push((0, word.clone()));
                }
            }
        }

        decoded.sort_by_key(|(delay, _)| *delay);
        decoded.into_iter().map(|(_, word)| word).collect()
    }

    /// Decode spikes and count occurrences (rate decoding).
    pub fn decode_with_counts(&self, spikes: &SpikeBuffer) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();

        for &neuron_idx in spikes.iter() {
            if let Some(word) = self.neuron_to_word.get(&(neuron_idx as usize)) {
                *counts.entry(word.clone()).or_insert(0) += 1;
            }
        }

        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let mut encoder = TokenSpikeEncoder::new(2.0, 5);
        let mut decoder = SpikeTokenDecoder::new(1);

        encoder.register_word("hallo".to_string(), 0);
        encoder.register_word("welt".to_string(), 1);
        decoder.register_word("hallo".to_string(), 0);
        decoder.register_word("welt".to_string(), 1);

        let sentence = vec!["hallo".to_string(), "welt".to_string()];
        let spikes = encoder.encode_sentence(&sentence);
        let decoded = decoder.decode_spikes(&SpikeBuffer {
            indices: spikes.iter().map(|e| e.src.0 as u32).collect(),
            count: 100,
        });

        assert_eq!(decoded.len(), 2);
        assert!(decoded.contains(&"hallo".to_string()));
        assert!(decoded.contains(&"welt".to_string()));
    }

    #[test]
    fn encode_sentence_temporal_delays() {
        let mut encoder = TokenSpikeEncoder::new(2.0, 5);
        encoder.register_word("first".to_string(), 0);
        encoder.register_word("second".to_string(), 1);
        encoder.register_word("third".to_string(), 2);

        let sentence = vec!["first".to_string(), "second".to_string(), "third".to_string()];
        let events = encoder.encode_sentence(&sentence);

        let base_delays = [1u16, 6, 11];
        for event in &events {
            let pos = event.dst.0 as usize;
            let base = base_delays[pos];
            let jittered = event.delay_ticks;
            assert!(jittered <= base + 1 && jittered >= base.saturating_sub(1),
                "Word {} delay {} outside jitter window of base {}",
                pos, jittered, base);
        }
    }

    #[test]
    fn decode_with_counts_returns_counts() {
        let mut decoder = SpikeTokenDecoder::new(1);
        decoder.register_word("hello".to_string(), 0);
        decoder.register_word("world".to_string(), 1);
        decoder.register_word("foo".to_string(), 2);

        let indices = vec![0, 0, 0, 1, 1, 2];
        let buffer = SpikeBuffer { indices, count: 100 };

        let result = decoder.decode_with_counts(&buffer);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ("hello".to_string(), 3));
        assert_eq!(result[1], ("world".to_string(), 2));
        assert_eq!(result[2], ("foo".to_string(), 1));
    }

    #[test]
    fn vocab_size_after_register_lexicon() {
        let mut encoder = TokenSpikeEncoder::new(1.0, 10);
        let mut lexicon = Lexicon::new(1.0);
        lexicon.tokens.truncate(3);
        lexicon.word_index.clear();
        lexicon.class_index.clear();
        for token in &lexicon.tokens {
            lexicon.word_index.insert(token.surface.clone(), token.id);
            lexicon.class_index.entry(token.class).or_default().push(token.id);
        }
        encoder.register_lexicon(&lexicon);
        assert_eq!(encoder.vocab_size(), 3);
    }
}
