//! Reasoning Engine — Multi-step inference chain for GoldWorm
//!
//! Orchestrates decomposition, recall, association, and synthesis
//! into an explicit chain of Thoughts for interpretability.

use super::{
    ConversationBuffer, TokenSpikeEncoder,
};
use crate::baby::TransitionalLearner;
use crate::semantics::SemanticTrainer;

// =============================================================================
// 1. THOUGHT — Single reasoning step
// =============================================================================

/// A single reasoning step in the chain.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Thought {
    pub step: usize,
    pub input: String,
    pub output: String,
    pub confidence: f64,
    pub spikes: usize,
}

impl Thought {
    pub fn new(step: usize, input: impl Into<String>, output: impl Into<String>, confidence: f64, spikes: usize) -> Self {
        Self {
            step,
            input: input.into(),
            output: output.into(),
            confidence: tanh_soft_clamp(confidence),
            spikes,
        }
    }
}

// =============================================================================
// 2. THOUGHT CHAIN — Ordered reasoning steps
// =============================================================================

/// A sequence of reasoning steps.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ThoughtChain {
    pub thoughts: Vec<Thought>,
    pub max_steps: usize,
}

impl ThoughtChain {
    pub fn new(max_steps: usize) -> Self {
        Self {
            thoughts: Vec::with_capacity(max_steps),
            max_steps,
        }
    }

    /// Add a thought to the chain. Returns false if the chain is full.
    pub fn add(&mut self, thought: Thought) -> bool {
        if self.thoughts.len() >= self.max_steps {
            return false;
        }
        self.thoughts.push(thought);
        true
    }

    /// Get the last thought in the chain.
    pub fn last(&self) -> Option<&Thought> {
        self.thoughts.last()
    }

    /// Number of thoughts in the chain.
    pub fn len(&self) -> usize {
        self.thoughts.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.thoughts.is_empty()
    }

    /// Format the chain as readable text.
    pub fn to_string(&self) -> String {
        let mut out = String::with_capacity(self.thoughts.len() * 64);
        for thought in &self.thoughts {
            out.push_str(&format!(
                "[Step {}] {}\n  → {}\n  (confidence: {:.3}, spikes: {})\n",
                thought.step,
                truncate(&thought.input, 40),
                truncate(&thought.output, 60),
                thought.confidence,
                thought.spikes,
            ));
        }
        out
    }
}

// =============================================================================
// 3. REASONING ENGINE — Multi-step inference orchestrator
// =============================================================================

/// Orchestrates multi-step reasoning using semantic and SNN components.
pub struct ReasoningEngine<'a> {
    pub trainer: &'a mut SemanticTrainer,
    pub transitional: &'a mut TransitionalLearner,
    pub encoder: &'a mut TokenSpikeEncoder,
    pub chain_max_steps: usize,
}

impl<'a> ReasoningEngine<'a> {
    pub fn new(
        trainer: &'a mut SemanticTrainer,
        transitional: &'a mut TransitionalLearner,
        encoder: &'a mut TokenSpikeEncoder,
    ) -> Self {
        Self {
            trainer,
            transitional,
            encoder,
            chain_max_steps: 5,
        }
    }

    /// Main reasoning entry point.
    pub fn reason(&mut self, input: &str, _conv: &ConversationBuffer) -> ThoughtChain {
        let mut chain = ThoughtChain::new(self.chain_max_steps);
        let tokens: Vec<String> = input
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        // Step 1 — Decompose
        let spike_events = self.encoder.encode_sentence(&tokens);
        let spike_count = spike_events.len();

        let step0 = Thought::new(
            0,
            input,
            format!("Analyzing: {:?}", tokens),
            0.8,
            spike_count,
        );
        if !chain.add(step0) {
            return chain;
        }

        // Step 2 — Recall
        let resolved = self.trainer.composer.resolve(&tokens);
        let (recall_output, recall_conf) = if resolved.is_empty() {
            let start = tokens.first().map(|s| s.as_str()).unwrap_or("???");
            let mut rng = rand::thread_rng();
            let generated = self.transitional.generate(start, 3, &mut rng);
            (
                format!("No known concepts. Generated: {:?}", generated),
                0.5,
            )
        } else {
            let concept = &resolved[0];
            (
                format!("Found concept: {}", concept.surface),
                0.7,
            )
        };

        let step1 = Thought::new(1, input, recall_output, recall_conf, spike_count);
        if !chain.add(step1) {
            return chain;
        }

        // Step 3 — Associate
        if let Some(first) = resolved.first() {
            match self
                .trainer
                .concept_graph
                .nearest_neighbors(&first.hyperbolic, 2)
            {
                Ok(nn) if nn.len() >= 2 => {
                    let (neighbor_id, dist) = nn[1];
                    if dist < 0.5 {
                        let label = &self.trainer.concept_graph.nodes[neighbor_id].label;
                        let assoc_conf = 0.6 * (1.0 - dist);
                        let step2 = Thought::new(
                            2,
                            input,
                            format!("Related: {} (dist: {:.3})", label, dist),
                            assoc_conf,
                            spike_count,
                        );
                        if !chain.add(step2) {
                            return chain;
                        }
                    } else {
                        let step2 = Thought::new(
                            2,
                            input,
                            "No close semantic neighbors".to_string(),
                            0.2,
                            spike_count,
                        );
                        if !chain.add(step2) {
                            return chain;
                        }
                    }
                }
                _ => {
                    let step2 = Thought::new(
                        2,
                        input,
                        "Association step failed".to_string(),
                        0.2,
                        spike_count,
                    );
                    if !chain.add(step2) {
                        return chain;
                    }
                }
            }
        } else {
            let step2 = Thought::new(
                2,
                input,
                "No concepts to associate".to_string(),
                0.2,
                spike_count,
            );
            if !chain.add(step2) {
                return chain;
            }
        }

        // Step 4 — Synthesize
        let synth_output = if !resolved.is_empty() {
            let noun = &resolved[0].surface;
            let verb = tokens.last().map(|s| s.as_str()).unwrap_or("ist");
            let sentence = self.trainer.composer.build_sentence_simple(noun, verb);
            format!("Synthesized: {:?}", sentence)
        } else {
            let start = tokens.first().map(|s| s.as_str()).unwrap_or("???");
            let mut rng = rand::thread_rng();
            let generated = self.transitional.generate(start, 3, &mut rng);
            format!("Synthesized: {:?}", generated)
        };

        let step3 = Thought::new(3, input, synth_output, 0.6, spike_count);
        let _ = chain.add(step3);

        chain
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// tanh-based soft clamping: maps any value to [0.0, 1.0].
#[inline]
fn tanh_soft_clamp(val: f64) -> f64 {
    val.tanh() * 0.5 + 0.5
}

#[inline]
fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantics::Lexicon;

    fn make_dummy_trainer() -> SemanticTrainer {
        SemanticTrainer::new(1.0)
    }

    fn make_dummy_encoder() -> TokenSpikeEncoder {
        let mut encoder = TokenSpikeEncoder::new(1.0, 5);
        let lex = Lexicon::new(1.0);
        encoder.register_lexicon(&lex);
        encoder
    }

    #[test]
    fn new_creates_engine() {
        let mut trainer = make_dummy_trainer();
        let mut transitional = TransitionalLearner::new();
        let mut encoder = make_dummy_encoder();

        let _engine = ReasoningEngine::new(&mut trainer, &mut transitional, &mut encoder);
    }

    #[test]
    fn reason_produces_chain() {
        let mut trainer = make_dummy_trainer();
        let mut transitional = TransitionalLearner::new();
        let mut encoder = make_dummy_encoder();
        let conv = ConversationBuffer::new(10);

        let mut engine = ReasoningEngine::new(&mut trainer, &mut transitional, &mut encoder);
        let chain = engine.reason("hallo", &conv);

        assert!(chain.len() >= 1, "Chain should have at least 1 thought, got {}", chain.len());
    }

    #[test]
    fn thought_chain_add_and_len() {
        let mut chain = ThoughtChain::new(2);

        let t1 = Thought::new(0, "in", "out1", 0.9, 1);
        let t2 = Thought::new(1, "in", "out2", 0.8, 2);
        let t3 = Thought::new(2, "in", "out3", 0.7, 3);

        assert!(chain.add(t1));
        assert!(chain.add(t2));
        assert_eq!(chain.len(), 2);

        assert!(!chain.add(t3), "Chain should reject when full");
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn thought_chain_to_string() {
        let mut chain = ThoughtChain::new(10);
        chain.add(Thought::new(0, "hello", "greeting", 0.9, 5));
        chain.add(Thought::new(1, "world", "planet", 0.8, 3));

        let s = chain.to_string();
        assert!(s.contains("greeting"), "to_string should contain first output");
        assert!(s.contains("planet"), "to_string should contain second output");
    }
}
