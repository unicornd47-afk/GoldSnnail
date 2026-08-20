//! Semantics — Token Encoding, Concept Graphs, and Contrastive Learning
//!
//! This module bridges raw spike trains and abstract knowledge:
//! - `encoder`: Maps discrete tokens to Hyperbolic Quaternion embeddings
//! - `concept_graph`: Stores concepts as nodes and relations as sparse edges
//! - `contrastive`: Self-supervised learning in hyperbolic space
//! - `curriculum`: Training data generation for semantic acquisition
//! - `token_engine`: Token lexicon, noise injection, multi-objective reward
//! - `learner`: Bridge from reward to weight updates via RSTDP

pub mod encoder;
pub mod concept_graph;
pub mod contrastive;
pub mod curriculum;
pub mod token_engine;
pub mod learner;

pub use encoder::SemanticEncoder;
pub use concept_graph::{ConceptGraph, ConceptNode, RelationType, SemanticEdge, BridgeEdge};
pub use contrastive::HyperbolicContrastive;
pub use token_engine::{
    Lexicon, LexiconToken, TokenClass, NoiseInjector,
    SemanticRewardEngine, RewardWeights, RewardSignal,
    TokenComposer, SemanticTrainer,
};
pub use learner::{SemanticLearner, LearningRates, LearningMetrics, EpochMetrics};
