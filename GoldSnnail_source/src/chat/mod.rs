//! Chat Module — SNN-LLM Bridge for conversational AI
//!
//! This module bridges spiking neural networks and natural language:
//! - `SpikeTokenBridge`: Convert between spikes and tokens
//! - `ConversationBuffer`: Ring buffer for conversation context
//! - `ChatEngine`: Main chat loop with SNN processing

pub mod spike_token_bridge;
pub mod conversation_buffer;
pub mod thought_chain;
pub mod online_learning;
pub mod world_chat;
pub mod config;
pub mod lexicon_builder;
pub mod avalanche_guided;
pub mod dvs_encoder;

pub use spike_token_bridge::{TokenSpikeEncoder, SpikeTokenDecoder};
pub use conversation_buffer::{ConversationBuffer, ConversationTurn};
pub use thought_chain::{Thought, ThoughtChain, ReasoningEngine};
pub use online_learning::OnlineLearner;
pub use world_chat::WorldChat;
pub use config::{ChatConfig, WorldGeometry, GeometryError};
pub use lexicon_builder::{build_extended_lexicon, standard_geometry};
pub use avalanche_guided::{AvalancheGuidedSelector, AvalancheSelection, build_response_from_selection};
pub use dvs_encoder::{DvsEvent, DvsEncoder, DvsEncoderConfig, project_dvs_to_histogram, project_dvs_to_time_surface, project_dvs_to_combined_features, project_dvs_to_multiscale_features, normalize_timestamps, histogram_to_hyperbolic};
pub use crate::substrate::ChatArena;
