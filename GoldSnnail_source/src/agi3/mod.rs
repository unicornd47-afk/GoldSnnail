//! ARC-AGI-3 — Interactive Agent Module for the GoldWorm Project
//!
//! This module implements an interactive ARC-AGI-3 (Abstraction and Reasoning
//! Corpus - 3rd generation) agent. It builds on top of the existing GoldWorm
//! substrate, reusing three foundational components:
//!
//! - **SNN**: The spiking neural network engine (`swarm::snn_core`) provides
//!   the underlying neural dynamics and state arena used as the agent's
//!   recurrent controller.
//! - **World Model**: The predictive `world_model::WorldModel` supplies latent
//!   state estimation and transition prediction, which the agent leverages for
//!   model-based planning and curiosity-driven exploration.
//! - **RL Substrate**: The `rl` module's value/policy heads and TD-learning
//!   machinery ground the agent's policy optimization and reward evaluation.
//!
//! The agent operates in grid-based environments where it must perceive
//! observations, reason over short horizons, and select actions to maximize
//! cumulative reward. Submodule responsibilities are partitioned as follows:
//!
//! - [`types`]: Shared data structures (actions, observations, rewards, etc.).
//! - [`environment`]: Grid-world environments, the environment step API, and a
//!   registry/loader for demo tasks.
//! - [`perception`]: Transforms raw grid observations into latent embeddings
//!   consumed by the controller.
//! - [`memory`]: Episodic memory and working-memory buffers for temporal
//!   context.
//! - [`planning`]: Short-horizon model-predictive planning using the world
//!   model.
//! - [`exploration`]: Curiosity and uncertainty-driven exploration strategies.
//! - [`agent`]: The top-level `ArcAgent3` that orchestrates perception, memory,
//!   planning, and action selection.
//!
//! Core shared types are defined directly in this file so that every submodule
//! can reference them without introducing cross-submodule coupling.

pub mod environment;
pub mod perception;
pub mod memory;
pub mod planning;
pub mod exploration;
pub mod agent;

/// The 6 basic actions an agent can take in an ARC-AGI-3 grid environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Action {
    /// Do nothing this step.
    Noop,
    /// Move one cell up.
    Up,
    /// Move one cell down.
    Down,
    /// Move one cell left.
    Left,
    /// Move one cell right.
    Right,
    /// Interact with the cell the agent is facing.
    Interact,
}

/// A raw observation of the environment state at a single timestep.
///
/// The `frame` is a 2D grid of cell values (`Vec<Vec<u8>>`), where each
/// `u8` encodes a tile type or object identifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Observation {
    /// 2D grid of cell values representing the current environment state.
    pub frame: Vec<Vec<u8>>,
    /// Width of the grid (number of columns).
    pub width: usize,
    /// Height of the grid (number of rows).
    pub height: usize,
    /// Global timestep / turn counter.
    pub turn: u64,
    /// Unique identifier for the current episode.
    pub episode_id: String,
}

/// A structured reward signal combining external task rewards with
/// intrinsic motivation (e.g. curiosity).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Reward {
    /// Reward provided by the environment for task progress.
    pub extrinsic: f64,
    /// Reward derived from internal drives such as prediction error
    /// reduction or information gain.
    pub intrinsic: f64,
    /// Combined reward: `extrinsic + intrinsic * curiosity_weight`.
    pub total: f64,
}

/// Hyperparameters governing agent behavior, planning, and learning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentConfig {
    /// Maximum number of steps allowed per episode.
    pub max_episode_steps: u64,
    /// Exploration vs. exploitation trade-off coefficient for intrinsic
    /// motivation. Higher values weight curiosity more heavily.
    pub exploration_beta: f64,
    /// Number of future steps considered during model-predictive planning.
    pub planning_horizon: usize,
    /// Learning rate for the value (critic) head.
    pub learning_rate_value: f64,
    /// Learning rate for the policy (actor) head.
    pub learning_rate_policy: f64,
    /// Discount factor for future rewards.
    pub discount_gamma: f64,
    /// Weight applied to intrinsic reward when computing the total reward.
    pub curiosity_weight: f64,
    /// Input feature dimension for the grid encoder.
    pub input_dim: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 200,
            exploration_beta: 0.3,
            planning_horizon: 5,
            learning_rate_value: 0.01,
            learning_rate_policy: 0.005,
            discount_gamma: 0.95,
            curiosity_weight: 0.1,
            input_dim: 100,
        }
    }
}

/// Outcome of a completed episode.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpisodeResult {
    /// Unique identifier for the episode.
    pub episode_id: String,
    /// Cumulative reward obtained over the episode.
    pub total_reward: f64,
    /// Number of steps taken before termination.
    pub steps: u64,
    /// Whether the episode was solved (e.g. goal reached).
    pub solved: bool,
    /// Snapshot of the final environment observation, if recorded.
    pub final_state: Option<Observation>,
}

pub use agent::ArcAgent3;

pub use environment::{
    ArcEnvironment, ArcEnvironmentStep, EnvironmentRegistry, load_demo_environments,
};
