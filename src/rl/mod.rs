//! RL — Value/Policy Head with TD-Learning and R-STDP
//!
//! Combines a value head (critic), a policy head (actor), and R-STDP
//! plasticity into a single `RLAgent`. All state vectors are flat `Vec<f64>`
//! to stay DOD-compatible.

pub mod value_policy;

pub use value_policy::{ValueHead, PolicyHead, RLAgent, Transition, StateVector};
