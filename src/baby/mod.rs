//! Baby Learning Frameworks — 6 Mechanisms for Emergent Learning
//!
//! This module implements 3 missing learning principles:
//! - `infomax`: Mutual Information as intrinsic curiosity reward
//! - `ucb`: Upper Confidence Bound for uncertainty-driven exploration
//! - `transitional`: Transitional Probability Learner for grammar acquisition

pub mod infomax;
pub mod ucb;
pub mod transitional;

pub use infomax::InfomaxReward;
pub use ucb::UCBExplorer;
pub use transitional::TransitionalLearner;