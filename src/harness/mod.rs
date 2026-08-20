//! Harness Module — Self-Learning SSN Training Infrastructure
//!
//! Phase 1+2 implementation of the PLAN.md architecture, extended with
//! fractal 3-1-4-1 scalable blocks.

pub mod replay;
pub mod forward;
pub mod reward;
pub mod plasticity;
pub mod meta;
pub mod curriculum;
pub mod eval;
pub mod scale;
pub mod fractal_core;
pub mod fractal_layer;
pub mod fractal_network;
pub mod fractal_scaler;
pub mod arc_tripartite;
pub mod note_core_layer;
pub mod arc_streaming_loop;
pub mod arc_grid_decoder;

pub use replay::{ReplayBuffer, Transition, SamplingStrategy, ReplayConfig};
pub use forward::ForwardEngine;
pub use reward::{RewardEngine, RewardWeights};
pub use plasticity::{PlasticityEngine, PlasticityConfig};
pub use meta::{MetaController, MetaConfig};
pub use curriculum::{Curriculum, CurriculumTask};
pub use eval::{EvalMetrics, EvalTracker};
pub use scale::ScaleProfile;
pub use fractal_core::{FrozenCore, FrozenCoreConfig, FrozenCoreResult};
pub use fractal_layer::{FractalLayer, FractalLayerResult, SpikeAdapter};
pub use fractal_network::{FractalNetwork, FractalNetworkResult, build_3141_fractal, scale_3141};
pub use fractal_scaler::{scale_network, ScaleDir};
pub use arc_tripartite::{ArcTripartiteEncoder, ArcPhase};
pub use note_core_layer::{NoteCoreLayer, NoteCoreLayerResult};
pub use arc_streaming_loop::{ArcStreamingLoop, ArcStreamResult};
pub use arc_grid_decoder::{ArcGridDecoder};

/// Operational mode of the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HarnessMode {
    #[default]
    Train,
    Eval,
    Dream,
    Consolidate,
}

impl HarnessMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            HarnessMode::Train => "TRAIN",
            HarnessMode::Eval => "EVAL",
            HarnessMode::Dream => "DREAM",
            HarnessMode::Consolidate => "CONSOLIDATE",
        }
    }
}





