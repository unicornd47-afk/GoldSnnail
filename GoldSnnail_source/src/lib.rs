#![warn(unsafe_code)]

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "cuda")]
pub mod cuda;

pub mod semantics;
pub mod substrate;
pub mod geometry;
pub mod swarm;
pub mod routing;
pub mod telemetry;
pub mod plasticity;
pub mod attention;
pub mod memory;
pub mod memory_seed;
pub mod compression;
pub mod world_model;
pub mod rl;
pub mod vision;
pub mod audio;
pub mod baby;
pub mod chat;
pub mod arc_program;
pub mod arc_apply;
pub mod arc_parser;
pub mod arc_search;
pub mod agi3;

pub use telemetry::{PowerLawObserver, simulate_avalanche};
pub use vision::{PatchEncoder, VisualToken, ImagePatch, generate_test_image, Cifar10Loader, CifarImage, CIFAR10_CLASSES, map_cifar_label_to_lexicon, generate_synthetic_cifar10_batch, EncoderTrainer, SeparationMetrics, load_pretrained_encoder, NmnistSample, NmnistDataset, DvsGestureSample, DvsGestureDataset, load_gesture_train_set, load_gesture_test_set, GESTURE_LABELS, ProjectionLayer, init_class_centers, load_train_set, load_test_set, ArcGrid, ArcTask, ArcDataset, GridEncoder, TransformationAnalyzer, TransformationAnalysis, TransformationVector};
pub use vision::hybrid_solver::{HybridSolver, evaluate_hybrid_solver, Heuristic};
pub use vision::dsl_solver::{Op, Program, find_solving_program, evaluate_solver, SolverResult, infer_color_map, apply_color_map, grids_equal};
pub use baby::{InfomaxReward, UCBExplorer, TransitionalLearner};

#[derive(Debug, Clone)]
pub enum LabError {
    Geometry(String),
    InvalidState,
    DimensionMismatch { expected: usize, got: usize },
}

pub use plasticity::RSTDP;
pub use geometry::{Quaternion, HyperbolicPoint, PoincareBall};
pub use substrate::{SpikeBuffer, NeuronIdx, StateArena, WeightMatrix, ChatArena, batch_euclidean_distances, batch_euclidean_distances_scalar, batch_argmax, dot_product};
#[cfg(feature = "rayon")]
pub use substrate::batch_distances_parallel;
pub use attention::QuaternionAttention;
pub use memory::WorkingMemory;
pub use swarm::neuron::QLIFNeuron;
pub use swarm::snn_core::{SnnCore, SnnStepResult, SnnStateDto, NeuronStateDto, SynapseStateDto, TOTAL_NEURONS, STAGES, NEURONS_PER_STAGE, STAGE_NAMES, STAGE_COLORS};
pub use compression::{GeometricBottleneck, CompressionRouter};
pub use world_model::WorldModel;
pub use rl::{RLAgent, ValueHead, PolicyHead, Transition, StateVector};
pub use semantics::{
    SemanticEncoder, ConceptGraph, HyperbolicContrastive,
    Lexicon, LexiconToken, TokenClass, NoiseInjector,
    SemanticRewardEngine, RewardWeights, RewardSignal,
    TokenComposer, SemanticTrainer,
    SemanticLearner, LearningRates, LearningMetrics, EpochMetrics,
    BridgeEdge, SemanticEdge, RelationType,
};
pub use chat::{
    TokenSpikeEncoder, SpikeTokenDecoder,
    ConversationBuffer, ConversationTurn,
    Thought, ThoughtChain, ReasoningEngine,
    OnlineLearner, WorldChat,
    ChatConfig, WorldGeometry, GeometryError,
    build_extended_lexicon, standard_geometry,
    AvalancheGuidedSelector, AvalancheSelection, build_response_from_selection,
    DvsEvent, DvsEncoder, DvsEncoderConfig, project_dvs_to_histogram, project_dvs_to_time_surface, project_dvs_to_combined_features, project_dvs_to_multiscale_features, normalize_timestamps, histogram_to_hyperbolic,
};

pub use arc_program::{ArcOpCode, ArcOpToken, ArcProgram, serialize_program, deserialize_program};
pub use arc_apply::{apply_arc_op, apply_program, program_solves_train};
pub use arc_parser::{ColorStats, Component, ObjectGraph, extract_components};
pub use arc_search::{SearchConfig, SearchResult, find_program, find_program_with_depth, search_program};

pub use routing::datatype_universal::{DataType, TypeTag, encode_datatype, decode_datatype, data_type_tag, type_tag_name};


