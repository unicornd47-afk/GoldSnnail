//! Vision — Patch Extraction, Visual Encoding, and Multimodal Binding
//!
//! This module bridges raw 2D images and abstract semantic concepts:
//! - `patch_encoder`: Extracts patches, encodes to quaternion/hyperbolic embeddings
//! - `cifar10`: Loads real CIFAR-10 binaries or generates synthetic images
//! - `pretrain_encoder`: Contrastive pre-training for PatchEncoder weights
//! - Visual tokens bind to semantic labels via the existing `SemanticEncoder`

pub mod patch_encoder;
pub mod cifar10;
pub mod pretrain_encoder;
pub mod nmnist_loader;
pub mod dvs_gesture_loader;
pub mod projection_layer;
pub mod arc_loader;
pub mod grid_encoder;
pub mod object_descriptor;
pub mod transform_codec;
pub mod transform_memory;
pub mod committee;
pub mod transformation_analyzer;
pub mod hybrid_solver;
pub mod dsl_solver;

pub use patch_encoder::{PatchEncoder, VisualToken, ImagePatch, generate_test_image};
pub use cifar10::{Cifar10Loader, CifarImage, CIFAR10_CLASSES, map_cifar_label_to_lexicon, generate_synthetic_cifar10_batch};
pub use pretrain_encoder::{EncoderTrainer, SeparationMetrics, load_pretrained_encoder};
pub use nmnist_loader::{NmnistSample, NmnistDataset, load_train_set, load_test_set};
pub use dvs_gesture_loader::{DvsGestureSample, DvsGestureDataset, load_train_set as load_gesture_train_set, load_test_set as load_gesture_test_set, GESTURE_LABELS};
pub use projection_layer::{ProjectionLayer, init_class_centers};
pub use arc_loader::{ArcGrid, ArcTask, ArcDataset};
pub use grid_encoder::GridEncoder;
pub use object_descriptor::{hu_moments, ObjectDescriptor};
pub use transform_codec::{
    apply_d4, extract_transform, apply_transform, find_color_map, find_d4, find_tiling, similarity_fit, is_10x10,
    SimilarityFit, TransformCode, TransformKind, TransformParams,
};
pub use transform_memory::{build_memory_from_tasks, TransformMemory};
pub use committee::{Committee, VoterPrediction};
pub use transformation_analyzer::{TransformationAnalyzer, TransformationAnalysis, TransformationVector};