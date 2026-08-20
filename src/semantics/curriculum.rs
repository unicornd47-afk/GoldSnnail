//! Semantic Curriculum — Training Data Generation
//!
//! Generates synthetic training data for semantic acquisition:
//! Level 1: Simple taxonomy (Is-A hierarchy)
//! Level 2: Sequential associations (sentence fragments)
//! Level 3: Contrastive pairs (anchor, positive, negative)

use crate::semantics::{ConceptGraph, RelationType};
use crate::geometry::HyperbolicPoint;
use ndarray::array;

/// Generates synthetic training data for semantics
pub struct SemanticCurriculum;

impl SemanticCurriculum {
    /// Level 1: Simple taxonomy (Is-A hierarchy)
    pub fn level1_taxonomy() -> ConceptGraph {
        let mut g = ConceptGraph::new(1.0);

        let concepts = [
            ("tier", array![0.0, 0.0]),
            ("hund", array![0.1, 0.0]),
            ("katze", array![0.1, 0.05]),
            ("säugetier", array![0.05, 0.0]),
            ("vogel", array![0.08, 0.1]),
            ("haustier", array![0.12, 0.02]),
        ];

        for (label, coords) in concepts {
            let pt = HyperbolicPoint::new(coords).unwrap();
            g.add_concept(label, pt);
        }

        g.add_edge("hund", "tier", RelationType::IsA, 0.9).unwrap();
        g.add_edge("katze", "tier", RelationType::IsA, 0.9).unwrap();
        g.add_edge("hund", "säugetier", RelationType::IsA, 0.9).unwrap();
        g.add_edge("katze", "säugetier", RelationType::IsA, 0.9).unwrap();
        g.add_edge("vogel", "tier", RelationType::IsA, 0.9).unwrap();
        g.add_edge("hund", "haustier", RelationType::IsA, 0.7).unwrap();
        g.add_edge("katze", "haustier", RelationType::IsA, 0.7).unwrap();

        g
    }

    /// Level 2: Sequential associations (sentence fragments)
    pub fn level2_sequences() -> Vec<Vec<String>> {
        vec![
            vec!["der".into(), "hund".into(), "läuft".into()],
            vec!["die".into(), "katze".into(), "springt".into()],
            vec!["ein".into(), "tier".into(), "atmet".into()],
            vec!["der".into(), "hund".into(), "bellt".into()],
        ]
    }

    /// Positive/Negative pairs for contrastive learning
    pub fn contrastive_pairs() -> Vec<(String, String, String)> {
        // (anchor, positive, negative)
        vec![
            ("hund".into(), "hündisch".into(), "tisch".into()),
            ("katze".into(), "kätzchen".into(), "auto".into()),
            ("tier".into(), "lebewesen".into(), "stein".into()),
        ]
    }
}
