//! Concept Graph — Knowledge as Hyperbolic Graph
//!
//! Meaning emerges from relations. This graph stores concepts as nodes and
//! relations as typed edges in sparse DOD format — directly compatible with
//! the existing GraphSNN substrate.

use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::LabError;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationType {
    IsA,        // Hypernymy (dog → animal)
    HasA,       // Meronymy (dog → tail)
    RelatedTo,  // Association (dog → leash)
    Antonym,    // Opposite (hot → cold)
    Causes,     // Causality (rain → wet)
}

/// A Concept = neuron in the semantic graph
#[derive(Debug, Clone)]
pub struct ConceptNode {
    pub id: usize,
    pub label: String,
    pub embedding: HyperbolicPoint,
}

/// Sparse edge list: DOD-compatible
#[derive(Debug, Clone, Copy)]
pub struct SemanticEdge {
    pub source: usize,
    pub target: usize,
    pub rel: RelationType,
    pub weight: f64,
}

/// Fixed-weight bridge between visual and language modality clusters.
#[derive(Debug, Clone, Copy)]
pub struct BridgeEdge {
    pub visual_cluster: usize,
    pub language_cluster: usize,
    pub weight: f64,
    pub bidirectional: bool,
}

pub struct ConceptGraph {
    pub nodes: Vec<ConceptNode>,
    pub edges: Vec<SemanticEdge>,
    pub ball: PoincareBall,
    /// Index: label → id for fast lookup
    pub index: HashMap<String, usize>,
    /// Cross-modal bridge edges (visual ↔ language)
    pub bridge_edges: Vec<BridgeEdge>,
}

impl ConceptGraph {
    pub fn new(curvature: f64) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            ball: PoincareBall::new(curvature),
            index: HashMap::new(),
            bridge_edges: Vec::new(),
        }
    }

    pub fn add_concept(&mut self, label: &str, embedding: HyperbolicPoint) -> usize {
        let id = self.nodes.len();
        self.nodes.push(ConceptNode {
            id,
            label: label.to_string(),
            embedding,
        });
        self.index.insert(label.to_string(), id);
        id
    }

    pub fn add_edge(
        &mut self,
        from: &str,
        to: &str,
        rel: RelationType,
        weight: f64,
    ) -> Result<(), LabError> {
        let src = self.index.get(from).copied()
            .ok_or_else(|| LabError::InvalidState)?;
        let dst = self.index.get(to).copied()
            .ok_or_else(|| LabError::InvalidState)?;

        self.edges.push(SemanticEdge {
            source: src,
            target: dst,
            rel,
            weight: weight.clamp(-1.0, 1.0),
        });
        Ok(())
    }

    /// Adds a fixed-weight bridge between a visual cluster and a language cluster.
    pub fn add_bridge(&mut self, visual_cluster: usize, language_cluster: usize) {
        self.bridge_edges.push(BridgeEdge {
            visual_cluster,
            language_cluster,
            weight: 0.15,
            bidirectional: true,
        });
    }

    /// Adds a bridge with custom weight and bidirectionality.
    pub fn add_bridge_with(
        &mut self,
        visual_cluster: usize,
        language_cluster: usize,
        weight: f64,
        bidirectional: bool,
    ) {
        self.bridge_edges.push(BridgeEdge {
            visual_cluster,
            language_cluster,
            weight: weight.clamp(0.0, 1.0),
            bidirectional,
        });
    }

    /// Cross-modal propagation: visual cluster → language cluster(s)
    pub fn propagate_visual_to_language(&self, visual_cluster: usize) -> Vec<usize> {
        self.bridge_edges.iter()
            .filter(|b| b.visual_cluster == visual_cluster && b.bidirectional)
            .map(|b| b.language_cluster)
            .collect()
    }

    /// Cross-modal propagation: language cluster → visual cluster(s)
    pub fn propagate_language_to_visual(&self, language_cluster: usize) -> Vec<usize> {
        self.bridge_edges.iter()
            .filter(|b| b.language_cluster == language_cluster && b.bidirectional)
            .map(|b| b.visual_cluster)
            .collect()
    }

    /// Returns all bridge edges for inspection.
    pub fn bridges(&self) -> &[BridgeEdge] {
        &self.bridge_edges
    }

    /// Semantic neighborhood: find similar concepts via hyperbolic distance
    pub fn nearest_neighbors(
        &self,
        query: &HyperbolicPoint,
        k: usize,
    ) -> Result<Vec<(usize, f64)>, LabError> {
        let dim = query.coords.len();
        if dim == 0 || self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Build flat f32 database: all node embeddings concatenated
        let mut database_f32 = Vec::with_capacity(self.nodes.len() * dim);
        for node in &self.nodes {
            for &c in &node.embedding.coords {
                database_f32.push(c as f32);
            }
        }

        // Build flat f32 query
        let query_f32: Vec<f32> = query.coords.iter().map(|&c| c as f32).collect();

        // Use AVX2 batch distance if available
        let distances_f32 = crate::batch_euclidean_distances(&query_f32, &database_f32);

        // Convert back to (usize, f64) pairs
        let mut dists: Vec<(usize, f64)> = self.nodes.iter().enumerate()
            .map(|(i, n)| (n.id, distances_f32.get(i).copied().unwrap_or(f32::INFINITY) as f64))
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k);
        Ok(dists)
    }

    /// Propagation: activate neighbors weighted (Graph-SNN step)
    pub fn propagate(
        &self,
        active: &[bool],
    ) -> Vec<f64> {
        let mut activation = vec![0.0f64; self.nodes.len()];
        for edge in &self.edges {
            if active[edge.source] {
                activation[edge.target] += edge.weight;
            }
        }
        activation
    }

    /// Adds recurrent self-connections to all nodes.
    pub fn add_self_connections(&mut self) {
        for i in 0..self.nodes.len() {
            self.edges.push(SemanticEdge {
                source: i,
                target: i,
                rel: RelationType::RelatedTo,
                weight: 0.3,
            });
        }
    }

    /// Adds random recurrent connections between nodes in the same cluster.
    /// A cluster is defined by the first letter of the label (simple heuristic).
    pub fn add_recurrent_connections(&mut self, density: f64) {
        let mut rng = rand::thread_rng();
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                if rng.r#gen::<f64>() < density {
                    let weight = rng.r#gen::<f64>() * 0.5;
                    self.edges.push(SemanticEdge {
                        source: i,
                        target: j,
                        rel: RelationType::RelatedTo,
                        weight,
                    });
                    self.edges.push(SemanticEdge {
                        source: j,
                        target: i,
                        rel: RelationType::RelatedTo,
                        weight,
                    });
                }
            }
        }
    }

    /// Adds preferential attachment edges to create a scale-free network topology.
    /// Uses a fixed seed for deterministic graph construction.
    pub fn add_preferential_attachment(&mut self, num_edges: usize) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        if self.nodes.is_empty() {
            return;
        }

        let mut out_degree = vec![0usize; self.nodes.len()];
        let mut total_degree = 0usize;

        for _ in 0..num_edges {
            let target = if total_degree == 0 {
                rng.r#gen::<usize>() % self.nodes.len()
            } else {
                let mut r = rng.r#gen::<usize>() % total_degree;
                let mut target = 0;
                for (i, &deg) in out_degree.iter().enumerate() {
                    r = r.saturating_sub(deg);
                    if r == 0 {
                        target = i;
                        break;
                    }
                }
                target
            };

            let source = rng.r#gen::<usize>() % self.nodes.len();
            if source != target {
                let weight = rng.r#gen::<f64>() * 0.5 + 0.1;
                self.edges.push(SemanticEdge {
                    source,
                    target,
                    rel: RelationType::RelatedTo,
                    weight,
                });
                self.edges.push(SemanticEdge {
                    source: target,
                    target: source,
                    rel: RelationType::RelatedTo,
                    weight,
                });
                out_degree[source] += 1;
                out_degree[target] += 1;
                total_degree += 2;
            }
        }
    }

    /// Adds regular ring-lattice edges for uniform branching dynamics.
    /// Each node gets `k` outgoing edges to nodes (i+1), (i+2), ..., (i+k) mod N.
    /// This produces a uniform out-degree for consistent critical branching.
    pub fn add_regular_edges(&mut self, k: usize) {
        let n = self.nodes.len();
        if n == 0 { return; }
        for i in 0..n {
            for j in 1..=k {
                let target = (i + j) % n;
                if target != i {
                    self.edges.push(SemanticEdge {
                        source: i,
                        target,
                        rel: RelationType::RelatedTo,
                        weight: 0.5,
                    });
                }
            }
        }
    }

    /// Adds random directed edges to produce a uniform out-degree K Erdős-Rényi graph.
    /// Each node gets exactly K random outgoing edges (no self-loops).
    /// This produces uniform branching dynamics ideal for criticality testing.
    pub fn add_random_edges(&mut self, k: usize) {
        let n = self.nodes.len();
        if n == 0 { return; }
        let mut rng = rand::thread_rng();
        for i in 0..n {
            let mut targets: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            targets.shuffle(&mut rng);
            for &target in targets.iter().take(k) {
                self.edges.push(SemanticEdge {
                    source: i,
                    target,
                    rel: RelationType::RelatedTo,
                    weight: 0.5,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_graph() -> ConceptGraph {
        let mut g = ConceptGraph::new(1.0);
        let e1 = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
        let e2 = HyperbolicPoint::new(array![0.11, 0.01]).unwrap();
        let e3 = HyperbolicPoint::new(array![0.5, 0.0]).unwrap();

        g.add_concept("hund", e1);
        g.add_concept("katze", e2);
        g.add_concept("tier", e3);

        g.add_edge("hund", "tier", RelationType::IsA, 0.8).unwrap();
        g.add_edge("katze", "tier", RelationType::IsA, 0.8).unwrap();
        g
    }

    #[test]
    fn taxonomy_links_work() {
        let g = make_graph();
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn nearest_neighbor_finds_related() {
        let g = make_graph();
        let q = HyperbolicPoint::new(array![0.105, 0.005]).unwrap();
        let nn = g.nearest_neighbors(&q, 2).unwrap();
        assert_eq!(nn.len(), 2);
    }

    #[test]
    fn propagation_spreads_activation() {
        let g = make_graph();
        let mut active = vec![false; 3];
        active[0] = true; // dog active
        let act = g.propagate(&active);
        assert!(act[2] > 0.5); // animal should be activated
    }
}
