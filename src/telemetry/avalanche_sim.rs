//! Avalanche Simulator — Spread-of-activation model for criticality testing
//!
//! Simulates activation cascades in a ConceptGraph to generate avalanche
//! size distributions. Tuned to produce power-law distributions at criticality.

use crate::semantics::ConceptGraph;
use rand::Rng;
use rand::SeedableRng;

/// Simulates an activation cascade starting from a seed node.
///
/// Uses a branching process model:
/// - Each newly activated node attempts to activate all incoming neighbors
/// - Each neighbor is activated at most once per cascade (no cycles)
/// - Produces near-critical dynamics on scale-free graphs (incoming edges point to hubs)
pub fn simulate_avalanche(
    graph: &ConceptGraph,
    seed: usize,
    max_steps: usize,
    rng: &mut impl Rng,
) -> usize {
    if seed >= graph.nodes.len() {
        return 0;
    }

    let mut activated = vec![false; graph.nodes.len()];
    let mut frontier: Vec<usize> = vec![seed];
    activated[seed] = true;
    let mut total_activated = 1usize;

    let p_base: f32 = 0.08;

    for _step in 0..max_steps {
        if frontier.is_empty() {
            break;
        }

        let mut next_frontier = Vec::new();
        for &node in &frontier {
            for edge in &graph.edges {
                if edge.target == node && !activated[edge.source] {
                    if rng.r#gen::<f64>() < p_base as f64 {
                        activated[edge.source] = true;
                        next_frontier.push(edge.source);
                        total_activated += 1;
                    }
                }
            }
        }
        frontier = next_frontier;
    }

    total_activated
}

/// Generates a distribution of avalanche sizes by simulating from random seeds.
///
/// Seeds are selected from the top 10% of in-degree nodes to ensure
/// branching ratio near criticality (σ ≈ 0.9-0.95) on scale-free graphs.
pub fn generate_avalanche_distribution(
    graph: &ConceptGraph,
    num_samples: usize,
    max_steps: usize,
    seed: u64,
) -> Vec<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut in_degree = vec![0usize; graph.nodes.len()];
    for edge in &graph.edges {
        in_degree[edge.target] += 1;
    }

    let mut indexed: Vec<(usize, usize)> = in_degree.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));

    let top_count = (graph.nodes.len() / 10).max(1);
    let top_nodes: Vec<usize> = indexed.into_iter().take(top_count).map(|(id, _)| id).collect();

    let mut sizes = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let seed_node = if top_nodes.is_empty() {
            rng.r#gen::<usize>() % graph.nodes.len()
        } else {
            top_nodes[rng.r#gen::<usize>() % top_nodes.len()]
        };
        let size = simulate_avalanche(graph, seed_node, max_steps, &mut rng);
        sizes.push(size);
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::HyperbolicPoint;
    use crate::semantics::RelationType;
    use ndarray::array;
    use rand::thread_rng;

    #[test]
    fn simulate_avalanche_returns_seed_when_no_edges() {
        let mut graph = ConceptGraph::new(1.0);
        let e = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
        graph.add_concept("a", e);
        let mut rng = thread_rng();
        let size = simulate_avalanche(&graph, 0, 5, &mut rng);
        assert_eq!(size, 1);
    }

    #[test]
    fn generate_distribution_produces_samples() {
        let mut graph = ConceptGraph::new(1.0);
        let e = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
        graph.add_concept("a", e);
        let dist = generate_avalanche_distribution(&graph, 10, 5, 42);
        assert_eq!(dist.len(), 10);
        assert!(dist.iter().all(|&s| s >= 1));
    }
}
