//! Unsupervised k-Means Clustering of ARC Transformation Vectors
//!
//! Loads ARC tasks, computes transformation vectors for each train pair,
//! projects them to tangent space at origin, and runs k-Means clustering
//! with silhouette scoring to find the optimal number of clusters.

use goldsnnail::ArcDataset;
use goldsnnail::vision::grid_encoder::GridEncoder;
use goldsnnail::vision::transformation_analyzer::{TransformationAnalyzer, TransformationVector};

// ============================================================================
// Distance & Clustering Primitives
// ============================================================================

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vector_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize_vector(v: &mut Vec<f64>) {
    let norm = vector_norm(v);
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Simple k-Means with Euclidean distance.
struct KMeans {
    k: usize,
    max_iter: usize,
    tolerance: f64,
}

impl KMeans {
    fn new(k: usize, max_iter: usize, tolerance: f64) -> Self {
        Self { k, max_iter, tolerance }
    }

    fn fit(&self, data: &[Vec<f64>], rng: &mut impl rand::Rng) -> (Vec<Vec<f64>>, Vec<usize>) {
        let n = data.len();
        if n == 0 || self.k == 0 {
            return (Vec::new(), Vec::new());
        }
        let dim = data[0].len();
        let k = self.k.min(n);

        let mut centroids: Vec<Vec<f64>> = (0..k)
            .map(|_| data[rng.gen_range(0..n)].clone())
            .collect();

        let mut assignments = vec![0usize; n];

        for _ in 0..self.max_iter {
            let mut changed = false;

            for (i, point) in data.iter().enumerate() {
                let mut best = 0;
                let mut best_dist = euclidean_distance(point, &centroids[0]);
                for c in 1..k {
                    let d = euclidean_distance(point, &centroids[c]);
                    if d < best_dist {
                        best_dist = d;
                        best = c;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, point) in data.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..dim {
                    new_centroids[c][j] += point[j];
                }
            }

            let mut max_shift = 0.0;
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f64;
                    }
                    let shift = euclidean_distance(&centroids[c], &new_centroids[c]);
                    if shift > max_shift {
                        max_shift = shift;
                    }
                }
            }

            centroids = new_centroids;

            if max_shift < self.tolerance {
                break;
            }
        }

        (centroids, assignments)
    }
}

// ============================================================================
// Silhouette Score
// ============================================================================

fn compute_silhouette_score(
    data: &[Vec<f64>],
    assignments: &[usize],
    centroids: &[Vec<f64>],
) -> f64 {
    let n = data.len();
    if n <= 1 || centroids.is_empty() {
        return 0.0;
    }

    let mut total_sil = 0.0;

    for i in 0..n {
        let c_i = assignments[i];

        let mut a = 0.0;
        let mut a_count = 0;

        let mut b = f64::INFINITY;

        for c in 0..centroids.len() {
            if c == c_i {
                continue;
            }
            let mut cluster_sum = 0.0;
            let mut cluster_count = 0;
            for j in 0..n {
                if assignments[j] == c {
                    cluster_sum += euclidean_distance(&data[i], &data[j]);
                    cluster_count += 1;
                }
            }
            if cluster_count > 0 {
                let avg_dist = cluster_sum / cluster_count as f64;
                if avg_dist < b {
                    b = avg_dist;
                }
            }
        }

        for j in 0..n {
            if assignments[j] == c_i && i != j {
                a += euclidean_distance(&data[i], &data[j]);
                a_count += 1;
            }
        }

        let a_val = if a_count > 0 { a / a_count as f64 } else { 0.0 };

        let s = if b.is_infinite() { 0.0 } else { (b - a_val) / a_val.max(b).max(1e-12) };
        total_sil += s;
    }

    total_sil / n as f64
}

// ============================================================================
// Cluster Statistics
// ============================================================================

#[derive(Debug, Clone)]
struct ClusterStats {
    pub id: usize,
    pub centroid: Vec<f64>,
    pub member_count: usize,
    pub max_radius: f64,
    pub avg_intra_distance: f64,
}

fn compute_cluster_stats(
    data: &[Vec<f64>],
    assignments: &[usize],
    centroids: &[Vec<f64>],
    k: usize,
) -> Vec<ClusterStats> {
    let mut stats = Vec::with_capacity(k);

    for c in 0..k {
        let members: Vec<usize> = assignments.iter().enumerate()
            .filter(|&(_, &a)| a == c)
            .map(|(i, _)| i)
            .collect();

        let member_count = members.len();
        if member_count == 0 {
            stats.push(ClusterStats {
                id: c,
                centroid: centroids[c].clone(),
                member_count: 0,
                max_radius: 0.0,
                avg_intra_distance: 0.0,
            });
            continue;
        }

        let mut max_radius = 0.0;
        let mut total_dist = 0.0;
        let mut pair_count = 0;

        for &i in &members {
            let d = euclidean_distance(&data[i], &centroids[c]);
            if d > max_radius {
                max_radius = d;
            }
            for &j in &members {
                if i < j {
                    total_dist += euclidean_distance(&data[i], &data[j]);
                    pair_count += 1;
                }
            }
        }

        let avg_intra = if pair_count > 0 { total_dist / pair_count as f64 } else { 0.0 };

        stats.push(ClusterStats {
            id: c,
            centroid: centroids[c].clone(),
            member_count,
            max_radius,
            avg_intra_distance: avg_intra,
        });
    }

    stats
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== GoldSnnail Transformation Clustering ===\n");

    let dataset = match ArcDataset::load_from_directory("data/arc") {
        Ok(ds) => ds,
        Err(e) => {
            println!("Failed to load ARC data: {}", e);
            println!("Download with: git clone https://github.com/fchollet/ARC.git data/arc");
            return;
        }
    };

    let total_tasks = dataset.tasks.len();
    println!("Loaded {} tasks", total_tasks);

    if total_tasks < 2 {
        println!("Need at least 2 tasks for clustering.");
        return;
    }

    let mut encoder = GridEncoder::new(100, 32, 16, 0.75);
    println!("Encoder: 100 -> 32 -> 16 (target_radius={})\n", encoder.target_radius);

    println!("Computing transformation vectors...");
    let mut vectors: Vec<TransformationVector> = Vec::new();
    let mut skipped = 0;

    let analyzer = TransformationAnalyzer::new(&encoder);

    for task in &dataset.tasks {
        if task.train_pairs.is_empty() {
            skipped += 1;
            continue;
        }

        for (input, output) in &task.train_pairs {
            let t_type = GridEncoder::classify_transformation(input, output);
            match analyzer.compute_vector(&task.id, input, output, t_type) {
                Ok(vec) => vectors.push(vec),
                Err(e) => {
                    eprintln!("  Encode failed for task {}: {}", task.id, e);
                    skipped += 1;
                }
            }
        }
    }

    println!("Computed {} transformation vectors ({} skipped)\n", vectors.len(), skipped);

    if vectors.len() < 2 {
        println!("Need at least 2 valid vectors for clustering.");
        return;
    }

    let mut rng = rand::thread_rng();

    let mut tangent_space: Vec<Vec<f64>> = vectors.iter().map(|v| v.vector.clone()).collect();

    let dim = tangent_space[0].len();

    println!("Tangent space dimension: {}", dim);

    println!("\n=== k-Means Clustering (k = 2..20) ===");
    println!("{:<6} {:>14} {:>12}", "k", "Silhouette", "Time (ms)");
    println!("{}", "-".repeat(36));

    let mut best_k = 2;
    let mut best_score = -1.0;
    let mut best_centroids = Vec::new();
    let mut best_assignments = Vec::new();

    for k in 2..=20 {
        let km = KMeans::new(k, 100, 1e-6);
        let start = std::time::Instant::now();
        let (centroids, assignments) = km.fit(&tangent_space, &mut rng);
        let elapsed = start.elapsed().as_millis();

        let score = compute_silhouette_score(&tangent_space, &assignments, &centroids);
        println!("{:<6} {:>14.4} {:>12}", k, score, elapsed);

        if score > best_score {
            best_score = score;
            best_k = k;
            best_centroids = centroids;
            best_assignments = assignments;
        }
    }

    println!("\n=== Optimal Clustering ===");
    println!("Optimal k: {}", best_k);
    println!("Silhouette score: {:.4}", best_score);

    let stats = compute_cluster_stats(&tangent_space, &best_assignments, &best_centroids, best_k);

    let mut sorted_stats = stats.clone();
    sorted_stats.sort_by(|a, b| b.member_count.cmp(&a.member_count));

    println!("\n=== Top 5 Clusters by Size ===");
    println!(
        "{:<6} {:>12} {:>12} {:>14} {:>14}",
        "Cluster", "Members", "Max Radius", "Avg Intra Dist", "Centroid Norm"
    );
    println!("{}", "-".repeat(62));

    for stat in sorted_stats.iter().take(5) {
        let centroid_norm = vector_norm(&stat.centroid);
        println!(
            "{:<6} {:>12} {:>12.4} {:>14.4} {:>14.4}",
            stat.id,
            stat.member_count,
            stat.max_radius,
            stat.avg_intra_distance,
            centroid_norm
        );
    }

    let total_members: usize = stats.iter().map(|s| s.member_count).sum();
    let singleton_count = stats.iter().filter(|s| s.member_count == 1).count();
    println!("\nTotal clustered vectors: {}", total_members);
    println!("Singleton clusters: {}", singleton_count);
    println!("Multi-member clusters: {}", stats.len() - singleton_count);

    if best_score > 0.5 {
        println!("\n✅ Strong clustering structure found (silhouette > 0.5)");
    } else if best_score > 0.25 {
        println!("\n⚠️ Weak clustering structure (0.25 < silhouette <= 0.5)");
    } else {
        println!("\n❌ No clear clustering structure (silhouette <= 0.25)");
    }
}
