//! Projection Layer — MLP mapping DVS histograms to hyperbolic lexicon space
//!
//! Architecture:
//! - Input: flattened histogram + time-surface (bins*bins*3)
//! - Hidden 1: linear + LeakyReLU (input_dim -> 128)
//! - Hidden 2: linear + LeakyReLU (128 -> 64)
//! - Hidden 3: linear + LeakyReLU (64 -> 32)
//! - Output: linear + L2-normalize (32 -> 8) scaled to target_radius
//!
//! Training:
//! - Radial cross-entropy loss on 8D unit sphere
//! - Direction-only learning (radius is fixed by normalization)

use ndarray::{Array1, Array2};
use rand::Rng;

use crate::vision::NmnistSample;

/// MLP projection layer with direction-only learning in configurable output dimension.
#[derive(Debug, Clone)]
pub struct ProjectionLayer {
    /// Layer 1: input_dim -> 128
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    /// Layer 2: 128 -> 64
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
    /// Layer 3: 64 -> 32
    pub w3: Array2<f32>,
    pub b3: Array1<f32>,
    /// Output layer: 32 -> output_dim (direction only, L2-normalized)
    pub w4: Array2<f32>,
    pub b4: Array1<f32>,
    /// Learning rate
    pub learning_rate: f32,
    /// Available digits and their class indices
    pub available_digits: Vec<u8>,
    /// Target radius for L2-normalization
    pub target_radius: f32,
    /// Output dimension (e.g., 8 for 3-4 classes, 16 for 10 classes)
    pub output_dim: usize,
}

impl ProjectionLayer {
    /// Creates a new MLP projection layer with Xavier initialization.
    pub fn new(input_dim: usize, learning_rate: f32, available_digits: Vec<u8>, output_dim: usize) -> Self {
        let lr = if learning_rate > 0.0 { learning_rate } else { 0.001 };
        let mut rng = rand::thread_rng();
        let output_dim = output_dim.max(1);

        let w1 = Array2::from_shape_fn((128, input_dim), |(_, _)| {
            (rng.r#gen::<f32>() - 0.5) * (2.0f32 / input_dim as f32).sqrt()
        });
        let b1 = Array1::zeros(128);

        let w2 = Array2::from_shape_fn((64, 128), |(_, _)| {
            (rng.r#gen::<f32>() - 0.5) * (2.0f32 / 128.0).sqrt()
        });
        let b2 = Array1::zeros(64);

        let w3 = Array2::from_shape_fn((32, 64), |(_, _)| {
            (rng.r#gen::<f32>() - 0.5) * (2.0f32 / 64.0).sqrt()
        });
        let b3 = Array1::zeros(32);

        let w4 = Array2::from_shape_fn((output_dim, 32), |(_, _)| {
            (rng.r#gen::<f32>() - 0.5) * (2.0f32 / 32.0).sqrt()
        });
        let b4 = Array1::zeros(output_dim);

        Self { w1, b1, w2, b2, w3, b3, w4, b4, learning_rate: lr, available_digits, target_radius: 0.7, output_dim }
    }

    /// Sets the learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Forward pass: histogram -> output_dim direction (L2-normalized, scaled to target_radius).
    pub fn project(&self, histogram: &[f32]) -> Vec<f32> {
        let input = Array1::from_vec(histogram.to_vec());

        // Layer 1: input -> 128 + LeakyReLU
        let z1 = self.w1.dot(&input) + &self.b1;
        let a1 = z1.mapv(|x| if x > 0.0 { x } else { 0.01 * x });

        // Layer 2: 128 -> 64 + LeakyReLU
        let z2 = self.w2.dot(&a1) + &self.b2;
        let a2 = z2.mapv(|x| if x > 0.0 { x } else { 0.01 * x });

        // Layer 3: 64 -> 32 + LeakyReLU
        let z3 = self.w3.dot(&a2) + &self.b3;
        let a3 = z3.mapv(|x| if x > 0.0 { x } else { 0.01 * x });

        // Layer 4: 32 -> output_dim (linear output, no activation)
        let z4 = self.w4.dot(&a3) + &self.b4;
        let raw: Vec<f32> = z4.iter().cloned().collect();

        // L2-normalize and scale to target_radius
        l2_normalize_scale_f32(&raw, self.target_radius)
    }

    /// Trains using radial cross-entropy on output_dim class centers.
    pub fn train_step(
        &mut self,
        histogram: &[f32],
        _target_digit: u8,
        target_index: usize,
        _num_classes: usize,
        class_centers: &[Vec<f64>],
    ) -> f32 {
        let input = Array1::from_vec(histogram.to_vec());
        let _num_classes = _num_classes.max(1);

        // Forward pass
        let z1 = self.w1.dot(&input) + &self.b1;
        let a1 = z1.mapv(|x| if x > 0.0 { x } else { 0.01 * x });
        let z2 = self.w2.dot(&a1) + &self.b2;
        let a2 = z2.mapv(|x| if x > 0.0 { x } else { 0.01 * x });
        let z3 = self.w3.dot(&a2) + &self.b3;
        let a3 = z3.mapv(|x| if x > 0.0 { x } else { 0.01 * x });
        let z4 = self.w4.dot(&a3) + &self.b4;
        let raw: Vec<f64> = z4.iter().map(|&x| x as f64).collect();
        let raw_norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        let output = l2_normalize_scale(&raw, self.target_radius as f64);

        // Radial cross-entropy loss
        let loss = radial_cross_entropy_loss(&output, target_index, class_centers);

        if loss > 1e-9 {
            // Gradient through L2-norm
            let grad_output = radial_cross_entropy_grad(&output, target_index, class_centers, self.target_radius as f64);
            let grad_raw = backward_l2(&grad_output, &output, self.target_radius as f64, raw_norm);

            // Convert back to f32 for weight updates
            let grad_z4: Vec<f32> = grad_raw.iter().map(|&g| g as f32).collect();
            let grad_z4_arr = Array1::from_vec(grad_z4);

            // Layer 4 gradients
            let mut grad_w4 = Array2::zeros((self.output_dim, 32));
            for i in 0..self.output_dim {
                for j in 0..32 {
                    grad_w4[[i, j]] = grad_z4_arr[i] * a3[j];
                }
            }
            let grad_b4 = grad_z4_arr.clone();
            let grad_a3 = self.w4.t().dot(&grad_z4_arr);

            // Layer 3 gradients
            let leaky_relu3 = z3.mapv(|z| if z > 0.0 { 1.0 } else { 0.01 });
            let grad_z3 = grad_a3 * leaky_relu3;
            let mut grad_w3 = Array2::zeros((32, 64));
            for i in 0..32 {
                for j in 0..64 {
                    grad_w3[[i, j]] = grad_z3[i] * a2[j];
                }
            }
            let grad_b3 = grad_z3.clone();
            let grad_a2 = self.w3.t().dot(&grad_z3);

            // Layer 2 gradients
            let leaky_relu2 = z2.mapv(|z| if z > 0.0 { 1.0 } else { 0.01 });
            let grad_z2 = grad_a2 * leaky_relu2;
            let mut grad_w2 = Array2::zeros((64, 128));
            for i in 0..64 {
                for j in 0..128 {
                    grad_w2[[i, j]] = grad_z2[i] * a1[j];
                }
            }
            let grad_b2 = grad_z2.clone();
            let grad_a1 = self.w2.t().dot(&grad_z2);

            // Layer 1 gradients
            let leaky_relu1 = z1.mapv(|z| if z > 0.0 { 1.0 } else { 0.01 });
            let grad_z1 = grad_a1 * leaky_relu1;
            let mut grad_w1 = Array2::zeros((128, input.len()));
            for i in 0..128 {
                for j in 0..input.len() {
                    grad_w1[[i, j]] = grad_z1[i] * input[j];
                }
            }
            let grad_b1 = grad_z1.clone();

            // Update weights
            self.w4 = &self.w4 - &(grad_w4 * self.learning_rate);
            self.b4 = &self.b4 - &(grad_b4 * self.learning_rate);
            self.w3 = &self.w3 - &(grad_w3 * self.learning_rate);
            self.b3 = &self.b3 - &(grad_b3 * self.learning_rate);
            self.w2 = &self.w2 - &(grad_w2 * self.learning_rate);
            self.b2 = &self.b2 - &(grad_b2 * self.learning_rate);
            self.w1 = &self.w1 - &(grad_w1 * self.learning_rate);
            self.b1 = &self.b1 - &(grad_b1 * self.learning_rate);
        }

        loss as f32
    }

    /// Evaluates accuracy on a dataset using combined features.
    pub fn evaluate(&self, dataset: &[NmnistSample], bins: usize, class_centers: &[Vec<f64>]) -> (f32, Vec<(u8, usize, usize)>) {
        let mut correct = 0;
        let mut per_digit: Vec<(u8, usize, usize)> = Vec::new();

        for &digit in &self.available_digits {
            let digit_samples: Vec<_> = dataset.iter().filter(|s| s.digit == digit).collect();
            let total = digit_samples.len();
            let mut digit_correct = 0;

            for sample in digit_samples {
                let histogram = crate::project_dvs_to_combined_features(&sample.events, bins, 50000.0);
                let output = self.project(&histogram);
                let output_f64: Vec<f64> = output.iter().map(|&x| x as f64).collect();

                let mut best_digit = self.available_digits[0];
                let mut best_sim = f64::NEG_INFINITY;

                for (class_idx, &d) in self.available_digits.iter().enumerate() {
                    let sim = cosine_similarity(&output_f64, &class_centers[class_idx]);
                    if sim > best_sim {
                        best_sim = sim;
                        best_digit = d;
                    }
                }

                if best_digit == digit {
                    correct += 1;
                    digit_correct += 1;
                }
            }

            per_digit.push((digit, digit_correct, total));
        }

        let accuracy = correct as f32 / dataset.len() as f32;
        (accuracy, per_digit)
    }
}

// =============================================================================
// Math Helpers
// =============================================================================

/// L2-normalize a vector and scale to target_radius (f32 version).
fn l2_normalize_scale_f32(v: &[f32], radius: f32) -> Vec<f32> {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt().max(1e-12);
    let scale = radius / norm;
    v.iter().map(|x| x * scale).collect()
}

/// L2-normalize a vector and scale to target_radius.
fn l2_normalize_scale(v: &[f64], radius: f64) -> Vec<f64> {
    let norm_sq: f64 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt().max(1e-12);
    let scale = radius / norm;
    v.iter().map(|x| x * scale).collect()
}

/// Cosine similarity between two vectors (assumes both are L2-normalized).
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b).max(1e-12)
}

/// Radial cross-entropy loss for direction-only classification.
fn radial_cross_entropy_loss(
    output: &[f64],
    target: usize,
    centers: &[Vec<f64>],
) -> f64 {
    let similarities: Vec<f64> = centers.iter()
        .map(|c| cosine_similarity(output, c))
        .collect();

    let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sims: Vec<f64> = similarities.iter().map(|s| (s - max_sim).exp()).collect();
    let sum_exp = exp_sims.iter().sum::<f64>();

    -exp_sims[target].ln() + sum_exp.ln()
}

/// Gradient of radial cross-entropy loss with respect to output direction.
fn radial_cross_entropy_grad(
    output: &[f64],
    target: usize,
    centers: &[Vec<f64>],
    radius: f64,
) -> Vec<f64> {
    let similarities: Vec<f64> = centers.iter()
        .map(|c| cosine_similarity(output, c))
        .collect();

    let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sims: Vec<f64> = similarities.iter().map(|s| (s - max_sim).exp()).collect();
    let sum_exp = exp_sims.iter().sum::<f64>();

    let mut grad = vec![0.0; output.len()];
    let radius_sq = radius * radius;
    for i in 0..output.len() {
        for (k, center) in centers.iter().enumerate() {
            let softmax_weight = exp_sims[k] / sum_exp;
            let one_hot = if k == target { 1.0 } else { 0.0 };
            grad[i] += (softmax_weight - one_hot) * center[i] / radius_sq;
        }
    }

    grad
}

/// Backward pass through L2-normalization: project gradient onto tangent space.
fn backward_l2(grad_from_loss: &[f64], output: &[f64], radius: f64, raw_norm: f64) -> Vec<f64> {
    let dot: f64 = output.iter().zip(grad_from_loss).map(|(o, g)| o * g).sum();
    let scale = radius / raw_norm;

    grad_from_loss.iter().zip(output).map(|(&g, &o)| {
        (g - dot * o / (radius * radius)) * scale
    }).collect()
}

/// Initialize class centers on the radius-0.7 ring with maximum angular separation.
///
/// For `n_classes <= dim`, centers are placed on orthogonal axes.
/// For `n_classes > dim`, centers are initialized randomly with a fixed seed
/// to ensure reproducibility while providing good separation.
pub fn init_class_centers(n_classes: usize, dim: usize, radius: f64) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut centers = Vec::new();

    if n_classes <= dim {
        // Orthogonal axis placement for n_classes <= dim
        for k in 0..n_classes {
            let mut center = vec![0.0; dim];
            center[k] = radius;
            centers.push(center);
        }
    } else {
        // Random uniform distribution on sphere for n_classes > dim
        // Use Box-Muller transform for Gaussian samples, then normalize
        for _ in 0..n_classes {
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                let u1: f64 = rng.r#gen::<f64>().max(1e-12);
                let u2: f64 = rng.r#gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                vec.push(z);
            }
            let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            let scaled: Vec<f64> = vec.iter().map(|x| x * radius / norm).collect();
            centers.push(scaled);
        }
    }

    centers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_8d_creates_valid_output() {
        let layer = ProjectionLayer::new(192, 0.01, vec![3, 4, 9], 8);
        let hist = vec![0.5; 192];
        let output = layer.project(&hist);
        assert_eq!(output.len(), 8);
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 0.7).abs() < 1e-5, "norm={}", norm);
    }

    #[test]
    fn l2_normalize_scale_works() {
        let v = vec![3.0, 4.0];
        let result = l2_normalize_scale(&v, 0.7);
        let norm = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 0.7).abs() < 1e-10);
    }

    #[test]
    fn radial_cross_entropy_loss_decreases_with_correct_class() {
        let centers = init_class_centers(3, 8, 0.7);
        let output = centers[0].clone(); // Perfect prediction
        let loss = radial_cross_entropy_loss(&output, 0, &centers);
        assert!(loss < 0.6, "loss should be <0.6 for perfect prediction in 3-class, got {}", loss);
    }
}
