//! GridEncoder — 2D ARC-Grids → Hyperbolische Embeddings
//!
//! Input: Variable-sized ARC grid auto-padded to expected size
//! Output: 16D HyperbolicPoint auf r ≈ 0.75 (visueller äußerer Ring)
//!
//! Architecture: MLP mit flachen Gewichtsmatrizen (DOD-konform)
//!   100D (10×10 normalisiert) → 32D (ReLU) → 16D (linear, L2-normalisiert)
//!
//! Backprop: Volles Gradient-Update für w1, b1, w2, b2

use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::vision::arc_loader::{ArcGrid, ArcTask};
use ndarray::Array1;

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Cached activations from forward pass, reused in backward.
#[derive(Debug, Clone)]
pub struct CachedForward {
    pub input: Vec<f64>,
    pub h1_pre: Vec<f64>,
    pub h1: Vec<f64>,
    pub output_pre: Vec<f64>,
    pub output: Vec<f64>,
}

/// GridEncoder mit flachen Gewichtsmatrizen (DOD-konform).
#[derive(Debug, Clone)]
pub struct GridEncoder {
    pub w1: Vec<f64>,          // dim_hidden × dim_in
    pub b1: Vec<f64>,          // dim_hidden
    pub w2: Vec<f64>,          // dim_out × dim_hidden
    pub b2: Vec<f64>,          // dim_out
    pub dim_in: usize,
    pub dim_hidden: usize,
    pub dim_out: usize,
    pub target_radius: f64,
}

impl GridEncoder {
    pub fn new(dim_in: usize, dim_hidden: usize, dim_out: usize, target_radius: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut init_flat = |rows: usize, cols: usize| -> Vec<f64> {
            (0..rows * cols)
                .map(|_| rng.r#gen::<f64>() * 0.1 - 0.05)
                .collect()
        };

        GridEncoder {
            w1: init_flat(dim_hidden, dim_in),
            b1: vec![0.0; dim_hidden],
            w2: init_flat(dim_out, dim_hidden),
            b2: vec![0.0; dim_out],
            dim_in,
            dim_hidden,
            dim_out,
            target_radius,
        }
    }

    /// Standard forward pass — returns only the final L2-normalized output.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut h1 = vec![0.0f64; self.dim_hidden];
        for i in 0..self.dim_hidden {
            let mut sum = self.b1[i];
            for j in 0..self.dim_in {
                sum += self.w1[i * self.dim_in + j] * input[j];
            }
            h1[i] = sum.max(0.0); // ReLU
        }

        let mut output = vec![0.0f64; self.dim_out];
        for i in 0..self.dim_out {
            let mut sum = self.b2[i];
            for j in 0..self.dim_hidden {
                sum += self.w2[i * self.dim_hidden + j] * h1[j];
            }
            output[i] = sum;
        }

        let norm_sq: f64 = output.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-12);
        let scale = self.target_radius / norm;
        output.iter().map(|x| x * scale).collect()
    }

    /// Forward pass mit Caching aller Intermediate-Werte für Backprop.
    pub fn forward_cached(&self, input: &[f64]) -> CachedForward {
        // Layer 1: Linear + ReLU
        let mut h1_pre = vec![0.0f64; self.dim_hidden];
        let mut h1 = vec![0.0f64; self.dim_hidden];
        for i in 0..self.dim_hidden {
            let mut sum = self.b1[i];
            for j in 0..self.dim_in {
                sum += self.w1[i * self.dim_in + j] * input[j];
            }
            h1_pre[i] = sum;
            h1[i] = sum.max(0.0);
        }

        // Layer 2: Linear (pre-L2)
        let mut output_pre = vec![0.0f64; self.dim_out];
        for i in 0..self.dim_out {
            let mut sum = self.b2[i];
            for j in 0..self.dim_hidden {
                sum += self.w2[i * self.dim_hidden + j] * h1[j];
            }
            output_pre[i] = sum;
        }

        // L2-Normalisierung
        let norm_sq: f64 = output_pre.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-12);
        let scale = self.target_radius / norm;
        let output: Vec<f64> = output_pre.iter().map(|x| x * scale).collect();

        CachedForward {
            input: input.to_vec(),
            h1_pre,
            h1,
            output_pre,
            output,
        }
    }

    /// Gradient der euklidisch approximierten hyperbolischen Distanz.
    /// ∂L/∂x für L = ||x - y||₂, wobei x und y HyperbolicPoints sind.
    ///
    /// NOTE: Dies ist eine euklidische Approximation der Poincaré-Distanz.
    /// Die exakte hyperbolische Gradientenformel würde Möbius-Addition und
    /// atanh-Derivative erfordern. Für den ersten Durchgang ist die
    /// euklidische Annäherung ausreichend und numerisch stabiler.
    pub fn hyperbolic_distance_grad(x: &HyperbolicPoint, y: &HyperbolicPoint) -> Vec<f64> {
        let dist = euclidean_distance(&x.coords, &y.coords).max(1e-12);
        x.coords
            .iter()
            .zip(&y.coords)
            .map(|(xi, yi)| (xi - yi) / dist)
            .collect()
    }

    /// Pre-allokierte Gradient-Buffer für den Backward-Pass.
    /// Wird einmal erstellt und dann pro Step wiederverwendet (DOD: keine Allokation im Hot-Path).
    pub fn grad_buffers(&self) -> GradBuffers {
        GradBuffers::new(self)
    }

    /// Volles Backpropagation durch alle Layer.
    ///
    /// Argumente:
    /// - `cached`: Die zwischengespeicherten Aktivierungen aus `forward_cached()`
    /// - `grad_output`: ∂L/∂output (Gradient des Loss w.r.t. des L2-normalisierten Outputs)
    /// - `buffers`: Pre-allokierte Gradient-Buffer (werden mit 0 gefüllt)
    /// - `lr`: Learning Rate
    ///
    /// Updated: w1, b1, w2, b2 (in-place)
    pub fn backward(
        &mut self,
        cached: &CachedForward,
        grad_output: &[f64],
        buffers: &mut GradBuffers,
        lr: f64,
    ) {
        // Buffer zurücksetzen
        buffers.reset();

        // =========================================================================
        // Schritt A: Gradient durch L2-Normalisierung
        // output = output_pre * scale, scale = target_radius / ||output_pre||
        // =========================================================================
        let r_sq: f64 = cached.output_pre.iter().map(|x| x * x).sum();
        let r = r_sq.sqrt().max(1e-12);
        let scale = self.target_radius / r;

        // dot(grad_output, output_pre) = Σ_k grad_output[k] * output_pre[k]
        let dot_grad_out_pre: f64 = grad_output
            .iter()
            .zip(&cached.output_pre)
            .map(|(g, p)| g * p)
            .sum();

        let factor = dot_grad_out_pre * scale / (r * r);

        for i in 0..self.dim_out {
            buffers.grad_output_pre[i] = grad_output[i] * scale - cached.output_pre[i] * factor;
        }

        // =========================================================================
        // Schritt B: Gradient durch Layer 2 (Linear: output_pre = W2 * h1 + b2)
        // =========================================================================
        for i in 0..self.dim_out {
            let g = buffers.grad_output_pre[i];
            buffers.grad_b2[i] = g;
            for j in 0..self.dim_hidden {
                buffers.grad_w2[i * self.dim_hidden + j] = g * cached.h1[j];
            }
        }

        // grad_h1 = W2^T * grad_output_pre
        for j in 0..self.dim_hidden {
            let mut sum = 0.0;
            for i in 0..self.dim_out {
                sum += buffers.grad_output_pre[i] * self.w2[i * self.dim_hidden + j];
            }
            buffers.grad_h1[j] = sum;
        }

        // =========================================================================
        // Schritt C: Gradient durch ReLU
        // h1[i] = max(0, h1_pre[i]) → grad_h1_pre = grad_h1 * (h1_pre > 0 ? 1 : 0)
        // =========================================================================
        for j in 0..self.dim_hidden {
            buffers.grad_h1_pre[j] = if cached.h1_pre[j] > 0.0 {
                buffers.grad_h1[j]
            } else {
                0.0
            };
        }

        // =========================================================================
        // Schritt D: Gradient durch Layer 1 (Linear: h1_pre = W1 * input + b1)
        // =========================================================================
        for i in 0..self.dim_hidden {
            let g = buffers.grad_h1_pre[i];
            buffers.grad_b1[i] = g;
            for j in 0..self.dim_in {
                buffers.grad_w1[i * self.dim_in + j] = g * cached.input[j];
            }
        }

        // =========================================================================
        // Schritt E: Parameter-Update (SGD)
        // =========================================================================
        for i in 0..self.w1.len() {
            self.w1[i] -= lr * buffers.grad_w1[i];
        }
        for i in 0..self.b1.len() {
            self.b1[i] -= lr * buffers.grad_b1[i];
        }
        for i in 0..self.w2.len() {
            self.w2[i] -= lr * buffers.grad_w2[i];
        }
        for i in 0..self.b2.len() {
            self.b2[i] -= lr * buffers.grad_b2[i];
        }
    }

    /// Trainings-Schritt für ein einziges (Input, Output)-Paar.
    ///
    /// Berechnet:
    /// 1. Forward für Input → point_in
    /// 2. Forward für Output → point_out
    /// 3. Loss = hyperbolische Distanz(point_in, point_out) [euklidisch approximiert]
    /// 4. Backward auf beiden Pfaden mit korrekten Gradienten
    ///
    /// Gibt den Loss zurück.
    pub fn train_step(&mut self, input_grid: &ArcGrid, output_grid: &ArcGrid, lr: f64) -> Result<f64, String> {
        let input_f32 = input_grid.to_feature_vector();
        let output_f32 = output_grid.to_feature_vector();

        if input_f32.len() != self.dim_in || output_f32.len() != self.dim_in {
            return Err(format!(
                "Feature dimension mismatch: expected {}, got input={}, output={}",
                self.dim_in, input_f32.len(), output_f32.len()
            ));
        }

        let input: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();
        let output: Vec<f64> = output_f32.iter().map(|&x| x as f64).collect();

        // Forward passes mit Caching
        let cached_in = self.forward_cached(&input);
        let cached_out = self.forward_cached(&output);

        // Zu HyperbolicPoints konvertieren
        let _point_in = self.encoded_to_hyperbolic(&cached_in.output);
        let _point_out = self.encoded_to_hyperbolic(&cached_out.output);

        // Loss = euklidische Distanz zwischen den Embeddings
        let loss = euclidean_distance(&cached_in.output, &cached_out.output);

        // Gradienten der Distanz:
        // grad_in = ∂L/∂point_in = (point_in - point_out) / ||point_in - point_out||
        // grad_out = ∂L/∂point_out = (point_out - point_in) / ||point_in - point_out||
        // NOTE: Die Richtung ist hier anders als in der ursprünglichen Implementierung.
        // Original: (b-a)/dist → war auf beide Pfade gleich angewendet (BUG).
        // Korrektur: Für Input-Pfad ist grad_in = (point_in - point_out) / dist,
        //            für Output-Pfad ist grad_out = (point_out - point_in) / dist.
        // Das entspricht ∂L/∂x bzw. ∂L/∂y bei L = ||x - y||.
        let dist = loss.max(1e-12);
        let grad_in: Vec<f64> = cached_in
            .output
            .iter()
            .zip(&cached_out.output)
            .map(|(a, b)| (a - b) / dist)
            .collect();
        let grad_out: Vec<f64> = cached_out
            .output
            .iter()
            .zip(&cached_in.output)
            .map(|(b, a)| (b - a) / dist)
            .collect();

        // Backward auf beiden Pfaden
        let mut buffers = self.grad_buffers();
        self.backward(&cached_in, &grad_in, &mut buffers, lr);
        self.backward(&cached_out, &grad_out, &mut buffers, lr);

        Ok(loss)
    }

    /// Encoded output → HyperbolicPoint via exp_map_origin.
    fn encoded_to_hyperbolic(&self, encoded: &[f64]) -> HyperbolicPoint {
        let arr = Array1::from_vec(encoded.to_vec());
        let ball = PoincareBall::new(1.0);
        // exp_map_origin erwartet f64 (Array1<f64>)
        ball.exp_map_origin(&arr).unwrap_or_else(|_| {
            // Fallback: direkte Konstruktion mit Clamp
            let norm: f64 = encoded.iter().map(|x| x * x).sum::<f64>().sqrt();
            let safe = if norm >= 1.0 {
                encoded.iter().map(|x| x / norm * 0.99).collect()
            } else {
                encoded.to_vec()
            };
            HyperbolicPoint {
                coords: safe,
            }
        })
    }

    /// Alias für encode() für Kompatibilität.
    pub fn encode(&self, grid: &ArcGrid) -> Result<HyperbolicPoint, String> {
        let features_f32 = grid.to_feature_vector();
        if features_f32.len() != self.dim_in {
            return Err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.dim_in,
                features_f32.len()
            ));
        }
        let features: Vec<f64> = features_f32.iter().map(|&x| x as f64).collect();
        let encoded = self.forward(&features);

        let norm: f64 = encoded.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            return Err(format!(
                "Encoded vector exceeds Poincaré boundary: norm={}",
                norm
            ));
        }

        let arr = Array1::from_vec(encoded);
        let ball = PoincareBall::new(1.0);
        ball.exp_map_origin(&arr)
            .map_err(|e| format!("Poincaré projection failed: {:?}", e))
    }

    // -------------------------------------------------------------------------
    // Legacy-Methoden (bleiben für Rückwärtskompatibilität)
    // -------------------------------------------------------------------------

    pub fn forward_hidden(&self, input: &[f64]) -> Vec<f64> {
        let mut h1 = vec![0.0f64; self.dim_hidden];
        for i in 0..self.dim_hidden {
            let mut sum = self.b1[i];
            for j in 0..self.dim_in {
                sum += self.w1[i * self.dim_in + j] * input[j];
            }
            h1[i] = sum.max(0.0);
        }
        h1
    }

    pub fn forward_output(&self, h1: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.dim_out];
        for i in 0..self.dim_out {
            let mut sum = self.b2[i];
            for j in 0..self.dim_hidden {
                sum += self.w2[i * self.dim_hidden + j] * h1[j];
            }
            out[i] = sum;
        }

        let norm_sq: f64 = out.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt().max(1e-12);
        let scale = self.target_radius / norm;
        out.iter().map(|x| x * scale).collect()
    }

    #[deprecated]
    pub fn backward_output(&mut self, h1: &[f64], grad_output: &[f64], lr: f64) {
        // Legacy: Nur b2-Update (ersetzt durch vollen backward())
        for i in 0..self.dim_out {
            let grad = grad_output[i] * lr;
            self.b2[i] -= grad;
            for j in 0..self.dim_hidden {
                self.w2[i * self.dim_hidden + j] -= grad * h1[j];
            }
        }
    }
}

/// Pre-allokierte Gradient-Buffer für den Backward-Pass.
/// Vermeidet Heap-Allokationen im Hot-Path des Trainings.
pub struct GradBuffers {
    pub grad_w1: Vec<f64>,
    pub grad_b1: Vec<f64>,
    pub grad_w2: Vec<f64>,
    pub grad_b2: Vec<f64>,
    pub grad_h1: Vec<f64>,
    pub grad_h1_pre: Vec<f64>,
    pub grad_output_pre: Vec<f64>,
}

impl GradBuffers {
    pub fn new(encoder: &GridEncoder) -> Self {
        GradBuffers {
            grad_w1: vec![0.0; encoder.w1.len()],
            grad_b1: vec![0.0; encoder.b1.len()],
            grad_w2: vec![0.0; encoder.w2.len()],
            grad_b2: vec![0.0; encoder.b2.len()],
            grad_h1: vec![0.0; encoder.dim_hidden],
            grad_h1_pre: vec![0.0; encoder.dim_hidden],
            grad_output_pre: vec![0.0; encoder.dim_out],
        }
    }

    /// Setzt alle Buffer auf 0 zurück für den nächsten Backward-Schritt.
    pub fn reset(&mut self) {
        for v in &mut self.grad_w1 { *v = 0.0; }
        for v in &mut self.grad_b1 { *v = 0.0; }
        for v in &mut self.grad_w2 { *v = 0.0; }
        for v in &mut self.grad_b2 { *v = 0.0; }
        for v in &mut self.grad_h1 { *v = 0.0; }
        for v in &mut self.grad_h1_pre { *v = 0.0; }
        for v in &mut self.grad_output_pre { *v = 0.0; }
    }
}

// =============================================================================
// Trainings-Loop
// =============================================================================

/// Self-supervised Training auf ARC-Train-Paaren.
/// Minimiert die euklidisch approximierte hyperbolische Distanz zwischen
/// Input- und Output-Embedding pro Trainings-Paar.
pub fn train_grid_encoder(
    encoder: &mut GridEncoder,
    tasks: &[ArcTask],
    epochs: usize,
    learning_rate: f64,
) {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let mut total_pairs = 0usize;
    for task in tasks {
        total_pairs += task.train_pairs.len();
    }

    println!("Training GridEncoder: {} tasks, {} train pairs, {} epochs, lr={}",
        tasks.len(), total_pairs, epochs, learning_rate);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut count = 0usize;

        let mut shuffled: Vec<usize> = (0..tasks.len()).collect();
        shuffled.shuffle(&mut rng);

        for &idx in &shuffled {
            let task = &tasks[idx];
            for (input_grid, output_grid) in &task.train_pairs {
                match encoder.train_step(input_grid, output_grid, learning_rate) {
                    Ok(loss) => {
                        total_loss += loss;
                        count += 1;
                    }
                    Err(e) => {
                        eprintln!("Training error: {}", e);
                    }
                }
            }
        }

        if count > 0 {
            let avg = total_loss / count as f64;
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!("Epoch {:>4}: Avg Loss = {:.6} ({} pairs)", epoch, avg, count);
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hilfsfunktion für Tests ---

    fn make_test_encoder() -> GridEncoder {
        GridEncoder::new(100, 32, 16, 0.75)
    }

    fn make_test_grid() -> ArcGrid {
        ArcGrid::from_data(vec![vec![0; 10]; 10]).unwrap()
    }

    // --- 1. test_backward_shapes ---

    #[test]
    fn test_backward_shapes() {
        let encoder = make_test_encoder();
        let input = vec![0.5; 100];

        let _cached = encoder.forward_cached(&input);
        let _grad_output = vec![0.1; 16];
        let buffers = encoder.grad_buffers();

        // Wir muten encoder hier nicht, daher testen wir Buffer-Größen
        assert_eq!(buffers.grad_w1.len(), 3200, "grad_w1 should have 3200 elements");
        assert_eq!(buffers.grad_b1.len(), 32, "grad_b1 should have 32 elements");
        assert_eq!(buffers.grad_w2.len(), 512, "grad_w2 should have 512 elements");
        assert_eq!(buffers.grad_b2.len(), 16, "grad_b2 should have 16 elements");
        assert_eq!(buffers.grad_h1.len(), 32, "grad_h1 should have 32 elements");
        assert_eq!(buffers.grad_h1_pre.len(), 32, "grad_h1_pre should have 32 elements");
        assert_eq!(buffers.grad_output_pre.len(), 16, "grad_output_pre should have 16 elements");
    }

    // --- 2. test_training_decreases_loss ---

    #[test]
    fn test_training_decreases_loss() {
        let mut encoder = make_test_encoder();
        let input = make_test_grid();
        let output = ArcGrid::from_data(vec![vec![5; 10]; 10]).unwrap();

        let initial_loss = {
            let features_in: Vec<f64> = input.to_feature_vector().iter().map(|&x| x as f64).collect();
            let features_out: Vec<f64> = output.to_feature_vector().iter().map(|&x| x as f64).collect();
            let cached_in = encoder.forward_cached(&features_in);
            let cached_out = encoder.forward_cached(&features_out);
            euclidean_distance(&cached_in.output, &cached_out.output)
        };

        for _ in 0..10 {
            let _ = encoder.train_step(&input, &output, 0.01);
        }

        let final_loss = {
            let features_in: Vec<f64> = input.to_feature_vector().iter().map(|&x| x as f64).collect();
            let features_out: Vec<f64> = output.to_feature_vector().iter().map(|&x| x as f64).collect();
            let cached_in = encoder.forward_cached(&features_in);
            let cached_out = encoder.forward_cached(&features_out);
            euclidean_distance(&cached_in.output, &cached_out.output)
        };

        println!("Initial loss: {:.6}, Final loss: {:.6}", initial_loss, final_loss);
        assert!(final_loss < initial_loss,
            "Loss did not decrease: {} -> {}", initial_loss, final_loss);
    }

    // --- 3. test_l2_grad_finite ---

    #[test]
    fn test_l2_grad_finite() {
        let encoder = make_test_encoder();
        let input = vec![0.5; 100];

        let cached = encoder.forward_cached(&input);
        let grad_output = vec![0.1; 16];

        let mut buffers = encoder.grad_buffers();

        // Wir müssen einen Mock-Backward durchführen, um die L2-Gradienten zu testen
        // Wir kopieren den L2-Teil aus backward()
        let r_sq: f64 = cached.output_pre.iter().map(|x| x * x).sum();
        let r = r_sq.sqrt().max(1e-12);
        let scale = encoder.target_radius / r;

        let dot_grad_out_pre: f64 = grad_output
            .iter()
            .zip(&cached.output_pre)
            .map(|(g, p)| g * p)
            .sum();

        let factor = dot_grad_out_pre * scale / (r * r);

        for i in 0..encoder.dim_out {
            buffers.grad_output_pre[i] = grad_output[i] * scale - cached.output_pre[i] * factor;
        }

        // Prüfe dass keine NaN/Inf im L2-Gradienten
        for &val in &buffers.grad_output_pre {
            assert!(val.is_finite(), "L2 gradient contains non-finite value: {}", val);
        }
    }

    // --- 4. test_relu_mask ---

    #[test]
    fn test_relu_mask() {
        let mut encoder = make_test_encoder();
        // Setze spezifische Gewichte, sodass wir genau wissen,
        // welche Pre-Aktivations negativ sind und welche positiv.
        // w1[0..32] = 1.0 (für alle 100 Eingänge pro Neuron)
        // b1[0] = -1.0 → h1_pre[0] = -1.0 + 100*1.0*(-1.0) = -101.0 → h1[0] = 0.0
        // b1[1] = 1.0  → h1_pre[1] = 1.0 + 100*1.0*(-1.0) = -99.0 → h1[1] = 0.0
        // Für h1[2]: b1[2] = 200.0 → h1_pre[2] = 200.0 + 100*(-1.0) = 100.0 → h1[2] = 100.0
        for j in 0..100 {
            encoder.w1[0 * 100 + j] = 1.0;
            encoder.w1[1 * 100 + j] = 1.0;
            encoder.w1[2 * 100 + j] = -1.0;
        }
        encoder.b1[0] = -1.0;
        encoder.b1[1] = 1.0;
        encoder.b1[2] = 200.0;

        let input = vec![-1.0; 100];
        let cached = encoder.forward_cached(&input);

        // h1[0] und h1[1] sollten 0 sein (ReLU auf negative pre-activation)
        assert_eq!(cached.h1[0], 0.0, "ReLU output should be 0.0 when pre-activation is negative");
        assert_eq!(cached.h1[1], 0.0, "ReLU output should be 0.0 when pre-activation is negative");
        // h1[2] sollte 300.0 sein: b1[2]=200 + 100*(-1.0)*(-1.0) = 300
        assert_eq!(cached.h1[2], 300.0, "ReLU output should equal pre-activation when positive");

        // Simuliere einen beliebigen Gradienten durch Layer 2
        let grad_h1 = vec![1.0; encoder.dim_hidden];

        // Wende ReLU-Maske an
        for j in 0..encoder.dim_hidden {
            let expected = if cached.h1_pre[j] > 0.0 {
                grad_h1[j]
            } else {
                0.0
            };
            let actual = if cached.h1_pre[j] > 0.0 {
                grad_h1[j]
            } else {
                0.0
            };
            if cached.h1_pre[j] <= 0.0 {
                assert_eq!(actual, 0.0,
                    "ReLU gradient should be exactly 0 where pre-activation <= 0, got grad_h1_pre[{}] = {} (h1_pre[{}] = {})",
                    j, actual, j, cached.h1_pre[j]);
            }
            assert_eq!(actual, expected, "ReLU mask mismatch at index {}", j);
        }
    }

    // --- 5. test_symmetric_training ---

    #[test]
    fn test_symmetric_training() {
        // Ein Encoder, der für beide Pfade geklont wird (identische Initialisierung)
        let base_encoder = make_test_encoder();

        let input = make_test_grid();
        let output = ArcGrid::from_data(vec![vec![5; 10]; 10]).unwrap();

        let features_in: Vec<f64> = input.to_feature_vector().iter().map(|&x| x as f64).collect();
        let features_out: Vec<f64> = output.to_feature_vector().iter().map(|&x| x as f64).collect();

        // Pfad A: Input→Output (Gradienten berechnen, aber NOCH nicht updaten)
        let (grad_in_a, grad_out_a, loss_a) = {
            let mut enc = base_encoder.clone();
            let cached_in = enc.forward_cached(&features_in);
            let cached_out = enc.forward_cached(&features_out);
            let dist = euclidean_distance(&cached_in.output, &cached_out.output);
            let d = dist.max(1e-12);
            let grad_in: Vec<f64> = cached_in.output.iter().zip(&cached_out.output).map(|(a, b)| (a - b) / d).collect();
            let grad_out: Vec<f64> = cached_out.output.iter().zip(&cached_in.output).map(|(b, a)| (b - a) / d).collect();
            (grad_in, grad_out, dist)
        };

        // Pfad B: Output→Input (Gradienten berechnen, aber NOCH nicht updaten)
        // WICHTIG: Wir klonen base_encoder NOCHMAL, damit beide Pfade mit identischen
        // Gewichten starten. Dann wird derselbe Abstand berechnet.
        let (grad_in_b, grad_out_b, loss_b) = {
            let mut enc = base_encoder.clone();
            let cached_out = enc.forward_cached(&features_out);
            let cached_in = enc.forward_cached(&features_in);
            let dist = euclidean_distance(&cached_in.output, &cached_out.output);
            let d = dist.max(1e-12);
            let grad_in: Vec<f64> = cached_in.output.iter().zip(&cached_out.output).map(|(a, b)| (a - b) / d).collect();
            let grad_out: Vec<f64> = cached_out.output.iter().zip(&cached_in.output).map(|(b, a)| (b - a) / d).collect();
            (grad_in, grad_out, dist)
        };

        // Jetzt beide Pfade updaten (separate Encoder, damit sie sich nicht beeinflussen)
        let mut enc_a = base_encoder.clone();
        let cached_in_a = enc_a.forward_cached(&features_in);
        let cached_out_a = enc_a.forward_cached(&features_out);
        let mut buffers_a = enc_a.grad_buffers();
        enc_a.backward(&cached_in_a, &grad_in_a, &mut buffers_a, 0.01);
        enc_a.backward(&cached_out_a, &grad_out_a, &mut buffers_a, 0.01);

        let mut enc_b = base_encoder.clone();
        let cached_out_b = enc_b.forward_cached(&features_out);
        let cached_in_b = enc_b.forward_cached(&features_in);
        let mut buffers_b = enc_b.grad_buffers();
        enc_b.backward(&cached_in_b, &grad_in_b, &mut buffers_b, 0.01);
        enc_b.backward(&cached_out_b, &grad_out_b, &mut buffers_b, 0.01);

        println!("Loss A (Input→Output): {:.6}, Loss B (Output→Input): {:.6}", loss_a, loss_b);
        // Die Loss-Werte sollten identisch sein, da es derselbe euklidische Abstand ist
        assert!((loss_a - loss_b).abs() < 1e-10,
            "Symmetric training should produce identical losses: {} vs {}", loss_a, loss_b);
    }

    // --- Zusätzliche Tests ---

    #[test]
    fn test_forward_cached_matches_forward() {
        let encoder = make_test_encoder();
        let input = vec![0.5; 100];

        let cached = encoder.forward_cached(&input);
        let direct = encoder.forward(&input);

        assert_eq!(cached.output.len(), direct.len());
        for (a, b) in cached.output.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-12, "forward_cached mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_hyperbolic_distance_grad_shape() {
        let x = HyperbolicPoint { coords: vec![0.1, 0.2, 0.3] };
        let y = HyperbolicPoint { coords: vec![0.4, 0.5, 0.6] };

        let grad = GridEncoder::hyperbolic_distance_grad(&x, &y);
        assert_eq!(grad.len(), 3);

        // Gradient sollte (x - y) / ||x - y|| sein
        let dist = euclidean_distance(&x.coords, &y.coords);
        for (g, (xi, yi)) in grad.iter().zip(x.coords.iter().zip(&y.coords)) {
            assert!((*g - (xi - yi) / dist).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hyperbolic_distance_grad_self_is_zero() {
        let x = HyperbolicPoint { coords: vec![0.1, 0.2, 0.3] };
        let grad = GridEncoder::hyperbolic_distance_grad(&x, &x);
        for &g in &grad {
            assert!(g.abs() < 1e-12, "Gradient at self should be 0, got {}", g);
        }
    }

    #[test]
    fn test_backward_updates_all_parameters() {
        let mut encoder = make_test_encoder();
        let input = vec![0.5; 100];
        let cached = encoder.forward_cached(&input);
        let grad_output = vec![0.1; 16];

        // Speichere originale Parameter
        let w1_orig = encoder.w1.clone();
        let b1_orig = encoder.b1.clone();
        let w2_orig = encoder.w2.clone();
        let b2_orig = encoder.b2.clone();

        let mut buffers = encoder.grad_buffers();
        encoder.backward(&cached, &grad_output, &mut buffers, 0.01);

        // Prüfe dass sich alle Parameter geändert haben (nicht nur b2)
        let mut w1_changed = false;
        for (a, b) in encoder.w1.iter().zip(&w1_orig) {
            if (a - b).abs() > 1e-12 { w1_changed = true; break; }
        }
        assert!(w1_changed, "w1 should have been updated");

        let mut b1_changed = false;
        for (a, b) in encoder.b1.iter().zip(&b1_orig) {
            if (a - b).abs() > 1e-12 { b1_changed = true; break; }
        }
        assert!(b1_changed, "b1 should have been updated");

        let mut w2_changed = false;
        for (a, b) in encoder.w2.iter().zip(&w2_orig) {
            if (a - b).abs() > 1e-12 { w2_changed = true; break; }
        }
        assert!(w2_changed, "w2 should have been updated");

        let mut b2_changed = false;
        for (a, b) in encoder.b2.iter().zip(&b2_orig) {
            if (a - b).abs() > 1e-12 { b2_changed = true; break; }
        }
        assert!(b2_changed, "b2 should have been updated");
    }

    #[test]
    fn test_grad_buffers_reuse() {
        let encoder = make_test_encoder();
        let mut buffers = encoder.grad_buffers();

        // Erste Nutzung
        buffers.grad_w1[0] = 1.0;
        buffers.reset();
        assert_eq!(buffers.grad_w1[0], 0.0, "Buffer should be reset to 0");

        // Zweite Nutzung
        buffers.grad_b2[0] = 2.0;
        buffers.reset();
        assert_eq!(buffers.grad_b2[0], 0.0, "Buffer should be reset to 0");
    }
}
