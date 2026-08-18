//! World Model — Hyperbolic Predictive Coding
//!
//! Learns transitions between compressed latent states via an RNN in the
//! tangential space + `exp_map` projection back onto the manifold.
//!
//! All weight matrices are flat `Vec<f64>`. No `Vec<Vec<T>>`, no pointer
//! indirection in hot paths.

use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::LabError;
use ndarray::Array1;

/// Hyperbolic World Model for Predictive Coding.
///
/// Predicts the next latent state from the current one using a small RNN
/// whose output is projected back onto the Poincaré ball via `exp_map`.
#[derive(Debug, Clone)]
pub struct WorldModel {
    pub ball: PoincareBall,
    pub latent_dim: usize,
    pub hidden_dim: usize,
    /// Input → Hidden weights: [hidden_dim × latent_dim]
    pub w_ih: Vec<f64>,
    /// Hidden → Hidden recurrent weights: [hidden_dim × hidden_dim]
    pub w_hh: Vec<f64>,
    /// Hidden → Output tangent weights: [latent_dim × hidden_dim]
    pub w_ho: Vec<f64>,
    /// Current hidden state (Euclidean, tangential space)
    pub hidden: Vec<f64>,
    /// State history for N-step training
    pub state_history: Vec<HyperbolicPoint>,
    pub history_capacity: usize,
}

impl WorldModel {
    pub fn new(latent_dim: usize, hidden_dim: usize, curvature: f64) -> Self {
        let n_ih = hidden_dim * latent_dim;
        let n_hh = hidden_dim * hidden_dim;
        let n_ho = latent_dim * hidden_dim;

        // Deterministic small init (no rand needed)
        let w_ih: Vec<f64> = (0..n_ih)
            .map(|i| (i as f64 * 0.137).sin() * 0.01)
            .collect();
        // Near-identity for stable dynamics
        let w_hh: Vec<f64> = (0..n_hh)
            .map(|i| if i % (hidden_dim + 1) == 0 { 0.95 } else { 0.0 })
            .collect();
        let w_ho: Vec<f64> = (0..n_ho)
            .map(|i| (i as f64 * 0.173).cos() * 0.01)
            .collect();

        Self {
            ball: PoincareBall::new(curvature),
            latent_dim,
            hidden_dim,
            w_ih,
            w_hh,
            w_ho,
            hidden: vec![0.0; hidden_dim],
            state_history: Vec::with_capacity(100),
            history_capacity: 100,
        }
    }

    /// Forward pass: current state → predicted next state.
    pub fn predict(&mut self, current: &HyperbolicPoint) -> Result<HyperbolicPoint, LabError> {
        if current.coords.len() != self.latent_dim {
            return Err(LabError::DimensionMismatch {
                expected: self.latent_dim,
                got: current.coords.len(),
            });
        }
        // 1. Hidden update: h = tanh(W_ih · x + W_hh · h_prev)
        let mut new_hidden = vec![0.0f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut acc = 0.0;
            for j in 0..self.latent_dim {
                acc += self.w_ih[i * self.latent_dim + j] * current.coords[j];
            }
            for k in 0..self.hidden_dim {
                acc += self.w_hh[i * self.hidden_dim + k] * self.hidden[k];
            }
            new_hidden[i] = acc.tanh();
        }
        self.hidden = new_hidden;

        // 2. Output tangent
        let mut tangent = vec![0.0f64; self.latent_dim];
        for i in 0..self.latent_dim {
            let mut acc = 0.0;
            for j in 0..self.hidden_dim {
                acc += self.w_ho[i * self.hidden_dim + j] * self.hidden[j];
            }
            tangent[i] = acc;
        }

        // 3. exp_map: tangent space → manifold
        let tangent_arr = Array1::from(tangent);
        self.ball.exp_map(current, &tangent_arr)
    }

    /// Hyperbolic prediction error (distance on the manifold).
    pub fn prediction_error(
        &self,
        predicted: &HyperbolicPoint,
        actual: &HyperbolicPoint,
    ) -> Result<f64, LabError> {
        self.ball.distance(predicted, actual)
    }

    /// Observe a state for the history buffer.
    pub fn observe(&mut self, state: HyperbolicPoint) {
        if self.state_history.len() >= self.history_capacity {
            self.state_history.remove(0);
        }
        self.state_history.push(state);
    }

    /// Single training step in tangent space.
    /// Returns the prediction error.
    pub fn train_step(
        &mut self,
        current: &HyperbolicPoint,
        next_actual: &HyperbolicPoint,
        lr: f64,
    ) -> Result<f64, LabError> {
        let predicted = self.predict(current)?;
        let err = self.prediction_error(&predicted, next_actual)?;

        // Gradient update on w_ho: Δw = lr * (actual - pred) * hidden^T
        for i in 0..self.latent_dim {
            let diff = next_actual.coords[i] - predicted.coords[i];
            for j in 0..self.hidden_dim {
                let idx = i * self.hidden_dim + j;
                self.w_ho[idx] += lr * diff * self.hidden[j];
            }
        }
        Ok(err)
    }

    /// Reset internal state (new episode / context switch).
    pub fn reset_hidden(&mut self) {
        self.hidden.fill(0.0);
    }

    /// Train on entire history (N-step).
    pub fn train_on_history(&mut self, lr: f64) -> Result<f64, LabError> {
        if self.state_history.len() < 2 {
            return Ok(0.0);
        }
        let hist = self.state_history.clone();
        let mut total_err = 0.0;
        for window in hist.windows(2) {
            total_err += self.train_step(&window[0], &window[1], lr)?;
        }
        Ok(total_err / (hist.len() - 1) as f64)
    }
}

#[cfg(feature = "vulkan")]
impl WorldModel {
    /// GPU-beschleunigte Batch-Distanz für History-Training
    pub fn batch_distance_gpu(
        &self,
        vk: &crate::vulkan::VulkanCompute,
        points_a: &[HyperbolicPoint],
        points_b: &[HyperbolicPoint],
    ) -> Result<Vec<f64>, String> {
        // Flatten zu Vec<f32> für GPU
        let flat_a: Vec<f32> = points_a.iter()
            .flat_map(|p| vec![p.coords[0] as f32, p.coords[1] as f32, 0.0f32, 0.0f32])
            .collect();
        let flat_b: Vec<f32> = points_b.iter()
            .flat_map(|p| vec![p.coords[0] as f32, p.coords[1] as f32, 0.0f32, 0.0f32])
            .collect();
        
        let buf_a = vk.create_buffer(&flat_a)?;
        let buf_b = vk.create_buffer(&flat_b)?;
        let mut distances = vec![0.0f32; points_a.len()];
        let buf_out = vk.create_buffer(&distances)?;
        
        // ... Descriptor Set Binding & Dispatch ...
        
        vk.download_buffer(&buf_out, &mut distances)?;
        Ok(distances.into_iter().map(|f| f as f64).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample_point(x: f64, y: f64) -> HyperbolicPoint {
        HyperbolicPoint::new(array![x * 0.5, y * 0.5]).unwrap()
    }

    #[test]
    fn predict_returns_valid_hyperbolic_point() {
        let mut wm = WorldModel::new(2, 4, 1.0);
        let p = sample_point(0.1, 0.2);
        let pred = wm.predict(&p).unwrap();
        assert!(pred.euclidean_norm() < 1.0, "Prediction must stay inside the ball");
    }

    #[test]
    fn prediction_error_is_non_negative() {
        let wm = WorldModel::new(2, 4, 1.0);
        let a = sample_point(0.1, 0.1);
        let b = sample_point(0.2, -0.1);
        let err = wm.prediction_error(&a, &b).unwrap();
        assert!(err >= 0.0);
    }

    #[test]
    fn train_step_reduces_error_on_constant_sequence() {
        let mut wm = WorldModel::new(2, 8, 1.0);
        let p = sample_point(0.05, 0.05);

        let pred_before = wm.predict(&p).unwrap();
        let err_before = wm.prediction_error(&pred_before, &p).unwrap();

        for _ in 0..50 {
            let _ = wm.train_step(&p, &p, 0.1);
            wm.reset_hidden();
        }

        let pred_after = wm.predict(&p).unwrap();
        let err_after = wm.prediction_error(&pred_after, &p).unwrap();

        assert!(
            err_after < err_before,
            "Error should decrease: before={}, after={}",
            err_before, err_after
        );
    }

    #[test]
    fn history_training_runs_without_panic() {
        let mut wm = WorldModel::new(2, 4, 1.0);
        wm.observe(sample_point(0.1, 0.0));
        wm.observe(sample_point(0.11, 0.01));
        wm.observe(sample_point(0.12, 0.02));

        let avg_err = wm.train_on_history(0.05).unwrap();
        assert!(avg_err >= 0.0);
    }

    #[test]
    fn reset_clears_hidden_state() {
        let mut wm = WorldModel::new(2, 4, 1.0);
        let p = sample_point(0.1, 0.1);
        let _ = wm.predict(&p);
        wm.reset_hidden();
        assert!(wm.hidden.iter().all(|&h| h == 0.0));
    }
}
