use crate::geometry::HyperbolicPoint;

/// 100D Rate-Coding → 32D → 16D HyperbolicPoint
/// Identisches Backprop-Framework wie GridEncoder, aber Audio-Domäne
pub struct ShdEncoder {
    pub w1: Vec<f64>, // 100 * 32 = 3200
    pub b1: Vec<f64>, // 32
    pub w2: Vec<f64>, // 32 * 16 = 512
    pub b2: Vec<f64>, // 16
    pub target_radius: f64,
}

/// Cache für Backprop (DOD: keine Allokation im Hot-Path)
pub struct ForwardCache {
    pub input: Vec<f64>,
    pub h1_pre: Vec<f64>,
    pub h1: Vec<f64>,
    pub output_pre: Vec<f64>,
    pub output: Vec<f64>,
}

/// Pre-allokierte Gradient-Buffer
pub struct GradBuffers {
    pub dw1: Vec<f64>,
    pub db1: Vec<f64>,
    pub dw2: Vec<f64>,
    pub db2: Vec<f64>,
}

impl ShdEncoder {
    pub fn new() -> Self {
        let mut encoder = Self {
            w1: vec![0.0; 3200],
            b1: vec![0.0; 32],
            w2: vec![0.0; 512],
            b2: vec![0.0; 16],
            target_radius: 0.75,
        };
        encoder.init_xavier();
        encoder
    }

    fn init_xavier(&mut self) {
        let scale1 = (2.0 / 100.0f64).sqrt();
        let scale2 = (2.0 / 32.0f64).sqrt();
        for i in 0..3200 {
            self.w1[i] = (rand::random::<f64>() * 2.0 - 1.0) * scale1;
        }
        for i in 0..512 {
            self.w2[i] = (rand::random::<f64>() * 2.0 - 1.0) * scale2;
        }
    }

    pub fn forward_cached(&self, input: &[f64]) -> ForwardCache {
        let mut h1_pre = vec![0.0; 32];
        let mut h1 = vec![0.0; 32];

        // Layer 1: 100 → 32
        for j in 0..32 {
            let mut sum = self.b1[j];
            for i in 0..100 {
                sum += input[i] * self.w1[i * 32 + j];
            }
            h1_pre[j] = sum;
            h1[j] = sum.max(0.0); // ReLU
        }

        // Layer 2: 32 → 16
        let mut output_pre = vec![0.0; 16];
        for j in 0..16 {
            let mut sum = self.b2[j];
            for i in 0..32 {
                sum += h1[i] * self.w2[i * 16 + j];
            }
            output_pre[j] = sum;
        }

        // L2-Norm auf target_radius
        let norm: f64 = output_pre.iter().map(|x| x * x).sum::<f64>().sqrt();
        let output = if norm > 1e-9 {
            let scale = self.target_radius / norm;
            output_pre.iter().map(|x| x * scale).collect()
        } else {
            output_pre.clone()
        };

        ForwardCache {
            input: input.to_vec(),
            h1_pre,
            h1,
            output_pre,
            output,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.forward_cached(input).output
    }

    pub fn backward(
        &self,
        cache: &ForwardCache,
        grad_output: &[f64],
        grads: &mut GradBuffers,
        lr: f64,
    ) {
        // L2-Gradient
        let r = self.target_radius;
        let r_pre = cache.output_pre.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = r / r_pre.max(1e-9);
        let dot = cache.output_pre.iter().zip(grad_output.iter()).map(|(a, b)| a * b).sum::<f64>();
        let factor = dot * scale / (r_pre * r_pre).max(1e-9);

        let mut grad_pre = vec![0.0; 16];
        for i in 0..16 {
            grad_pre[i] = scale * grad_output[i] - cache.output_pre[i] * factor;
        }

        // ReLU-Maske
        let relu_mask: Vec<f64> = cache.h1_pre.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();

        // Layer 2 Backprop
        let mut grad_h1 = vec![0.0; 32];
        for i in 0..32 {
            for j in 0..16 {
                let g = grad_pre[j] * relu_mask[i];
                grads.dw2[i * 16 + j] += lr * g * cache.h1[i];
                grad_h1[i] += grad_pre[j] * self.w2[i * 16 + j] * relu_mask[i];
            }
        }
        for j in 0..16 {
            grads.db2[j] += lr * grad_pre[j];
        }

        // Layer 1 Backprop
        for i in 0..100 {
            for j in 0..32 {
                grads.dw1[i * 32 + j] += lr * grad_h1[j] * cache.input[i];
            }
        }
        for j in 0..32 {
            grads.db1[j] += lr * grad_h1[j];
        }
    }

    pub fn apply_gradients(&mut self, grads: &GradBuffers) {
        for i in 0..3200 {
            self.w1[i] -= grads.dw1[i];
        }
        for i in 0..32 {
            self.b1[i] -= grads.db1[i];
        }
        for i in 0..512 {
            self.w2[i] -= grads.dw2[i];
        }
        for i in 0..16 {
            self.b2[i] -= grads.db2[i];
        }
    }

    pub fn zero_grads(grads: &mut GradBuffers) {
        grads.dw1.fill(0.0);
        grads.db1.fill(0.0);
        grads.dw2.fill(0.0);
        grads.db2.fill(0.0);
    }

    pub fn new_grad_buffers() -> GradBuffers {
        GradBuffers {
            dw1: vec![0.0; 3200],
            db1: vec![0.0; 32],
            dw2: vec![0.0; 512],
            db2: vec![0.0; 16],
        }
    }

    /// Hyperbolic Distance Loss: minimize distance between same-class pairs
    pub fn train_step(
        &mut self,
        a: &[f64],
        b: &[f64],
        label_same: bool,
        lr: f64,
    ) -> f64 {
        let cache_a = self.forward_cached(a);
        let cache_b = self.forward_cached(b);

        let dist = cache_a.output.iter().zip(cache_b.output.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let target = if label_same { 0.0 } else { 2.0 }; // 2.0 = großer Abstand für verschiedene Klassen
        let loss = (dist - target).powi(2);

        let dloss = 2.0 * (dist - target);
        if dist < 1e-9 {
            return loss;
        }

        let mut grads = Self::new_grad_buffers();

        // Gradient für a
        let grad_a: Vec<f64> = cache_a.output.iter().zip(cache_b.output.iter())
            .map(|(ai, bi)| dloss * (ai - bi) / dist)
            .collect();
        self.backward(&cache_a, &grad_a, &mut grads, lr);

        // Gradient für b (entgegengesetztes Vorzeichen)
        let grad_b: Vec<f64> = cache_b.output.iter().zip(cache_a.output.iter())
            .map(|(bi, ai)| dloss * (bi - ai) / dist)
            .collect();
        self.backward(&cache_b, &grad_b, &mut grads, lr);

        self.apply_gradients(&mut grads);
        loss
    }
}
