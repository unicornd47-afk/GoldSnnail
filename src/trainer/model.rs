//! Recurrent LIF network with hand-rolled BPTT (surrogate gradient) — Block 2.
//!
//! Forward (hard spike + hard reset) is used for training/inference; the
//! backward pass unrolls the network over time and flows gradient through each
//! spike using the arctan surrogate derivative. The backward is the exact
//! derivative of the *smooth* (no-reset) model, verified against finite
//! differences in the tests — this is the standard surrogate-gradient setup.

use super::lif::{hard_spike, surrogate_fire, surrogate_grad};

#[derive(Clone)]
pub struct RnnLif {
    pub n_in: usize,
    pub n_hid: usize,
    pub n_out: usize,
    /// Leak coefficient `k = dt/τ`, clamped to (0, 1).
    pub k: f32,
    /// Spike threshold.
    pub theta: f32,
    /// Surrogate slope.
    pub beta: f32,
    /// Input→hidden weights, `[n_in * n_hid]` (row = input, col = hidden).
    pub w_in: Vec<f32>,
    /// Hidden→hidden recurrent weights, `[n_hid * n_hid]`.
    pub w_rec: Vec<f32>,
    /// Hidden→output readout weights, `[n_hid * n_out]`.
    pub w_out: Vec<f32>,
}

/// Per-timestep state cached by [`RnnLif::forward`] for the backward pass.
#[derive(Clone)]
pub struct Forward {
    pub spikes: Vec<Vec<f32>>, // [T][n_hid]
    pub v_pre: Vec<Vec<f32>>,  // [T][n_hid] membrane before fire/reset
    pub v: Vec<Vec<f32>>,      // [T][n_hid] membrane after reset
    pub i_syn: Vec<Vec<f32>>,  // [T][n_hid] synaptic current
    pub mean_spike: Vec<f32>,  // [n_hid]
    pub logits: Vec<f32>,      // [n_out]
}

#[derive(Clone, Default)]
pub struct Grads {
    pub w_in: Vec<f32>,
    pub w_rec: Vec<f32>,
    pub w_out: Vec<f32>,
    pub k: f32,
}

/// Softmax cross-entropy. Returns `(loss, d_logits)`.
pub fn softmax_cross_entropy(logits: &[f32], label: usize) -> (f32, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    for x in &mut p {
        *x /= sum;
    }
    let loss = -p[label].ln();
    p[label] -= 1.0;
    (loss, p)
}

/// Deterministic uniform init in `(-scale, scale)` (LCG, no `rand` dependency).
fn init_weights(n: usize, m: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x517c_c1b7_2722_0a95);
    let mut out = vec![0.0f32; n * m];
    for v in out.iter_mut() {
        s = s
            .wrapping_mul(6364_1362_2384_6793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((s >> 33) as u32) as f32 / (u32::MAX as f32);
        *v = (u - 0.5) * 2.0 * scale;
    }
    out
}

impl RnnLif {
    pub fn new(n_in: usize, n_hid: usize, n_out: usize, k: f32, theta: f32, beta: f32, seed: u64) -> Self {
        let s_in = (2.0 / (n_in + n_hid) as f32).sqrt();
        let s_rec = (2.0 / (n_hid + n_hid) as f32).sqrt();
        let s_out = (2.0 / (n_hid + n_out) as f32).sqrt();
        Self {
            n_in,
            n_hid,
            n_out,
            k,
            theta,
            beta,
            w_in: init_weights(n_in, n_hid, s_in, seed),
            w_rec: init_weights(n_hid, n_hid, s_rec, seed.wrapping_add(1)),
            w_out: init_weights(n_hid, n_out, s_out, seed.wrapping_add(2)),
        }
    }

    /// Forward pass over `x: [T][n_in]`. `smooth = true` uses the arctan
    /// surrogate (no reset) — only for gradient verification, never for training.
    pub fn forward(&self, x: &[Vec<f32>], smooth: bool) -> Forward {
        let t_steps = x.len();
        let (n_in, n_hid, n_out) = (self.n_in, self.n_hid, self.n_out);
        let mut f = Forward {
            spikes: vec![vec![0.0; n_hid]; t_steps],
            v_pre: vec![vec![0.0; n_hid]; t_steps],
            v: vec![vec![0.0; n_hid]; t_steps],
            i_syn: vec![vec![0.0; n_hid]; t_steps],
            mean_spike: vec![0.0; n_hid],
            logits: vec![0.0; n_out],
        };
        for t in 0..t_steps {
            for j in 0..n_hid {
                let mut s = 0.0f32;
                for i in 0..n_in {
                    s += x[t][i] * self.w_in[i * n_hid + j];
                }
                if t > 0 {
                    for a in 0..n_hid {
                        s += f.spikes[t - 1][a] * self.w_rec[a * n_hid + j];
                    }
                }
                f.i_syn[t][j] = s;
                let v_prev = if t == 0 { 0.0 } else { f.v[t - 1][j] };
                let v_pre = (1.0 - self.k) * v_prev + self.k * s;
                f.v_pre[t][j] = v_pre;
                let sp = if smooth {
                    surrogate_fire(v_pre - self.theta, self.beta)
                } else {
                    hard_spike(v_pre - self.theta)
                };
                f.spikes[t][j] = sp;
                f.v[t][j] = if smooth {
                    v_pre
                } else if sp > 0.5 {
                    0.0
                } else {
                    v_pre
                };
            }
        }
        for j in 0..n_hid {
            let mut m = 0.0f32;
            for t in 0..t_steps {
                m += f.spikes[t][j];
            }
            f.mean_spike[j] = m / t_steps as f32;
        }
        for o in 0..n_out {
            let mut s = 0.0f32;
            for h in 0..n_hid {
                s += f.mean_spike[h] * self.w_out[h * n_out + o];
            }
            f.logits[o] = s;
        }
        f
    }

    /// Backward (BPTT) using the surrogate derivative. `x` is the input used
    /// in the matching forward; `d_logits` is the upstream gradient wrt logits.
    pub fn backward(&self, f: &Forward, x: &[Vec<f32>], d_logits: &[f32]) -> Grads {
        let t_steps = f.spikes.len();
        let (n_in, n_hid, n_out) = (self.n_in, self.n_hid, self.n_out);
        let mut g = Grads {
            w_in: vec![0.0; n_in * n_hid],
            w_rec: vec![0.0; n_hid * n_hid],
            w_out: vec![0.0; n_hid * n_out],
            k: 0.0,
        };

        // Readout: logits = w_out · mean_spike.
        for h in 0..n_hid {
            for o in 0..n_out {
                g.w_out[h * n_out + o] = d_logits[o] * f.mean_spike[h];
            }
        }
        let mut d_mean = vec![0.0f32; n_hid];
        for h in 0..n_hid {
            for o in 0..n_out {
                d_mean[h] += d_logits[o] * self.w_out[h * n_out + o];
            }
        }

        // dL/ds[t] starts from the mean readout, then accumulates recurrence.
        let mut d_spike = vec![vec![0.0f32; n_hid]; t_steps];
        for t in 0..t_steps {
            for j in 0..n_hid {
                d_spike[t][j] = d_mean[j] / t_steps as f32;
            }
        }

        let mut d_v = vec![0.0f32; n_hid];
        for t in (0..t_steps).rev() {
            let mut g_i = vec![0.0f32; n_hid];
            for j in 0..n_hid {
                let v_prev = if t == 0 { 0.0 } else { f.v[t - 1][j] };
                let g_vpre = d_spike[t][j]
                    * surrogate_grad(f.v_pre[t][j] - self.theta, self.beta)
                    + d_v[j];
                g_i[j] = g_vpre * self.k;
                d_v[j] = g_vpre * (1.0 - self.k);
                g.k += g_vpre * (f.i_syn[t][j] - v_prev);
            }
            for i in 0..n_in {
                for j in 0..n_hid {
                    g.w_in[i * n_hid + j] += g_i[j] * x[t][i];
                }
            }
            if t > 0 {
                for a in 0..n_hid {
                    let mut acc = 0.0f32;
                    for j in 0..n_hid {
                        acc += g_i[j] * self.w_rec[a * n_hid + j];
                    }
                    d_spike[t - 1][a] += acc;
                    for j in 0..n_hid {
                        g.w_rec[a * n_hid + j] += g_i[j] * f.spikes[t - 1][a];
                    }
                }
            }
        }
        g
    }

    /// Apply gradients in place (plain SGD). `k` is clamped to (0.05, 0.95).
    pub fn apply_gradients(&mut self, g: &Grads, lr: f32) {
        for i in 0..self.w_in.len() {
            self.w_in[i] -= lr * g.w_in[i];
        }
        for i in 0..self.w_rec.len() {
            self.w_rec[i] -= lr * g.w_rec[i];
        }
        for i in 0..self.w_out.len() {
            self.w_out[i] -= lr * g.w_out[i];
        }
        self.k = (self.k - lr * g.k).clamp(0.05, 0.95);
    }

    /// One SGD step (hard forward + surrogate backward). Returns the loss
    /// measured *before* the update.
    pub fn sgd_step(&mut self, x: &[Vec<f32>], label: usize, lr: f32) -> f32 {
        let f = self.forward(x, false);
        let (loss, d_logits) = softmax_cross_entropy(&f.logits, label);
        let g = self.backward(&f, x, &d_logits);
        self.apply_gradients(&g, lr);
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> RnnLif {
        RnnLif::new(3, 4, 2, 0.5, 1.0, 10.0, 42)
    }

    fn input() -> Vec<Vec<f32>> {
        vec![
            vec![0.8, 0.1, 0.6],
            vec![0.2, 0.9, 0.4],
            vec![0.5, 0.3, 0.7],
        ]
    }

    fn smooth_loss(m: &RnnLif, x: &[Vec<f32>], label: usize) -> f32 {
        let f = m.forward(x, true);
        softmax_cross_entropy(&f.logits, label).0
    }

    #[test]
    fn bptt_gradient_matches_finite_difference() {
        let m = model();
        let x = input();
        let label = 1usize;
        let f = m.forward(&x, true);
        let (_, d_logits) = softmax_cross_entropy(&f.logits, label);
        let g = m.backward(&f, &x, &d_logits);
        let eps = 1e-3f32;

        for idx in 0..m.w_out.len() {
            let mut mp = m.clone();
            let mut mm = m.clone();
            mp.w_out[idx] += eps;
            mm.w_out[idx] -= eps;
            let num = (smooth_loss(&mp, &x, label) - smooth_loss(&mm, &x, label)) / (2.0 * eps);
            assert!((num - g.w_out[idx]).abs() < 5e-3, "w_out[{idx}]: num={num} ana={}", g.w_out[idx]);
        }
        for idx in 0..m.w_in.len() {
            let mut mp = m.clone();
            let mut mm = m.clone();
            mp.w_in[idx] += eps;
            mm.w_in[idx] -= eps;
            let num = (smooth_loss(&mp, &x, label) - smooth_loss(&mm, &x, label)) / (2.0 * eps);
            assert!((num - g.w_in[idx]).abs() < 5e-3, "w_in[{idx}]: num={num} ana={}", g.w_in[idx]);
        }
        for idx in 0..m.w_rec.len() {
            let mut mp = m.clone();
            let mut mm = m.clone();
            mp.w_rec[idx] += eps;
            mm.w_rec[idx] -= eps;
            let num = (smooth_loss(&mp, &x, label) - smooth_loss(&mm, &x, label)) / (2.0 * eps);
            assert!((num - g.w_rec[idx]).abs() < 5e-3, "w_rec[{idx}]: num={num} ana={}", g.w_rec[idx]);
        }
        {
            let mut mp = m.clone();
            let mut mm = m.clone();
            mp.k += eps;
            mm.k -= eps;
            let num = (smooth_loss(&mp, &x, label) - smooth_loss(&mm, &x, label)) / (2.0 * eps);
            assert!((num - g.k).abs() < 5e-3, "k: num={num} ana={}", g.k);
        }
    }

    #[test]
    fn surrogate_gradient_descends_smooth_loss() {
        let m = model();
        let x = input();
        let label = 1usize;
        let f = m.forward(&x, true);
        let (loss0, d_logits) = softmax_cross_entropy(&f.logits, label);
        let g = m.backward(&f, &x, &d_logits);
        let lr = 1e-3f32;
        let mut m2 = m.clone();
        m2.apply_gradients(&g, lr);
        let f2 = m2.forward(&x, true);
        let (loss1, _) = softmax_cross_entropy(&f2.logits, label);
        assert!(loss1 < loss0, "smooth loss did not descend: {loss0} -> {loss1}");
    }

    #[test]
    fn training_reduces_hard_loss() {
        // Low threshold + scaled inputs guarantee initial spiking, so the hard
        // loss is not stuck on the ln(n_classes) plateau.
        let mut m = RnnLif::new(3, 8, 2, 0.8, 0.1, 10.0, 7);
        let x: Vec<Vec<f32>> = input().iter().map(|r| r.iter().map(|v| v * 2.0).collect()).collect();
        let label = 0usize;
        let f0 = m.forward(&x, false);
        let (loss0, _) = softmax_cross_entropy(&f0.logits, label);
        let mut loss = loss0;
        for _ in 0..1000 {
            loss = m.sgd_step(&x, label, 0.05);
        }
        assert!(loss < loss0, "hard loss did not decrease: {loss0} -> {loss}");
    }
}
