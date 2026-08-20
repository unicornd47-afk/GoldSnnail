//! SHD supervised classifier (softmax cross-entropy) — Block 2.
//! MLP: features -> HIDDEN (ReLU) -> 20 softmax, trained with SGD + momentum,
//! lr decay, L2 weight decay. Feature modes via SHD_FEAT: rate | ttfs | time.

use goldsnnail::audio::shd_loader::{ShdDataset, ShdSample};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

const HIDDEN: usize = 64;
const CLASSES: usize = 20;
const WEIGHT_DECAY: f64 = 1e-4;
const MOMENTUM: f64 = 0.9;
const TIME_BINS: usize = 10;
const FREQ_BINS: usize = 100;

struct Mlp {
    input: usize,
    w1: Vec<f64>, vw1: Vec<f64>, b1: Vec<f64>, vb1: Vec<f64>,
    w2: Vec<f64>, vw2: Vec<f64>, b2: Vec<f64>, vb2: Vec<f64>,
}

impl Mlp {
    fn new(input: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut m = Self {
            input,
            w1: vec![0.0; input * HIDDEN], vw1: vec![0.0; input * HIDDEN],
            b1: vec![0.0; HIDDEN], vb1: vec![0.0; HIDDEN],
            w2: vec![0.0; HIDDEN * CLASSES], vw2: vec![0.0; HIDDEN * CLASSES],
            b2: vec![0.0; CLASSES], vb2: vec![0.0; CLASSES],
        };
        let s1 = (2.0 / input as f64).sqrt();
        let s2 = (2.0 / HIDDEN as f64).sqrt();
        for w in m.w1.iter_mut() { *w = rng.gen_range(-1.0..1.0) * s1; }
        for w in m.w2.iter_mut() { *w = rng.gen_range(-1.0..1.0) * s2; }
        m
    }

    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut h1 = vec![0.0f64; HIDDEN];
        for j in 0..HIDDEN {
            let mut s = self.b1[j];
            for i in 0..self.input { s += x[i] * self.w1[i * HIDDEN + j]; }
            h1[j] = s.max(0.0);
        }
        let mut logits = vec![0.0f64; CLASSES];
        for j in 0..CLASSES {
            let mut s = self.b2[j];
            for i in 0..HIDDEN { s += h1[i] * self.w2[i * CLASSES + j]; }
            logits[j] = s;
        }
        (h1, logits)
    }

    fn train_step(&mut self, x: &[f64], y: usize, lr: f64) -> f64 {
        let (h1, logits) = self.forward(x);
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut p = vec![0.0f64; CLASSES];
        let mut sum = 0.0;
        for j in 0..CLASSES { p[j] = (logits[j] - max).exp(); sum += p[j]; }
        for j in 0..CLASSES { p[j] /= sum; }
        let loss = -p[y].ln();

        let mut g_logits = p;
        g_logits[y] -= 1.0;

        let mut g_h1 = vec![0.0f64; HIDDEN];
        for i in 0..HIDDEN {
            let mut g = 0.0;
            for j in 0..CLASSES { g += g_logits[j] * self.w2[i * CLASSES + j]; }
            g_h1[i] = if h1[i] > 0.0 { g } else { 0.0 };
        }

        for i in 0..HIDDEN {
            for j in 0..CLASSES {
                let idx = i * CLASSES + j;
                let g = g_logits[j] * h1[i] + WEIGHT_DECAY * self.w2[idx];
                self.vw2[idx] = MOMENTUM * self.vw2[idx] - lr * g;
                self.w2[idx] += self.vw2[idx];
            }
        }
        for j in 0..CLASSES {
            self.vb2[j] = MOMENTUM * self.vb2[j] - lr * g_logits[j];
            self.b2[j] += self.vb2[j];
        }
        for i in 0..self.input {
            for j in 0..HIDDEN {
                let idx = i * HIDDEN + j;
                let g = g_h1[j] * x[i] + WEIGHT_DECAY * self.w1[idx];
                self.vw1[idx] = MOMENTUM * self.vw1[idx] - lr * g;
                self.w1[idx] += self.vw1[idx];
            }
        }
        for j in 0..HIDDEN {
            self.vb1[j] = MOMENTUM * self.vb1[j] - lr * g_h1[j];
            self.b1[j] += self.vb1[j];
        }
        loss
    }

    fn predict(&self, x: &[f64]) -> usize {
        let (_, logits) = self.forward(x);
        let mut best = 0usize;
        for j in 1..CLASSES { if logits[j] > logits[best] { best = j; } }
        best
    }
}

fn to_time_binned(sample: &ShdSample, num_neurons: usize, duration_ms: f64, t_bins: usize) -> Vec<f64> {
    let bin_size = (num_neurons as f64 / FREQ_BINS as f64).ceil().max(1.0) as usize;
    let time_bin_ms = duration_ms / t_bins as f64;
    let mut feats = vec![0.0f64; FREQ_BINS * t_bins];
    for (time, neuron) in &sample.spikes {
        let fidx = (*neuron as usize / bin_size).min(FREQ_BINS - 1);
        let tidx = ((*time / time_bin_ms) as usize).min(t_bins - 1);
        feats[tidx * FREQ_BINS + fidx] += 1.0;
    }
    let secs = time_bin_ms / 1000.0;
    for f in &mut feats { *f /= secs; }
    let norm: f64 = feats.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-9 { let s = 0.75 / norm; for f in &mut feats { *f *= s; } }
    feats
}

fn main() {
    let data_path = std::env::var("SHD_DATA").unwrap_or_else(|_| "data/shd/shd.json".to_string());
    let dataset = ShdDataset::from_json(Path::new(&data_path)).expect("SHD-Daten nicht gefunden.");
    let mode = std::env::var("SHD_FEAT").unwrap_or_else(|_| "rate".to_string());

    println!(
        "SHD: {} train, {} test, {} classes | feature={}",
        dataset.train.len(), dataset.test.len(), dataset.num_classes, mode
    );

    let featurize = |s: &ShdSample| match mode.as_str() {
        "ttfs" => ShdDataset::to_feature_vector_ttfs(s, dataset.num_neurons, dataset.duration_ms),
        "time" => to_time_binned(s, dataset.num_neurons, dataset.duration_ms, TIME_BINS),
        _ => ShdDataset::to_feature_vector(s, dataset.num_neurons, dataset.duration_ms),
    };

    let train_feats: Vec<Vec<f64>> = dataset.train.iter().map(featurize).collect();
    let test_feats: Vec<Vec<f64>> = dataset.test.iter().map(featurize).collect();
    let train_labels: Vec<usize> = dataset.train.iter().map(|s| s.label as usize).collect();
    let test_labels: Vec<usize> = dataset.test.iter().map(|s| s.label as usize).collect();

    let input_dim = train_feats[0].len();
    println!("feature dim = {}", input_dim);

    let mut model = Mlp::new(input_dim, 42);
    let epochs = if mode == "time" { 50 } else { 40 };
    let base_lr = 0.01;
    let mut rng = StdRng::seed_from_u64(7);

    let mut best_acc = 0.0f64;
    let mut best_epoch = 0usize;

    for epoch in 0..epochs {
        let lr = base_lr / (1.0 + 0.05 * epoch as f64);
        let mut total_loss = 0.0;
        for _ in 0..train_feats.len() {
            let idx = rng.gen_range(0..train_feats.len());
            total_loss += model.train_step(&train_feats[idx], train_labels[idx], lr);
        }
        let mut correct = 0usize;
        for (f, &l) in test_feats.iter().zip(&test_labels) {
            if model.predict(f) == l { correct += 1; }
        }
        let acc = correct as f64 / test_feats.len() as f64;
        if acc > best_acc { best_acc = acc; best_epoch = epoch; }
        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!("Epoch {:>2}: loss {:.4} | test acc {:.2}% ({}/{})",
                epoch, total_loss / train_feats.len() as f64, acc * 100.0, correct, test_feats.len());
        }
    }

    println!("Benchmark: shd-classifier-{}", mode);
    println!("Best accuracy: {:.2}% (epoch {})", best_acc * 100.0, best_epoch);
    println!("Status: {}", if best_acc > 0.6 { "STRONG" } else { "BASELINE" });
}
