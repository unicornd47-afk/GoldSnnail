//! SHD novelty / open-set gate — Block 3.
//!
//! Trains a 20-way linear softmax on 19 classes (class 0 held out = "novel"),
//! then measures a novelty gate using maximum softmax probability (MSP):
//!   - known-class accuracy (the 19 trained classes)
//!   - at several thresholds, how many held-out samples are flagged "unknown"
//!     (novelty recall) vs how many known samples are falsely flagged.
//!
//! Usage: cargo run --release --example eval_shd_novelty

use goldsnnail::audio::shd_loader::{ShdDataset, ShdSample};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

const INPUT: usize = 100;
const CLASSES: usize = 20;

struct Linear {
    w: Vec<f64>, vw: Vec<f64>, b: Vec<f64>, vb: Vec<f64>,
}

impl Linear {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut m = Self {
            w: vec![0.0; INPUT * CLASSES], vw: vec![0.0; INPUT * CLASSES],
            b: vec![0.0; CLASSES], vb: vec![0.0; CLASSES],
        };
        let s = (2.0 / INPUT as f64).sqrt();
        for w in m.w.iter_mut() { *w = rng.gen_range(-1.0..1.0) * s; }
        m
    }
    fn logits(&self, x: &[f64]) -> Vec<f64> {
        let mut l = vec![0.0f64; CLASSES];
        for j in 0..CLASSES {
            let mut s = self.b[j];
            for i in 0..INPUT { s += x[i] * self.w[i * CLASSES + j]; }
            l[j] = s;
        }
        l
    }
    fn softmax(&self, l: &[f64]) -> Vec<f64> {
        let max = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut p = vec![0.0f64; CLASSES];
        let mut s = 0.0;
        for j in 0..CLASSES { p[j] = (l[j] - max).exp(); s += p[j]; }
        for j in 0..CLASSES { p[j] /= s; }
        p
    }
    fn train_step(&mut self, x: &[f64], y: usize, lr: f64) -> f64 {
        let l = self.logits(x);
        let p = self.softmax(&l);
        let loss = -p[y].ln();
        let mut g = p;
        g[y] -= 1.0;
        for i in 0..INPUT {
            for j in 0..CLASSES {
                let idx = i * CLASSES + j;
                self.vw[idx] = 0.9 * self.vw[idx] - lr * (g[j] * x[i] + 1e-4 * self.w[idx]);
                self.w[idx] += self.vw[idx];
            }
        }
        for j in 0..CLASSES {
            self.vb[j] = 0.9 * self.vb[j] - lr * g[j];
            self.b[j] += self.vb[j];
        }
        loss
    }
    fn max_prob(&self, x: &[f64]) -> f64 {
        self.softmax(&self.logits(x)).into_iter().fold(0.0, f64::max)
    }
    fn predict(&self, x: &[f64]) -> usize {
        let l = self.logits(x);
        let mut best = 0usize;
        for j in 1..CLASSES { if l[j] > l[best] { best = j; } }
        best
    }
}

fn main() {
    let data_path = std::env::var("SHD_DATA").unwrap_or_else(|_| "data/shd/shd.json".to_string());
    let dataset = ShdDataset::from_json(Path::new(&data_path)).expect("SHD-Daten nicht gefunden.");

    let held_out: usize = 0; // class we never train on = "novel"
    let feats = |s: &ShdSample| ShdDataset::to_feature_vector(s, dataset.num_neurons, dataset.duration_ms);

    let train_known: Vec<(Vec<f64>, usize)> = dataset.train.iter()
        .filter(|s| s.label as usize != held_out)
        .map(|s| (feats(s), s.label as usize))
        .collect();
    let test_known: Vec<(Vec<f64>, usize)> = dataset.test.iter()
        .filter(|s| s.label as usize != held_out)
        .map(|s| (feats(s), s.label as usize))
        .collect();
    let test_novel: Vec<Vec<f64>> = dataset.test.iter()
        .filter(|s| s.label as usize == held_out)
        .map(|s| feats(s))
        .collect();

    println!("held-out class = {} | train known {} | test known {} | test novel {}",
        held_out, train_known.len(), test_known.len(), test_novel.len());

    let mut model = Linear::new(42);
    let epochs = 40;
    let base_lr = 0.01;
    let mut rng = StdRng::seed_from_u64(7);

    for epoch in 0..epochs {
        let lr = base_lr / (1.0 + 0.05 * epoch as f64);
        let mut loss = 0.0;
        for _ in 0..train_known.len() {
            let idx = rng.gen_range(0..train_known.len());
            loss += model.train_step(&train_known[idx].0, train_known[idx].1, lr);
        }
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("Epoch {}: loss {:.4}", epoch, loss / train_known.len() as f64);
        }
    }

    let mut correct = 0usize;
    for (x, y) in &test_known { if model.predict(x) == *y { correct += 1; } }
    let known_acc = correct as f64 / test_known.len() as f64;
    println!("Known-class accuracy (19-way): {:.2}% ({}/{})", known_acc * 100.0, correct, test_known.len());

    println!("--- Novelty gate (max softmax probability) ---");
    for &t in &[0.3f64, 0.4, 0.5, 0.6, 0.7] {
        let mut novel_hit = 0usize;
        for x in &test_novel { if model.max_prob(x) < t { novel_hit += 1; } }
        let mut known_false = 0usize;
        for (x, _) in &test_known { if model.max_prob(x) < t { known_false += 1; } }
        println!("  threshold {:.1}: novelty recall {:.1}% | known false-flag {:.1}%",
            t,
            novel_hit as f64 / test_novel.len() as f64 * 100.0,
            known_false as f64 / test_known.len() as f64 * 100.0);
    }
}
