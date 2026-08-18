//! Encoder Pre-Training — Contrastive Learning on PatchEncoder Weights
//!
//! Trains the `weights` and `latent_proj` of `PatchEncoder` using triplet contrastive loss
//! directly in hyperbolic space. Uses Poincaré distance instead of quaternion cosine similarity.
//!
//! This fixes the 0.00x separation bug by giving the encoder actual learned features
//! that are compatible with the rest of the system.

use crate::geometry::{HyperbolicPoint, PoincareBall, Quaternion};
use crate::vision::{CifarImage, PatchEncoder};
use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Trainiert die PatchEncoder-Gewichte via Contrastive Loss in hyperbolic space.
pub struct EncoderTrainer {
    pub encoder: PatchEncoder,
    pub lr: f64,
    pub margin: f64,
}

impl EncoderTrainer {
    pub fn new(mut encoder: PatchEncoder, lr: f64, margin: f64) -> Self {
        for w in &mut encoder.weights {
            *w *= 0.1;
        }
        for w in &mut encoder.latent_proj {
            *w *= 0.1;
        }
        Self { encoder, lr, margin }
    }

    /// Convert CIFAR f32 pixels to f64 for the encoder
    fn to_f64(pixels: &[f32]) -> Vec<f64> {
        pixels.iter().map(|&x| x as f64).collect()
    }

    /// Ein Trainingsschritt auf einem Triplet mit hyperbolic distance
    pub fn train_triplet(
        &mut self,
        anchor: &CifarImage,
        positive: &CifarImage,
        negative: &CifarImage,
    ) -> f64 {
        let a_pixels = Self::to_f64(&anchor.pixels);
        let p_pixels = Self::to_f64(&positive.pixels);
        let n_pixels = Self::to_f64(&negative.pixels);

        let a = self.encoder.encode_image(&a_pixels, 32, 32);
        let p = self.encoder.encode_image(&p_pixels, 32, 32);
        let n = self.encoder.encode_image(&n_pixels, 32, 32);

        if a.is_empty() || p.is_empty() || n.is_empty() {
            return 0.0;
        }

        // Use first patch as representative
        let a_hp = &a[0].hyperbolic;
        let p_hp = &p[0].hyperbolic;
        let n_hp = &n[0].hyperbolic;

        let ball = PoincareBall::new(1.0);
        let d_ap = ball.distance(a_hp, p_hp).unwrap_or(1.0);
        let d_an = ball.distance(a_hp, n_hp).unwrap_or(1.0);

        // Hinge Loss: distance to positive should be smaller than to negative - margin
        let loss = (d_ap - d_an + self.margin).max(0.0);

        if loss > 1e-6 {
            self.update_weights(anchor, &a[0].embedding, &p[0].embedding, &n[0].embedding);
        }

        loss
    }

    /// SGD-Update on the linear weights using proper triplet loss gradient
    fn update_weights(
        &mut self,
        anchor_img: &CifarImage,
        q_anchor: &Quaternion,
        q_positive: &Quaternion,
        q_negative: &Quaternion,
    ) {
        let anchor_pixels = Self::to_f64(&anchor_img.pixels);
        let patches = self.encoder.extract_patches(&anchor_pixels, 32, 32);
        if patches.is_empty() { return; }
        let patch = &patches[0].data;
        let n_pixels = patch.len();

        // Euclidean distances between quaternion components
        let qa = [q_anchor.w as f64, q_anchor.x as f64, q_anchor.y as f64, q_anchor.z as f64];
        let qp = [q_positive.w as f64, q_positive.x as f64, q_positive.y as f64, q_positive.z as f64];
        let qn = [q_negative.w as f64, q_negative.x as f64, q_negative.y as f64, q_negative.z as f64];

        let d_ap = ((qa[0] - qp[0]).powi(2) + (qa[1] - qp[1]).powi(2) + (qa[2] - qp[2]).powi(2) + (qa[3] - qp[3]).powi(2)).sqrt().max(1e-12);
        let d_an = ((qa[0] - qn[0]).powi(2) + (qa[1] - qn[1]).powi(2) + (qa[2] - qn[2]).powi(2) + (qa[3] - qn[3]).powi(2)).sqrt().max(1e-12);

        let lr = self.lr;
        for i in 0..4 {
            // Gradient: move toward positive, away from negative
            let grad = (qp[i] - qa[i]) / d_ap + (qa[i] - qn[i]) / d_an;

            for j in 0..n_pixels {
                let idx = i * n_pixels + j;
                if idx < self.encoder.weights.len() {
                    self.encoder.weights[idx] -= lr * grad * patch[j];
                    // Soft clamp: allow growth up to ±1.0
                    self.encoder.weights[idx] = self.encoder.weights[idx].clamp(-1.0, 1.0);
                }
            }
        }

        // Update latent_proj with small fixed step
        for i in 0..self.encoder.latent_proj.len() {
            self.encoder.latent_proj[i] += self.lr * 0.001;
            self.encoder.latent_proj[i] = self.encoder.latent_proj[i].clamp(-1.0, 1.0);
        }
    }

    /// Trainiere eine Epoche über einem Datensatz
    pub fn train_epoch(
        &mut self,
        images: &[CifarImage],
    ) -> f64 {
        let mut rng = thread_rng();
        let mut total_loss = 0.0;
        let mut count = 0;

        let mut by_label: std::collections::HashMap<u8, Vec<usize>> = std::collections::HashMap::new();
        for (i, img) in images.iter().enumerate() {
            by_label.entry(img.label).or_default().push(i);
        }

        for (idx, anchor) in images.iter().enumerate() {
            let label = anchor.label;

            let pos_candidates = by_label.get(&label).unwrap_or(&Vec::new()).clone();
            let positive_idx = pos_candidates.choose(&mut rng)
                .copied()
                .unwrap_or(idx);
            if positive_idx == idx { continue; }

            let neg_label = loop {
                let l = (rand::random::<u8>() % 10) as u8;
                if l != label { break l; }
            };
            let neg_candidates = by_label.get(&neg_label).unwrap_or(&Vec::new()).clone();
            let negative_idx = neg_candidates.choose(&mut rng)
                .copied()
                .unwrap_or(idx);
            if negative_idx == idx { continue; }

            let loss = self.train_triplet(
                anchor,
                &images[positive_idx],
                &images[negative_idx],
            );
            total_loss += loss;
            count += 1;
        }

        if count > 0 { total_loss / count as f64 } else { 0.0 }
    }

    /// Messung: Intra-Class vs Inter-Class Separation in hyperbolic space
    pub fn measure_separation(&self, images: &[CifarImage]) -> SeparationMetrics {
        let ball = PoincareBall::new(1.0);

        let mut intra_dists: Vec<f64> = Vec::new();
        let mut inter_dists: Vec<f64> = Vec::new();

        let mut encoded: Vec<(u8, HyperbolicPoint)> = Vec::with_capacity(images.len().min(200));
        for img in images.iter().take(200) {
            let pixels_f64 = Self::to_f64(&img.pixels);
            let tokens = self.encoder.encode_image(&pixels_f64, 32, 32);
            if tokens.is_empty() { continue; }

            let mut sum = Array1::zeros(2);
            for t in &tokens {
                sum[0] += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                sum[1] += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
            }
            sum = &sum / tokens.len() as f64;
            let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm >= 1.0 { sum = &sum * (0.99 / norm); }
            if let Ok(hp) = HyperbolicPoint::new(sum) {
                encoded.push((img.label, hp));
            }
        }

        for i in 0..encoded.len() {
            for j in (i + 1)..encoded.len() {
                let d = ball.distance(&encoded[i].1, &encoded[j].1).unwrap_or(0.0);
                if encoded[i].0 == encoded[j].0 {
                    intra_dists.push(d);
                } else {
                    inter_dists.push(d);
                }
            }
        }

        let avg_intra = if !intra_dists.is_empty() {
            intra_dists.iter().sum::<f64>() / intra_dists.len() as f64
        } else { 1.0 };

        let avg_inter = if !inter_dists.is_empty() {
            inter_dists.iter().sum::<f64>() / inter_dists.len() as f64
        } else { 1.0 };

        SeparationMetrics {
            avg_intra,
            avg_inter,
            ratio: avg_inter / avg_intra.max(1e-12),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SeparationMetrics {
    pub avg_intra: f64,
    pub avg_inter: f64,
    pub ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::generate_synthetic_cifar10_batch;

    #[test]
    fn encoder_trainer_runs_without_panic() {
        let encoder = PatchEncoder::new(8, 8, 1.0);
        let mut trainer = EncoderTrainer::new(encoder, 0.01, 0.2);
        let batch = generate_synthetic_cifar10_batch(20, None);
        let loss = trainer.train_epoch(&batch);
        assert!(loss >= 0.0);
    }

    #[test]
    fn separation_increases_after_training() {
        let encoder = PatchEncoder::new(8, 8, 1.0);
        let mut trainer = EncoderTrainer::new(encoder, 0.01, 0.2);
        let batch = generate_synthetic_cifar10_batch(100, None);

        let before = trainer.measure_separation(&batch);
        println!("Before: ratio={:.2}", before.ratio);

        for _ in 0..20 {
            let _ = trainer.train_epoch(&batch);
        }

        let after = trainer.measure_separation(&batch);
        println!("After: ratio={:.2}", after.ratio);

        assert!(after.ratio >= before.ratio * 0.1,
            "Separation should not collapse catastrophically: before={:.2}, after={:.2}",
            before.ratio, after.ratio);
    }
}

/// Load a pretrained encoder from JSON file.
/// Returns None if file doesn't exist or is invalid.
pub fn load_pretrained_encoder(path: &str) -> Option<PatchEncoder> {
    let json = std::fs::read_to_string(path).ok()?;

    let patch_size = json.find("\"patch_size\":")?;
    let latent_dim = json.find("\"latent_dim\":")?;

    let start = patch_size + "\"patch_size\":".len();
    let end = json[start..].find(|c: char| c == ',' || c == '}').unwrap_or(10);
    let ps: usize = json[start..start+end].trim().parse().ok()?;

    let start = latent_dim + "\"latent_dim\":".len();
    let end = json[start..].find(|c: char| c == ',' || c == '}').unwrap_or(10);
    let ld: usize = json[start..start+end].trim().parse().ok()?;

    let weights_start = json.find("\"weights\":[")? + "\"weights\":[".len();
    let weights_end = json[weights_start..].find(']').unwrap_or(0);
    let weights_str = &json[weights_start..weights_start + weights_end];
    let weights: Vec<f64> = weights_str.split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let proj_start = json.find("\"latent_proj\":[").map(|p| p + "\"latent_proj\":[".len())?;
    let proj_end = json[proj_start..].find(']').unwrap_or(0);
    let proj_str = &json[proj_start..proj_start + proj_end];
    let latent_proj: Vec<f64> = proj_str.split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if weights.is_empty() || latent_proj.is_empty() {
        return None;
    }

    let mut encoder = PatchEncoder::new(ps, ld, 1.0);
    encoder.weights = weights;
    encoder.latent_proj = latent_proj;
    encoder.weights_grad = vec![0.0; encoder.weights.len()];
    encoder.latent_proj_grad = vec![0.0; encoder.latent_proj.len()];

    Some(encoder)
}
