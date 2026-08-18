//! Patch Encoder — 2D Image Patches → Hyperbolic Quaternion Embeddings
//!
//! Extracts local image patches and encodes them into the hyperbolic semantic
//! space using a shared quaternion projection. Visual tokens can be bound to
//! semantic labels via contrastive updates.

use crate::geometry::{HyperbolicPoint, PoincareBall, Quaternion};
use crate::LabError;
use crate::SemanticEncoder;
use ndarray::{array, Array1};

/// A rectangular patch extracted from an image.
#[derive(Debug, Clone)]
pub struct ImagePatch {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl ImagePatch {
    /// Creates a new patch from flat pixel data.
    pub fn new(width: usize, height: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }

    /// Returns the mean intensity of the patch.
    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Returns the standard deviation of the patch.
    pub fn std(&self) -> f64 {
        let m = self.mean();
        (self.data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.data.len() as f64).sqrt()
    }
}

/// A visual token in the hyperbolic semantic space.
#[derive(Debug, Clone)]
pub struct VisualToken {
    pub patch: ImagePatch,
    pub embedding: Quaternion,
    pub hyperbolic: HyperbolicPoint,
    pub label: String,
    pub salience: f64,
}

/// Encodes image patches into hyperbolic quaternion embeddings.
///
/// DOD: flat weight matrices, no nested allocations in the hot path.
#[derive(Clone)]
pub struct PatchEncoder {
    pub patch_size: usize,
    pub latent_dim: usize,
    pub ball: PoincareBall,
    pub weights: Vec<f64>,
    pub latent_proj: Vec<f64>,
    pub weights_grad: Vec<f64>,
    pub latent_proj_grad: Vec<f64>,
    semantic: Option<SemanticEncoder>,
}

impl PatchEncoder {
    /// Creates a new patch encoder.
    pub fn new(patch_size: usize, latent_dim: usize, curvature: f64) -> Self {
        let weights: Vec<f64> = (0..patch_size * patch_size * 4)
            .map(|i| {
                let px = (i % patch_size) as f64;
                let py = ((i / patch_size) % patch_size) as f64;
                let comp = (i / (patch_size * patch_size)) as f64;
                let spatial = ((px + py) * 0.3).sin() + 0.3; // bias to avoid zero-mean collapse
                let channel = (comp * 2.1).cos();
                spatial * channel * 0.8
            })
            .collect();
        let latent_proj: Vec<f64> = (0..latent_dim * 4)
            .map(|i| {
                let dim = (i % latent_dim) as f64;
                let comp = (i / latent_dim) as f64;
                (dim * 0.7 + comp * 2.3).sin() * 0.5
            })
            .collect();

        let w_len = weights.len();
        let l_len = latent_proj.len();
        Self {
            patch_size,
            latent_dim,
            ball: PoincareBall::new(curvature),
            weights,
            latent_proj,
            weights_grad: vec![0.0; w_len],
            latent_proj_grad: vec![0.0; l_len],
            semantic: None,
        }
    }

    /// Attaches a semantic encoder for multimodal binding.
    pub fn with_semantic(mut self, encoder: SemanticEncoder) -> Self {
        self.semantic = Some(encoder);
        self
    }

    /// Extracts non-overlapping patches from a flat image (height × width).
    /// Supports both grayscale (`len == width * height`) and RGB (`len == width * height * 3`).
    pub fn extract_patches(&self, image: &[f64], img_width: usize, img_height: usize) -> Vec<ImagePatch> {
        let ps = self.patch_size;
        let expected = img_width * img_height;
        let is_rgb = image.len() == expected * 3;
        assert!(image.len() == expected || is_rgb, "Image length {} does not match {}x{} ({}) or RGB ({}", 
            image.len(), img_width, img_height, expected, expected * 3);
        
        let mut patches = Vec::new();

        for y in (0..img_height).step_by(ps) {
            for x in (0..img_width).step_by(ps) {
                let mut data = Vec::with_capacity(ps * ps);
                for dy in 0..ps {
                    for dx in 0..ps {
                        let px = x + dx;
                        let py = y + dy;
                        if px < img_width && py < img_height {
                            let idx = py * img_width + px;
                            let val = if is_rgb {
                                let r = image[idx];
                                let g = image[expected + idx];
                                let b = image[expected * 2 + idx];
                                0.299 * r + 0.587 * g + 0.114 * b
                            } else {
                                image[idx]
                            };
                            data.push(val);
                        } else {
                            data.push(0.0);
                        }
                    }
                }
                patches.push(ImagePatch::new(ps, ps, data));
            }
        }
        patches
    }

    /// Encodes a single patch to a quaternion embedding.
    pub fn encode_patch(&self, patch: &ImagePatch) -> Quaternion {
        let mut w = 0.0;
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for (i, &v) in patch.data.iter().enumerate() {
            let base = i * 4;
            w += self.weights[base] * v;
            x += self.weights[base + 1] * v;
            y += self.weights[base + 2] * v;
            z += self.weights[base + 3] * v;
        }

        Quaternion::new(w as f32, x as f32, y as f32, z as f32).normalize()
    }

    /// Encodes an image to a list of visual tokens.
    pub fn encode_image(&self, image: &[f64], img_width: usize, img_height: usize) -> Vec<VisualToken> {
        let patches = self.extract_patches(image, img_width, img_height);
        patches
            .into_iter()
            .map(|patch| {
                let q = self.encode_patch(&patch);
                let h = self.to_hyperbolic(&q).unwrap();
                VisualToken {
                    patch,
                    embedding: q,
                    hyperbolic: h,
                    label: String::new(),
                    salience: 1.0,
                }
            })
            .collect()
    }

    /// Binds a visual token to a semantic label, shifting its hyperbolic position
    /// toward the label's embedding in the shared semantic space.
    pub fn bind_visual_semantic(&mut self, token: &mut VisualToken, label: &str) -> Result<(), LabError> {
        let semantic = match &self.semantic {
            Some(s) => s,
            None => return Err(LabError::InvalidState),
        };

        let q = semantic.encode_token(label).ok_or(LabError::InvalidState)?;
        let target = semantic.to_hyperbolic(&q)?;

        let mut new_coords = token.hyperbolic.coords.clone();
        for i in 0..new_coords.len() {
            new_coords[i] += (target.coords[i] - new_coords[i]) * 0.3;
        }

        let norm = new_coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm;
            for x in &mut new_coords {
                *x *= scale;
            }
        }

        token.hyperbolic = HyperbolicPoint::new(Array1::from(new_coords))?;
        token.label = label.to_string();
        Ok(())
    }

    /// Projects a quaternion into the hyperbolic latent space.
    pub fn to_hyperbolic(&self, q: &Quaternion) -> Result<HyperbolicPoint, LabError> {
        let mut latent = vec![0.0f64; self.latent_dim];
        let comps = [q.w as f64, q.x as f64, q.y as f64, q.z as f64];
        for i in 0..self.latent_dim {
            let mut acc = 0.0;
            for j in 0..4 {
                acc += self.latent_proj[i * 4 + j] * comps[j];
            }
            latent[i] = acc.tanh() * 0.95;
        }
        let norm = latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm;
            for x in &mut latent {
                *x *= scale;
            }
        }
        HyperbolicPoint::new(Array1::from(latent))
    }

    /// Encodes a single patch to a RAW quaternion (no normalization).
    /// Preserves magnitude information for discriminative power.
    pub fn encode_patch_raw(&self, patch: &ImagePatch) -> Quaternion {
        let mut w = 0.0;
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for (i, &v) in patch.data.iter().enumerate() {
            let base = i * 4;
            w += self.weights[base] * v;
            x += self.weights[base + 1] * v;
            y += self.weights[base + 2] * v;
            z += self.weights[base + 3] * v;
        }

        Quaternion::new(w as f32, x as f32, y as f32, z as f32)
    }

    /// Projects a quaternion into hyperbolic space WITHOUT tanh compression.
    /// Uses linear projection with elastic boundary enforcement.
    pub fn to_hyperbolic_raw(&self, q: &Quaternion) -> Result<HyperbolicPoint, LabError> {
        let mut latent = vec![0.0f64; self.latent_dim];
        let comps = [q.w as f64, q.x as f64, q.y as f64, q.z as f64];
        for i in 0..self.latent_dim {
            let mut acc = 0.0;
            for j in 0..4 {
                acc += self.latent_proj[i * 4 + j] * comps[j];
            }
            latent[i] = acc;
        }
        // Elastic boundary: if norm >= 1.0, scale down while preserving direction
        let norm = latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm;
            for x in &mut latent {
                *x *= scale;
            }
        }
        HyperbolicPoint::new(Array1::from(latent))
    }

    /// Encodes an image using RAW embeddings (no normalization, no tanh).
    pub fn encode_image_raw(&self, image: &[f64], img_width: usize, img_height: usize) -> Vec<VisualToken> {
        let patches = self.extract_patches(image, img_width, img_height);
        patches
            .into_iter()
            .map(|patch| {
                let q = self.encode_patch_raw(&patch);
                let h = self.to_hyperbolic_raw(&q).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0; self.latent_dim] });
                VisualToken {
                    patch,
                    embedding: q,
                    hyperbolic: h,
                    label: String::new(),
                    salience: 1.0,
                }
            })
            .collect()
    }

    /// Zero all accumulated gradients.
    pub fn zero_grad(&mut self) {
        for g in &mut self.weights_grad {
            *g = 0.0;
        }
        for g in &mut self.latent_proj_grad {
            *g = 0.0;
        }
    }

    /// Performs a single SGD step on all trainable parameters.
    /// `lr` = learning rate
    pub fn step(&mut self, lr: f64) {
        for (w, g) in self.weights.iter_mut().zip(&mut self.weights_grad) {
            *w -= lr * *g;
        }
        for (w, g) in self.latent_proj.iter_mut().zip(&mut self.latent_proj_grad) {
            *w -= lr * *g;
        }
    }

    /// Encodes a single patch AND caches intermediates for backward.
    pub fn encode_patch_trainable(&self, patch: &ImagePatch) -> (Vec<f64>, Vec<f64>) {
        let mut w = 0.0;
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for (i, &v) in patch.data.iter().enumerate() {
            let base = i * 4;
            w += self.weights[base] * v;
            x += self.weights[base + 1] * v;
            y += self.weights[base + 2] * v;
            z += self.weights[base + 3] * v;
        }

        let comps = vec![w, x, y, z];
        (patch.data.clone(), comps)
    }

    /// Projects quaternion to hyperbolic space AND caches intermediates.
    pub fn to_hyperbolic_trainable(&self, q_comps: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut latent = vec![0.0f64; self.latent_dim];
        for i in 0..self.latent_dim {
            let mut acc = 0.0;
            for j in 0..4 {
                acc += self.latent_proj[i * 4 + j] * q_comps[j];
            }
            latent[i] = acc.tanh() * 0.95;
        }
        (q_comps.to_vec(), latent)
    }

    /// Backward pass: dL/d_weights and dL/d_latent_proj from output gradient dL/d_latent.
    pub fn backward(
        &mut self,
        d_latent: &[f64],
        cached_q: &[f64],
        cached_patch: &[f64],
    ) {
        // dL/d_latent_before_tanh = dL/d_latent * 0.95 * (1 - tanh^2)
        let mut d_latent_pre = vec![0.0f64; self.latent_dim];
        for i in 0..self.latent_dim {
            let t = d_latent[i].tanh();
            d_latent_pre[i] = d_latent[i] * 0.95 * (1.0 - t * t);
        }

        // Accumulate into latent_proj_grad
        for i in 0..self.latent_dim {
            for j in 0..4 {
                let idx = i * 4 + j;
                self.latent_proj_grad[idx] += d_latent_pre[i] * cached_q[j];
            }
        }

        // dL/d_q = latent_proj^T * d_latent_pre
        let mut d_q = vec![0.0f64; 4];
        for i in 0..self.latent_dim {
            for j in 0..4 {
                d_q[j] += self.latent_proj[i * 4 + j] * d_latent_pre[i];
            }
        }

        // Accumulate into weights_grad
        for (i, &v) in cached_patch.iter().enumerate() {
            let base = i * 4;
            self.weights_grad[base] += d_q[0] * v;
            self.weights_grad[base + 1] += d_q[1] * v;
            self.weights_grad[base + 2] += d_q[2] * v;
            self.weights_grad[base + 3] += d_q[3] * v;
        }
    }
}

/// Generates a synthetic test image with the requested pattern.
pub fn generate_test_image(pattern: &str, width: usize, height: usize) -> Vec<f64> {
    let mut image = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            image[idx] = match pattern {
                "gradient" => (x as f64 / width as f64).clamp(0.0, 1.0),
                "horizontal_stripes" => {
                    if (y / (height / 4)) % 2 == 0 { 0.8 } else { 0.2 }
                }
                "vertical_stripes" => {
                    if (x / (width / 4)) % 2 == 0 { 0.8 } else { 0.2 }
                }
                "checkerboard" => {
                    if ((x / (width / 8)) + (y / (height / 8))) % 2 == 0 { 0.9 } else { 0.1 }
                }
                _ => 0.5,
            };
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_patches_covers_image() {
        let enc = PatchEncoder::new(4, 2, 1.0);
        let img = generate_test_image("gradient", 8, 8);
        let patches = enc.extract_patches(&img, 8, 8);
        assert_eq!(patches.len(), 4);
    }

    #[test]
    fn encode_patch_produces_unit_quaternion() {
        let enc = PatchEncoder::new(4, 2, 1.0);
        let patch = ImagePatch::new(4, 4, vec![0.5; 16]);
        let q = enc.encode_patch(&patch);
        assert!((q.norm() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn encode_image_returns_visual_tokens() {
        let enc = PatchEncoder::new(4, 2, 1.0);
        let img = generate_test_image("horizontal_stripes", 8, 8);
        let tokens = enc.encode_image(&img, 8, 8);
        assert_eq!(tokens.len(), 4);
        assert!(tokens[0].salience > 0.0);
    }

    #[test]
    fn multimodal_patch_to_hyperbolic() {
        let enc = PatchEncoder::new(4, 2, 1.0);
        let patch = ImagePatch::new(4, 4, vec![0.8; 16]);
        let q = enc.encode_patch(&patch);
        let h = enc.to_hyperbolic(&q).unwrap();
        assert!(h.euclidean_norm() < 1.0, "Hyperbolic point must lie inside the Poincaré ball");
    }

    #[test]
    fn visual_semantic_binding_shifts_toward_label() {
        let vocab = vec!["cat".into(), "dog".into(), "tree".into(), "car".into()];
        let semantic = SemanticEncoder::new(vocab.clone(), 2);
        let mut enc = PatchEncoder::new(4, 2, 1.0).with_semantic(semantic);

        let patch = ImagePatch::new(4, 4, vec![0.7; 16]);
        let mut token = VisualToken {
            patch,
            embedding: Quaternion::new(0.0, 0.0, 0.0, 0.0),
            hyperbolic: HyperbolicPoint::new(array![0.0, 0.0]).unwrap(),
            label: String::new(),
            salience: 1.0,
        };
        token.embedding = enc.encode_patch(&token.patch);
        token.hyperbolic = enc.to_hyperbolic(&token.embedding).unwrap();

        let before = token.hyperbolic.coords.clone();
        enc.bind_visual_semantic(&mut token, "dog").unwrap();

        let after = token.hyperbolic.coords.clone();
        for i in 0..before.len() {
            let shifted = (after[i] - before[i]).abs();
            assert!(shifted > 1e-6, "Binding should shift the visual token toward its label");
        }
    }

    #[test]
    fn test_image_patterns() {
        let grad = generate_test_image("gradient", 8, 8);
        let h_stripes = generate_test_image("horizontal_stripes", 8, 8);
        let v_stripes = generate_test_image("vertical_stripes", 8, 8);
        let checker = generate_test_image("checkerboard", 8, 8);

        assert_eq!(grad.len(), 64);
        assert_eq!(h_stripes.len(), 64);
        assert_eq!(v_stripes.len(), 64);
        assert_eq!(checker.len(), 64);

        let g_min = grad.iter().cloned().fold(f64::INFINITY, f64::min);
        let g_max = grad.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((g_max - g_min).abs() > 0.5, "Gradient should span a range");
    }
}
