//! Geometric Compression Bottleneck
//!
//! Event-driven delta-encoder with hyperbolic latent space.
//! Sends only when the manifold state changes significantly.

use crate::geometry::{Quaternion, PoincareBall, HyperbolicPoint};
use crate::substrate::SpikeBuffer;
use crate::LabError;
use ndarray::Array1;

/// Event-driven Delta-Encoder with Hyperbolic-Latent-Space.
/// Sends only when the manifold state changes significantly.
#[derive(Debug, Clone)]
pub struct GeometricBottleneck {
    pub delta_threshold: f64,
    pub latent_dim: usize,
    pub input_dim: usize,
    pub ball: PoincareBall,
    /// Flat projection weights: [latent_dim x input_dim]
    pub projection: Vec<f64>,
    last_latent: Option<HyperbolicPoint>,
}

impl GeometricBottleneck {
    pub fn new(
        input_dim: usize,
        latent_dim: usize,
        delta_threshold: f64,
        curvature: f64,
    ) -> Self {
        assert!(latent_dim > 0 && input_dim > 0);
        let mut projection = vec![0.0f64; latent_dim * input_dim];
        for i in 0..projection.len() {
            let x = (i as f64 + 1.0) / (projection.len() as f64 + 1.0);
            projection[i] = (x * std::f64::consts::TAU).sin() * 0.1;
        }

        Self {
            delta_threshold,
            latent_dim,
            input_dim,
            ball: PoincareBall::new(curvature),
            projection,
            last_latent: None,
        }
    }

    /// Compresses a spike batch into a HyperbolicPoint.
    /// Returns `None` if the delta distance is below threshold (event-driven: do not send).
    pub fn compress(
        &mut self,
        spikes: &SpikeBuffer,
        phases: &[Quaternion],
    ) -> Result<Option<HyperbolicPoint>, LabError> {
        if spikes.count == 0 {
            return Ok(None);
        }

        let mut latent = vec![0.0f64; self.latent_dim];

        for j in 0..self.latent_dim {
            let mut acc = 0.0;
            for k in 0..spikes.indices.len() {
                let n = spikes.indices[k] as usize;
                if n >= self.input_dim {
                    continue;
                }
                let w = self.projection[j * self.input_dim + n];
                acc += w * phases[n].norm() as f64;
            }
            latent[j] = acc.tanh() * 0.9;
        }

        let norm = latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm;
            for x in &mut latent {
                *x *= scale;
            }
        }
        let point = HyperbolicPoint::new(Array1::from(latent))?;

        if let Some(ref last) = self.last_latent {
            let dist = self.ball.distance(last, &point)?;
            if dist < self.delta_threshold {
                return Ok(None);
            }
        }

        self.last_latent = Some(point.clone());
        Ok(Some(point))
    }

    /// Explicit reset (e.g., on context switch)
    pub fn reset(&mut self) {
        self.last_latent = None;
    }

    /// Direct distance between two latent points (for World Model loss)
    pub fn latent_distance(
        &self,
        a: &HyperbolicPoint,
        b: &HyperbolicPoint,
    ) -> Result<f64, LabError> {
        self.ball.distance(a, b)
    }
}

/// Cross-Scale Router: Routes compressed packets between regions
#[derive(Debug, Clone)]
pub struct CompressionRouter {
    pub bottlenecks: Vec<GeometricBottleneck>,
}

impl CompressionRouter {
    pub fn new(count: usize, input_dim: usize, latent_dim: usize) -> Self {
        let bottlenecks = (0..count)
            .map(|_| GeometricBottleneck::new(input_dim, latent_dim, 0.1, 1.0))
            .collect();
        Self { bottlenecks }
    }

    /// Process all regions logically and collect outputs
    pub fn route_all(
        &mut self,
        inputs: &[(&SpikeBuffer, &[Quaternion])],
    ) -> Result<Vec<Option<HyperbolicPoint>>, LabError> {
        let mut out = Vec::with_capacity(self.bottlenecks.len());
        for (i, (spikes, phases)) in inputs.iter().enumerate() {
            if i >= self.bottlenecks.len() {
                break;
            }
            out.push(self.bottlenecks[i].compress(spikes, phases)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_spikes(ids: &[u32]) -> SpikeBuffer {
        let mut buf = SpikeBuffer::new(ids.len().max(100));
        for &id in ids {
            buf.push(id).unwrap();
        }
        buf
    }

    #[test]
    fn bottleneck_sends_on_first_spike() {
        let mut bn = GeometricBottleneck::new(10, 2, 0.1, 1.0);
        let spikes = fake_spikes(&[0, 2, 4]);
        let phases = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); 10];

        let result = bn.compress(&spikes, &phases).unwrap();
        assert!(result.is_some(), "First spike must always send");
    }

    #[test]
    fn bottleneck_suppresses_similar_input() {
        let mut bn = GeometricBottleneck::new(10, 2, 10.0, 1.0);
        let spikes = fake_spikes(&[0, 1]);
        let phases = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); 10];

        let _ = bn.compress(&spikes, &phases).unwrap();
        let result = bn.compress(&spikes, &phases).unwrap();
        assert!(result.is_none(), "Identical input should be suppressed");
    }

    #[test]
    fn bottleneck_resets_properly() {
        let mut bn = GeometricBottleneck::new(10, 2, 0.1, 1.0);
        let spikes = fake_spikes(&[3]);
        let phases = vec![Quaternion::new(2.0, 0.0, 0.0, 0.0); 10];

        let _ = bn.compress(&spikes, &phases).unwrap();
        bn.reset();
        let result = bn.compress(&spikes, &phases).unwrap();
        assert!(result.is_some(), "After reset must send again");
    }

    #[test]
    fn latent_point_inside_ball() {
        let mut bn = GeometricBottleneck::new(50, 4, 0.1, 1.0);
        let spikes = fake_spikes(&(0..20).collect::<Vec<_>>());
        let phases: Vec<Quaternion> = (0..50)
            .map(|i| Quaternion::new(i as f32, 0.0, 0.0, 0.0))
            .collect();

        let point = bn.compress(&spikes, &phases).unwrap().unwrap();
        assert!(point.euclidean_norm() < 1.0, "Latent point must be inside Poincaré ball");
    }

    #[test]
    fn router_collects_outputs() {
        let mut router = CompressionRouter::new(3, 10, 2);
        let spikes = fake_spikes(&[0, 1]);
        let phases = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); 10];

        let inputs = vec![
            (&spikes, phases.as_slice()),
            (&spikes, phases.as_slice()),
            (&spikes, phases.as_slice()),
        ];

        let outs = router.route_all(&inputs).unwrap();
        assert_eq!(outs.len(), 3);
        assert!(outs.iter().all(|o| o.is_some()));
    }
}


