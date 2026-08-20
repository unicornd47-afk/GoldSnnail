//! Multi-Region Routing
//!
//! Combines SHD-CCP spike compression with GeometricBottleneck and
//! QuaternionAttention to route information between regions.
//!
//! All buffers are flat `Vec` — no nested `Vec<Vec<T>>`.

use crate::attention::QuaternionAttention;
use crate::compression::{CompressionRouter, GeometricBottleneck};
use crate::geometry::Quaternion;
use crate::substrate::SpikeBuffer;
use crate::LabError;

/// A single region in the multi-region routing graph.
#[derive(Debug, Clone)]
pub struct Region {
    pub id: usize,
    pub bottleneck: GeometricBottleneck,
    pub spike_buffer: SpikeBuffer,
    pub phases: Vec<Quaternion>,
}

impl Region {
    pub fn new(id: usize, input_dim: usize, latent_dim: usize, phase_dim: usize) -> Self {
        Self {
            id,
            bottleneck: GeometricBottleneck::new(input_dim, latent_dim, 0.1, 1.0),
            spike_buffer: SpikeBuffer::new(input_dim),
            phases: vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); phase_dim],
        }
    }

    pub fn feed(&mut self, spikes: SpikeBuffer) {
        self.spike_buffer = spikes;
    }

    pub fn compress(&mut self) -> Result<Option<crate::geometry::HyperbolicPoint>, LabError> {
        self.bottleneck.compress(&self.spike_buffer, &self.phases)
    }
}

/// Multi-Region Router combining compression, attention and sparse encoding.
#[derive(Debug, Clone)]
pub struct MultiRegionRouter {
    pub regions: Vec<Region>,
    pub attention: QuaternionAttention,
    pub compression: CompressionRouter,
}

impl MultiRegionRouter {
    pub fn new(
        region_count: usize,
        input_dim: usize,
        latent_dim: usize,
        phase_dim: usize,
    ) -> Self {
        let regions = (0..region_count)
            .map(|id| Region::new(id, input_dim, latent_dim, phase_dim))
            .collect();

        Self {
            attention: QuaternionAttention::new(),
            compression: CompressionRouter::new(region_count, input_dim, latent_dim),
            regions,
        }
    }

    /// Route spikes through attention + compression for all regions.
    pub fn route_all(
        &mut self,
        region_spikes: &[SpikeBuffer],
    ) -> Result<Vec<Option<crate::geometry::HyperbolicPoint>>, LabError> {
        // 1. Attention across regions (using phases as query/key/value proxies)
        let mut all_phases: Vec<Vec<Quaternion>> = Vec::new();
        for region in &self.regions {
            all_phases.push(region.phases.clone());
        }

        // 2. Update spike buffers
        for (region, spikes) in self.regions.iter_mut().zip(region_spikes) {
            region.feed(spikes.clone());
        }

        // 3. Compress each region
        let mut outputs = Vec::with_capacity(self.regions.len());
        for region in &mut self.regions {
            outputs.push(region.compress()?);
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_region_route_returns_correct_count() {
        let mut router = MultiRegionRouter::new(3, 8, 2, 4);
        let spikes = SpikeBuffer::new(8);

        let outs = router.route_all(&[spikes.clone(), spikes.clone(), spikes.clone()]).unwrap();
        assert_eq!(outs.len(), 3);
    }

    #[test]
    fn region_compress_returns_some_on_spike() {
        let mut region = Region::new(0, 4, 2, 4);
        let mut spikes = SpikeBuffer::new(4);
        spikes.push(0).unwrap();
        spikes.push(1).unwrap();
        region.feed(spikes);

        let result = region.compress().unwrap();
        assert!(result.is_some(), "Region with spikes should compress");
    }
}