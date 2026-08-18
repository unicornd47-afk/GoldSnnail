use crate::geometry::HyperbolicPoint;
use crate::audio::shd_loader::ShdSample;
use std::collections::HashMap;

fn distance(a: &HyperbolicPoint, b: &HyperbolicPoint) -> f64 {
    a.coords.iter().zip(&b.coords)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub struct HyperbolicKnn {
    pub k: usize,
}

impl HyperbolicKnn {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    pub fn classify(
        &self,
        train: &[(ShdSample, HyperbolicPoint)],
        test: &HyperbolicPoint,
    ) -> u32 {
        let mut distances: Vec<(f64, u32)> = train
            .iter()
            .map(|(sample, point)| {
                let dist = distance(point, test);
                (dist, sample.label)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.min(distances.len()).max(1);
        let mut votes = HashMap::new();

        for &(_, label) in distances.iter().take(k) {
            *votes.entry(label).or_insert(0usize) += 1;
        }

        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or(0)
    }
}
