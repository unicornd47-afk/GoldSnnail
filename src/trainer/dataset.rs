//! SHD event → tensor pipeline — Block 1.
//!
//! Converts a sample's raw `(time_ms, neuron_id)` spike list into a binary
//! `[T, n_in]` event tensor (row-major, `tensor[t * n_in + ch]`). Timing is
//! preserved by binning; input channels are optionally compressed (700 → n_in).

use crate::audio::shd_loader::ShdSample;

/// Convert one sample's spikes to a binary `[t_steps, n_in]` event tensor.
///
/// * `num_neurons` — raw channel count (SHD = 700).
/// * `duration_ms` — recording length (SHD ≈ 1000 ms).
/// * `t_steps` — number of temporal bins.
/// * `n_in` — output channel count (≤ `num_neurons`); channels are grouped when
///   `n_in < num_neurons` (e.g. 700 → 70).
///
/// A channel that fires ≥1 spike in a bin is set to 1.0 (binary presence).
pub fn sample_to_tensor(
    sample: &ShdSample,
    num_neurons: usize,
    duration_ms: f64,
    t_steps: usize,
    n_in: usize,
) -> Vec<f32> {
    assert!(t_steps > 0 && n_in > 0 && num_neurons > 0);
    let bin_ms = duration_ms / t_steps as f64;
    let group = (num_neurons as f64 / n_in as f64).ceil().max(1.0) as usize;
    let mut tensor = vec![0.0f32; t_steps * n_in];
    for (time, neuron) in &sample.spikes {
        let t = ((*time / bin_ms) as usize).min(t_steps - 1);
        let ch = ((*neuron as usize) / group).min(n_in - 1);
        tensor[t * n_in + ch] = 1.0;
    }
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_spikes_to_correct_timesteps() {
        let sample = ShdSample {
            spikes: vec![(10.0, 0), (990.0, 699), (500.0, 350)],
            label: 3,
        };
        let tensor = sample_to_tensor(&sample, 700, 1000.0, 10, 700);
        // 100 ms bins: 10 ms → bin 0, 500 ms → bin 5, 990 ms → bin 9.
        assert_eq!(tensor[0 * 700 + 0], 1.0);
        assert_eq!(tensor[5 * 700 + 350], 1.0);
        assert_eq!(tensor[9 * 700 + 699], 1.0);
        let nonzero = tensor.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero, 3);
    }

    #[test]
    fn compresses_channels() {
        let sample = ShdSample {
            spikes: vec![(10.0, 0), (10.0, 349), (10.0, 350), (10.0, 699)],
            label: 0,
        };
        // 700 → 70 channels (group = 10).
        let tensor = sample_to_tensor(&sample, 700, 1000.0, 10, 70);
        assert_eq!(tensor[0 * 70 + 0], 1.0); // neuron 0
        assert_eq!(tensor[0 * 70 + 34], 1.0); // neuron 349
        assert_eq!(tensor[0 * 70 + 35], 1.0); // neuron 350
        assert_eq!(tensor[0 * 70 + 69], 1.0); // neuron 699
        let nonzero = tensor.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero, 4);
    }
}
