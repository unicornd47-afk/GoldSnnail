//! DVS128 Encoder — Event-based vision to spike trains
//!
//! Converts DVS128 event camera data into spike trains compatible with
//! the GoldWorm SNN-LLM bridge. DVS events are asynchronous pixel-level
//! brightness changes: (x, y, polarity, timestamp).
//!
//! # Architecture
//!
//! 1. Events are grouped into time windows (temporal pooling)
//! 2. Each event generates a spike at a neuron determined by (x, y, polarity)
//! 3. Spike timing encodes the event timestamp within the window
//! 4. The resulting spike train can be:
//!    - Decoded via SpikeTokenDecoder (if trained)
//!    - Projected to hyperbolic space via spatial histograms
//!    - Fed into ConceptGraph for semantic mapping

use crate::substrate::{SpikeEvent, NeuronIdx};
use ndarray::array;

/// A single DVS event from the DVS128 sensor.
///
/// DVS128 specs:
/// - Resolution: 128x128 pixels
/// - Polarity: 0 = ON (brightness increase), 1 = OFF (brightness decrease)
/// - Timestamp: microseconds since sensor start
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DvsEvent {
    /// X coordinate (0-127)
    pub x: u8,
    /// Y coordinate (0-127)
    pub y: u8,
    /// Polarity: 0 = ON, 1 = OFF
    pub polarity: u8,
    /// Timestamp in microseconds
    pub timestamp_us: u32,
}

impl DvsEvent {
    /// Create a new DVS event.
    pub fn new(x: u8, y: u8, polarity: u8, timestamp_us: u32) -> Self {
        Self { x, y, polarity, timestamp_us }
    }

    /// Returns the linear neuron index for this event.
    /// Layout: neurons are organized as [polarity][y][x]
    /// Each polarity has 128*128 = 16384 neurons
    pub fn neuron_index(&self) -> usize {
        (self.polarity as usize) * 128 * 128 + (self.y as usize) * 128 + (self.x as usize)
    }
}

/// Configuration for the DVS encoder.
#[derive(Debug, Clone, Copy)]
pub struct DvsEncoderConfig {
    /// Time window size in microseconds (default: 1000 µs = 1ms)
    pub window_size_us: u32,
    /// Spike rate per event (how many spikes each event generates)
    pub spikes_per_event: u16,
    /// Maximum delay ticks for temporal encoding
    pub max_delay_ticks: u16,
    /// Whether to include polarity in neuron mapping
    pub use_polarity: bool,
}

impl Default for DvsEncoderConfig {
    fn default() -> Self {
        Self {
            window_size_us: 1000,
            spikes_per_event: 1,
            max_delay_ticks: 10,
            use_polarity: true,
        }
    }
}

/// Encodes DVS events into spike trains.
///
/// Uses temporal coding: events earlier in the window produce earlier spikes.
/// Polarity is encoded via separate neuron populations (ON vs OFF channels).
pub struct DvsEncoder {
    config: DvsEncoderConfig,
    /// Base timestamp of the current window
    window_base_us: u32,
    /// Events accumulated in the current window
    window_events: Vec<DvsEvent>,
}

impl DvsEncoder {
    /// Creates a new DVS encoder with default configuration.
    pub fn new() -> Self {
        Self::with_config(DvsEncoderConfig::default())
    }

    /// Creates a new DVS encoder with custom configuration.
    pub fn with_config(config: DvsEncoderConfig) -> Self {
        Self {
            config,
            window_base_us: 0,
            window_events: Vec::new(),
        }
    }

    /// Feeds a single event into the encoder.
    ///
    /// Events are accumulated until a time window is complete,
    /// then encoded into spikes.
    pub fn feed_event(&mut self, event: DvsEvent) {
        if event.timestamp_us >= self.window_base_us + self.config.window_size_us {
            self.window_base_us = event.timestamp_us;
        }
        self.window_events.push(event);
    }

    /// Feeds a batch of events and returns spikes for the completed windows.
    pub fn feed_batch(&mut self, events: &[DvsEvent]) -> Vec<SpikeEvent> {
        let mut all_spikes = Vec::new();
        
        for &event in events {
            self.feed_event(event);
        }
        
        // Encode any remaining events
        if !self.window_events.is_empty() {
            let events = std::mem::take(&mut self.window_events);
            all_spikes.extend(self.encode_window(&events));
        }
        
        all_spikes
    }

    /// Encodes a window of events into spike trains.
    fn encode_window(&self, events: &[DvsEvent]) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        
        if events.is_empty() {
            return spikes;
        }

        // Sort events by timestamp for temporal ordering
        let mut sorted = events.to_vec();
        sorted.sort_by_key(|e| e.timestamp_us);

        let window_start = sorted.first().map(|e| e.timestamp_us).unwrap_or(0);
        let window_end = sorted.last().map(|e| e.timestamp_us).unwrap_or(0);
        let window_duration = (window_end - window_start).max(1) as f32;

        for event in &sorted {
            let neuron_idx = if self.config.use_polarity {
                event.neuron_index()
            } else {
                (event.y as usize) * 128 + (event.x as usize)
            };

            // Temporal encoding: earlier events get earlier delays
            let t_frac = (event.timestamp_us - window_start) as f32 / window_duration;
            let delay = (t_frac * self.config.max_delay_ticks as f32) as u16;

            for _ in 0..self.config.spikes_per_event {
                spikes.push(SpikeEvent {
                    src: NeuronIdx(neuron_idx),
                    dst: NeuronIdx(neuron_idx),
                    delay_ticks: delay,
                    amplitude_u8: 255,
                    flags: 0,
                });
            }
        }

        spikes
    }

    /// Resets the encoder state (clears pending events).
    pub fn reset(&mut self) {
        self.window_events.clear();
        self.window_base_us = 0;
    }

    /// Returns the number of pending events not yet encoded.
    pub fn pending_events(&self) -> usize {
        self.window_events.len()
    }
}

/// Projects a batch of DVS events into a 2D spatial histogram.
///
/// This creates a simplified "frame" representation that can be mapped
/// into the hyperbolic lexicon space for semantic matching.
pub fn project_dvs_to_histogram(events: &[DvsEvent], bins: usize) -> Vec<f32> {
    let mut histogram = vec![0.0f32; bins * bins];
    
    if events.is_empty() || bins == 0 {
        return histogram;
    }

    for event in events {
        let x_bin = (event.x as usize * bins) / 128;
        let y_bin = (event.y as usize * bins) / 128;
        let bin_idx = y_bin * bins + x_bin;
        if bin_idx < histogram.len() {
            let value = if event.polarity == 0 { 1.0 } else { -1.0 };
            histogram[bin_idx] += value;
        }
    }

    // Normalize to [0, 1]
    let max_val = histogram.iter().map(|&v| v.abs()).reduce(f32::max).unwrap_or(1.0);
    if max_val > 0.0 {
        for v in &mut histogram {
            *v = (*v / max_val + 1.0) / 2.0; // Map [-1,1] to [0,1]
        }
    }

    histogram
}

/// Projects DVS events into a Time-Surface representation.
///
/// A Time-Surface encodes the recency of the most recent event at each pixel
/// location for each polarity. The value at each pixel is:
///   `TS(x, y, polarity) = exp(-(t_ref - t_last(x, y, polarity)) / tau)`
///
/// where `t_ref` is the most recent event timestamp and `tau` is the decay
/// constant. This captures temporal dynamics that a simple spatial histogram
/// misses — events at the leading edge of a digit trajectory get higher
/// values, creating a directional signature.
///
/// Returns a vector of length `2 * bins * bins` (ON channel followed by OFF channel).
pub fn project_dvs_to_time_surface(events: &[DvsEvent], bins: usize, tau_us: f32) -> Vec<f32> {
    let total_bins = 2 * bins * bins;
    let mut surface = vec![0.0f32; total_bins];

    if events.is_empty() || bins == 0 {
        return surface;
    }

    let t_ref = events.iter().map(|e| e.timestamp_us).max().unwrap_or(0);
    let bin_scale = bins as f32 / 128.0;

    let mut last_ts: Vec<u32> = vec![0u32; total_bins];
    let mut has_event: Vec<bool> = vec![false; total_bins];

    for event in events {
        let x_bin = (event.x as f32 * bin_scale) as usize;
        let y_bin = (event.y as f32 * bin_scale) as usize;
        let polarity = event.polarity as usize;

        if x_bin < bins && y_bin < bins && polarity < 2 {
            let idx = polarity * bins * bins + y_bin * bins + x_bin;
            if !has_event[idx] || event.timestamp_us > last_ts[idx] {
                last_ts[idx] = event.timestamp_us;
                has_event[idx] = true;
            }
        }
    }

    for i in 0..total_bins {
        if has_event[i] {
            let dt = (t_ref - last_ts[i]) as f32;
            surface[i] = (-dt / tau_us).exp();
        }
    }

    surface
}

/// Projects DVS events into combined spatial-temporal features.
///
/// Concatenates the spatial histogram (polarity-aware, normalized to [0, 1])
/// with the Time-Surface channels (ON and OFF). This gives the MLP access to
/// both *where* events occurred and *when* the most recent event happened at
/// each pixel, producing a richer representation than either encoding alone.
///
/// Returns a vector of length `3 * bins * bins`.
pub fn project_dvs_to_combined_features(events: &[DvsEvent], bins: usize, tau_us: f32) -> Vec<f32> {
    let hist = project_dvs_to_histogram(events, bins);
    let ts = project_dvs_to_time_surface(events, bins, tau_us);
    hist.into_iter().chain(ts.into_iter()).collect()
}

/// Normalizes timestamps per sample to [0, 100ms] range.
///
/// Real N-MNIST data has timestamps spanning minutes, which causes
/// the time-surface to collapse (all values near 1.0). This function
/// rescales each sample's timestamps independently to a fixed range,
/// making the time-surface parameters meaningful.
pub fn normalize_timestamps(events: &[DvsEvent]) -> Vec<DvsEvent> {
    if events.is_empty() {
        return events.to_vec();
    }

    let t_min = events.iter().map(|e| e.timestamp_us).min().unwrap_or(0);
    let t_max = events.iter().map(|e| e.timestamp_us).max().unwrap_or(0);
    let duration = (t_max - t_min).max(1) as f32;
    let target_range = 100_000.0f32; // 100ms in microseconds
    let scale = target_range / duration;

    events
        .iter()
        .map(|e| {
            let new_ts = ((e.timestamp_us - t_min) as f32 * scale) as u32;
            DvsEvent::new(e.x, e.y, e.polarity, new_ts.min(100_000))
        })
        .collect()
}

/// Projects DVS events into multi-scale spatial-temporal features.
///
/// Uses multiple tau decay constants simultaneously to capture
/// temporal dynamics at different time scales. Each tau produces
/// an ON and OFF time-surface channel, concatenated after the
/// spatial histogram. This prevents feature collapse when a single
/// tau cannot capture the full temporal range of the data.
///
/// Returns a vector of length `(1 + 2 * taus.len()) * bins * bins`.
/// For bins=16 and taus=[10ms, 50ms, 100ms]: 7 * 256 = 1792 floats.
pub fn project_dvs_to_multiscale_features(events: &[DvsEvent], bins: usize, taus: &[f32]) -> Vec<f32> {
    let total_len = (1 + 2 * taus.len()) * bins * bins;
    if events.is_empty() || bins == 0 {
        return vec![0.0f32; total_len];
    }

    let mut features = project_dvs_to_histogram(events, bins);

    for &tau in taus {
        let ts = project_dvs_to_time_surface(events, bins, tau);
        features.extend(ts);
    }

    features
}/// Maps a DVS spatial histogram to a hyperbolic point in the lexicon space.
///
/// Uses the first two principal components of the histogram as 2D coordinates,
/// then projects into the Poincaré ball.
pub fn histogram_to_hyperbolic(histogram: &[f32]) -> crate::HyperbolicPoint {
    if histogram.len() < 2 {
        return crate::HyperbolicPoint::new(array![0.0, 0.0]).unwrap();
    }

    // Simple projection: use the center of mass of positive and negative events
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut count = 0usize;
    let bins = (histogram.len() as f32).sqrt() as usize;

    for (i, &v) in histogram.iter().enumerate() {
        if v > 0.5 {
            let x = (i % bins) as f64 / bins as f64;
            let y = (i / bins) as f64 / bins as f64;
            sum_x += x;
            sum_y += y;
            count += 1;
        }
    }

    if count == 0 {
        return crate::HyperbolicPoint::new(array![0.0, 0.0]).unwrap();
    }

    let x = (sum_x / count as f64) * 2.0 - 1.0; // Map to [-1, 1]
    let y = (sum_y / count as f64) * 2.0 - 1.0;
    
    // Clamp to stay inside Poincaré ball
    let norm = (x * x + y * y).sqrt();
    let scale = if norm > 0.85 { 0.85 / norm } else { 1.0 };
    
    crate::HyperbolicPoint::new(array![x * scale, y * scale]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dvs_event_neuron_index() {
        let event = DvsEvent::new(10, 20, 1, 1000);
        // polarity=1, y=20, x=10
        let expected = 1 * 128 * 128 + 20 * 128 + 10;
        assert_eq!(event.neuron_index(), expected);
    }

    #[test]
    fn dvs_event_neuron_index_no_polarity() {
        let event = DvsEvent::new(10, 20, 0, 1000);
        let idx = event.neuron_index();
        // With polarity, it should be in the OFF channel (pol=0)
        assert_eq!(idx, 0 * 128 * 128 + 20 * 128 + 10);
    }

    #[test]
    fn dvs_encoder_basic() {
        let mut encoder = DvsEncoder::new();
        let events = vec![
            DvsEvent::new(10, 20, 0, 1000),
            DvsEvent::new(10, 20, 0, 2000),
            DvsEvent::new(10, 20, 1, 3000),
        ];
        let spikes = encoder.feed_batch(&events);
        assert_eq!(spikes.len(), 3);
    }

    #[test]
    fn dvs_encoder_window_timing() {
        let mut encoder = DvsEncoder::new();
        let events = vec![
            DvsEvent::new(10, 20, 0, 1000),
            DvsEvent::new(20, 30, 1, 2000),
        ];
        let spikes = encoder.feed_batch(&events);
        // First event should have delay 0, second should have higher delay
        assert_eq!(spikes[0].delay_ticks, 0);
        assert!(spikes[1].delay_ticks > spikes[0].delay_ticks);
    }

    #[test]
    fn project_dvs_to_histogram_basic() {
        let events = vec![
            DvsEvent::new(10, 20, 0, 1000),
            DvsEvent::new(10, 20, 1, 2000),
        ];
        let hist = project_dvs_to_histogram(&events, 4);
        assert_eq!(hist.len(), 16);
        // Values should be in [0, 1]
        for &v in &hist {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn project_dvs_to_histogram_empty() {
        let hist = project_dvs_to_histogram(&[], 4);
        assert_eq!(hist.len(), 16);
        assert!(hist.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn normalize_timestamps_preserves_relative_order() {
        let events = vec![
            DvsEvent::new(10, 20, 0, 1_000_000),
            DvsEvent::new(10, 20, 0, 2_000_000),
            DvsEvent::new(10, 20, 0, 5_000_000),
        ];
        let normalized = normalize_timestamps(&events);
        assert_eq!(normalized.len(), 3);
        assert!(normalized[0].timestamp_us <= normalized[1].timestamp_us);
        assert!(normalized[1].timestamp_us <= normalized[2].timestamp_us);
        assert!(normalized[2].timestamp_us <= 100_000);
    }

    #[test]
    fn normalize_timestamps_empty() {
        let normalized = normalize_timestamps(&[]);
        assert!(normalized.is_empty());
    }

    #[test]
    fn normalize_timestamps_scales_to_100ms() {
        let events = vec![
            DvsEvent::new(10, 20, 0, 0),
            DvsEvent::new(10, 20, 0, 10_000_000), // 10 seconds
        ];
        let normalized = normalize_timestamps(&events);
        assert_eq!(normalized[0].timestamp_us, 0);
        assert!(normalized[1].timestamp_us > 0);
        assert!(normalized[1].timestamp_us <= 100_000);
    }

    #[test]
    fn project_dvs_to_multiscale_features_length() {
        let events = vec![
            DvsEvent::new(10, 20, 0, 1000),
            DvsEvent::new(20, 30, 1, 2000),
        ];
        let taus = [10_000.0, 50_000.0, 100_000.0];
        let features = project_dvs_to_multiscale_features(&events, 4, &taus);
        let expected_len = (1 + 2 * taus.len()) * 4 * 4;
        assert_eq!(features.len(), expected_len);
    }

    #[test]
    fn project_dvs_to_multiscale_features_values_in_range() {
        let events = vec![
            DvsEvent::new(10, 20, 0, 1000),
            DvsEvent::new(20, 30, 1, 2000),
        ];
        let taus = [10_000.0, 50_000.0, 100_000.0];
        let features = project_dvs_to_multiscale_features(&events, 4, &taus);
        for &v in &features {
            assert!(v >= 0.0 && v <= 1.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn project_dvs_to_multiscale_features_empty() {
        let taus = [10_000.0, 50_000.0];
        let features = project_dvs_to_multiscale_features(&[], 4, &taus);
        let expected_len = (1 + 2 * taus.len()) * 4 * 4;
        assert_eq!(features.len(), expected_len);
        assert!(features.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn histogram_to_hyperbolic_basic() {
        let hist = vec![0.0, 1.0, 0.0, 0.0];
        let hp = histogram_to_hyperbolic(&hist);
        assert_eq!(hp.coords.len(), 2);
        assert!(hp.euclidean_norm() < 1.0);
    }
}
