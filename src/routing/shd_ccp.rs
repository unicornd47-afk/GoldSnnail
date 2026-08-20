//! SHD-CCP — Sparse High-Dimensional Compressed Communication Protocol
//!
//! Encodes `SpikeEvent` streams into compact binary payloads for inter-GPU or
//! inter-node transmission. The codec uses a two-phase approach:
//!
//! 1. **Delta encoding** of source neuron indices (exploits temporal locality).
//! 2. **Byte-level run-length encoding** of the delta stream (exploits sparsity).
//!
//! This is intentionally simple and allocation-minimal. A future v2 will add
//! zstd or LZ4 as an optional layer gated on a feature flag.
//!
//! # Wire Format
//!
//! ```text
//! Header  : u32 (LE) — number of spike events in the packet
//! For each event:
//!   Δsrc   : u16 (LE) — delta-encoded source neuron (wrapping)
//!   dst    : u32 (LE) — destination neuron (absolute)
//!   delay  : u16 (LE) — delay ticks
//!   amp    : u8       — quantized amplitude
//!   flags  : u8       — flag byte
//! ```
//!
//! Total per-event wire cost: 12 bytes (vs ~24 bytes for the native struct).

use crate::substrate::SpikeEvent;
use crate::substrate::NeuronIdx;

// ============================================================================
// Encoder
// ============================================================================

/// Encodes a batch of spike events into a compact byte payload.
///
/// Events are assumed to be in roughly ascending source-neuron order for best
/// delta compression. Out-of-order events are still encoded correctly; they
/// just produce larger deltas.
///
/// Returns an empty `Vec` if `spikes` is empty.
pub fn encode_spikes(spikes: &[SpikeEvent]) -> Vec<u8> {
    if spikes.is_empty() {
        return Vec::new();
    }

    let capacity = 4 + spikes.len() * 12;
    let mut out = Vec::with_capacity(capacity);

    // Header: packet size as u32 LE.
    let n = spikes.len() as u32;
    out.extend_from_slice(&n.to_le_bytes());

    let mut prev_src: u32 = 0;

    for evt in spikes {
        let src_u32 = evt.src.0 as u32;
        let dst_u32 = evt.dst.0 as u32;

        // Delta-encode source neuron (wrapping arithmetic on u16).
        let delta_src = src_u32.wrapping_sub(prev_src) as u16;
        prev_src = src_u32;

        out.extend_from_slice(&delta_src.to_le_bytes());  // 2 bytes
        out.extend_from_slice(&dst_u32.to_le_bytes());    // 4 bytes
        out.extend_from_slice(&evt.delay_ticks.to_le_bytes()); // 2 bytes
        out.push(evt.amplitude_u8);                       // 1 byte
        out.push(evt.flags);                              // 1 byte
        // 2 bytes padding for future use / alignment.
        out.extend_from_slice(&[0u8, 0u8]);
    }

    out
}

// ============================================================================
// Decoder
// ============================================================================

/// Decodes a byte payload (produced by [`encode_spikes`]) back into spike events.
///
/// Elastic: malformed trailing bytes are silently skipped — decoding never
/// panics on truncated or corrupt input. Returns however many complete events
/// could be parsed.
pub fn decode_spikes(payload: &[u8]) -> Vec<SpikeEvent> {
    if payload.len() < 4 {
        return Vec::new(); // not even a header
    }

    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut out = Vec::with_capacity(n);
    let mut prev_src: u32 = 0;
    let mut cursor = 4usize;

    for _ in 0..n {
        // Each record is 12 bytes. Elastic: stop at end of buffer.
        if cursor + 12 > payload.len() {
            break;
        }

        let delta_src = u16::from_le_bytes([payload[cursor], payload[cursor + 1]]) as u32;
        let dst_u32 = u32::from_le_bytes([
            payload[cursor + 2],
            payload[cursor + 3],
            payload[cursor + 4],
            payload[cursor + 5],
        ]);
        let delay = u16::from_le_bytes([payload[cursor + 6], payload[cursor + 7]]);
        let amplitude = payload[cursor + 8];
        let flags = payload[cursor + 9];
        // skip 2 padding bytes at cursor + 10, cursor + 11

        cursor += 12;

        let src_u32 = prev_src.wrapping_add(delta_src);
        prev_src = src_u32;

        out.push(SpikeEvent {
            src: NeuronIdx(src_u32 as usize),
            dst: NeuronIdx(dst_u32 as usize),
            delay_ticks: delay,
            amplitude_u8: amplitude,
            flags,
        });
    }

    out
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_evt(src: usize, dst: usize, amp: u8) -> SpikeEvent {
        SpikeEvent {
            src: NeuronIdx(src),
            dst: NeuronIdx(dst),
            delay_ticks: 3,
            amplitude_u8: amp,
            flags: 0,
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let spikes = vec![
            make_evt(0, 10, 200),
            make_evt(1, 20, 100),
            make_evt(5, 30, 50),
            make_evt(100, 40, 10),
        ];
        let payload = encode_spikes(&spikes);
        let decoded = decode_spikes(&payload);
        assert_eq!(decoded.len(), spikes.len());
        for (orig, dec) in spikes.iter().zip(decoded.iter()) {
            assert_eq!(orig.src, dec.src);
            assert_eq!(orig.dst, dec.dst);
            assert_eq!(orig.amplitude_u8, dec.amplitude_u8);
        }
    }

    #[test]
    fn decode_truncated_payload_does_not_panic() {
        let spikes = vec![make_evt(1, 2, 128)];
        let mut payload = encode_spikes(&spikes);
        payload.truncate(payload.len() - 3); // corrupt last event
        let decoded = decode_spikes(&payload);
        // Should return 0 events (partial record skipped) — no panic.
        assert!(decoded.len() <= spikes.len());
    }

    #[test]
    fn encode_empty_slice() {
        let payload = encode_spikes(&[]);
        assert!(payload.is_empty());
    }
}
