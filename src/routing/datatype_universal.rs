//! Universal DataType Interface — SHD-CCP Extended Wire Format
//!
//! Provides a unified encode/decode layer for ALL GoldSnnail datatypes.
//! Each type gets a unique type tag in the extended SHD-CCP header:
//!
//! ```text
//! Header  : [4] magic = "SHD1"
//!          [1] version = 2
//!          [1] type_tag
//!          [2] reserved / flags
//!          [4] payload_len (u32 LE)
//! Payload : type-specific binary data
//! ```

use crate::substrate::{NeuronIdx, SpikeEvent, SpikeBuffer, StateArena, WeightMatrix};
use crate::routing::{MoAIndex, SHDCCP};
use crate::geometry::{HyperbolicPoint, PoincareBall, Quaternion};
use crate::audio::shd_loader::ShdSample;
use crate::vision::ArcGrid;
use crate::chat::dvs_encoder::{DvsEvent, DvsEncoderConfig};
use crate::swarm::SwarmConfig;
use crate::chat::ConversationTurn;
use crate::telemetry::AvalancheMetrics;
use crate::semantics::token_engine::{LexiconToken, TokenClass};

// ============================================================================
// Type Tags
// ============================================================================

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeTag {
    SpikeStream      = 0x01,
    StateArena       = 0x02,
    WeightMatrix     = 0x03,
    ShdSample        = 0x04,
    ArcGrid          = 0x05,
    HyperbolicPoint  = 0x06,
    Quaternion       = 0x07,
    DvsEventBatch    = 0x08,
    AvalancheMetrics = 0x09,
    LexiconToken     = 0x0A,
    ShdCcpSparse     = 0x0B,
    MoAIndex         = 0x0C,
    SwarmConfig      = 0x0D,
    PoincareBall     = 0x0E,
    DvsEncoderConfig = 0x0F,
    ConversationTurn = 0x10,
}

impl TryFrom<u8> for TypeTag {
    type Error = String;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(TypeTag::SpikeStream),
            0x02 => Ok(TypeTag::StateArena),
            0x03 => Ok(TypeTag::WeightMatrix),
            0x04 => Ok(TypeTag::ShdSample),
            0x05 => Ok(TypeTag::ArcGrid),
            0x06 => Ok(TypeTag::HyperbolicPoint),
            0x07 => Ok(TypeTag::Quaternion),
            0x08 => Ok(TypeTag::DvsEventBatch),
            0x09 => Ok(TypeTag::AvalancheMetrics),
            0x0A => Ok(TypeTag::LexiconToken),
            0x0B => Ok(TypeTag::ShdCcpSparse),
            0x0C => Ok(TypeTag::MoAIndex),
            0x0D => Ok(TypeTag::SwarmConfig),
            0x0E => Ok(TypeTag::PoincareBall),
            0x0F => Ok(TypeTag::DvsEncoderConfig),
            0x10 => Ok(TypeTag::ConversationTurn),
            _ => Err(format!("Unknown type tag: 0x{:02X}", value)),
        }
    }
}

// ============================================================================
// Universal DataType Enum
// ============================================================================

#[derive(Debug, Clone)]
pub enum DataType {
    SpikeStream(Vec<SpikeEvent>),
    SpikeBuffer(SpikeBuffer),
    StateArena(StateArena),
    WeightMatrix(WeightMatrix),
    ShdSample(ShdSample),
    ArcGrid(ArcGrid),
    HyperbolicPoint(HyperbolicPoint),
    Quaternion(Quaternion),
    DvsEventBatch(Vec<DvsEvent>),
    AvalancheMetrics(AvalancheMetrics),
    LexiconToken(LexiconToken),
    SHDCCP(SHDCCP),
    MoAIndex(MoAIndex),
    SwarmConfig(SwarmConfig),
    PoincareBall(PoincareBall),
    DvsEncoderConfig(DvsEncoderConfig),
    ConversationTurn(ConversationTurn),
}

// ============================================================================
// Extended SHD-CCP Header
// ============================================================================

const SHD1_MAGIC: [u8; 4] = *b"SHD1";
const SHD1_VERSION: u8 = 2;

#[derive(Debug, Clone, Copy)]
struct Shd1Header {
    type_tag: TypeTag,
    payload_len: u32,
}

impl Shd1Header {
    fn encode(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&SHD1_MAGIC);
        buf[4] = SHD1_VERSION;
        buf[5] = self.type_tag as u8;
        buf[8..12].copy_from_slice(&self.payload_len.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 12 { return Err("Buffer too short for SHD-CCP header".into()); }
        if &buf[0..4] != &SHD1_MAGIC { return Err("Invalid magic bytes (expected SHD1)".into()); }
        if buf[4] != SHD1_VERSION { return Err(format!("Unsupported version: {}", buf[4])); }
        let type_tag = TypeTag::try_from(buf[5])?;
        let payload_len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        Ok(Self { type_tag, payload_len })
    }
}

// ============================================================================
// Low-level Binary Encoders / Decoders
// ============================================================================

fn write_u8(out: &mut Vec<u8>, v: u8) { out.push(v); }
fn write_u16(out: &mut Vec<u8>, v: u16) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_u32(out: &mut Vec<u8>, v: u32) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_u64(out: &mut Vec<u8>, v: u64) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_f32(out: &mut Vec<u8>, v: f32) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_f64(out: &mut Vec<u8>, v: f64) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_bytes(out: &mut Vec<u8>, bytes: &[u8]) { out.extend_from_slice(bytes); }

fn read_u8(buf: &[u8], cursor: &mut usize) -> u8 {
    let v = buf[*cursor]; *cursor += 1; v
}
fn read_u16(buf: &[u8], cursor: &mut usize) -> u16 {
    let v = u16::from_le_bytes([buf[*cursor], buf[*cursor + 1]]); *cursor += 2; v
}
fn read_u32(buf: &[u8], cursor: &mut usize) -> u32 {
    let v = u32::from_le_bytes([buf[*cursor], buf[*cursor+1], buf[*cursor+2], buf[*cursor+3]]); *cursor += 4; v
}
fn read_u64(buf: &[u8], cursor: &mut usize) -> u64 {
    let v = u64::from_le_bytes([buf[*cursor], buf[*cursor+1], buf[*cursor+2], buf[*cursor+3], buf[*cursor+4], buf[*cursor+5], buf[*cursor+6], buf[*cursor+7]]); *cursor += 8; v
}
fn read_f32(buf: &[u8], cursor: &mut usize) -> f32 {
    let v = f32::from_le_bytes([buf[*cursor], buf[*cursor+1], buf[*cursor+2], buf[*cursor+3]]); *cursor += 4; v
}
fn read_f64(buf: &[u8], cursor: &mut usize) -> f64 {
    let v = f64::from_le_bytes([buf[*cursor], buf[*cursor+1], buf[*cursor+2], buf[*cursor+3], buf[*cursor+4], buf[*cursor+5], buf[*cursor+6], buf[*cursor+7]]); *cursor += 8; v
}
fn read_bytes(buf: &[u8], cursor: &mut usize, len: usize) -> Vec<u8> {
    let v = buf[*cursor..*cursor + len].to_vec(); *cursor += len; v
}

// ============================================================================
// Payload Encoders / Decoders per Type
// ============================================================================

fn encode_spike_stream(spikes: Vec<SpikeEvent>) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + spikes.len() * 12);
    write_u32(&mut out, spikes.len() as u32);
    let mut prev_src: u32 = 0;
    for evt in spikes {
        let src_u32 = evt.src.0 as u32;
        let dst_u32 = evt.dst.0 as u32;
        let delta_src = src_u32.wrapping_sub(prev_src) as u16;
        prev_src = src_u32;
        write_u16(&mut out, delta_src);
        write_u32(&mut out, dst_u32);
        write_u16(&mut out, evt.delay_ticks);
        write_u8(&mut out, evt.amplitude_u8);
        write_u8(&mut out, evt.flags);
        write_u16(&mut out, 0);
    }
    out
}

fn decode_spike_stream(buf: &[u8]) -> Vec<SpikeEvent> {
    let mut c = 0usize;
    let n = read_u32(buf, &mut c) as usize;
    let mut out = Vec::with_capacity(n);
    let mut prev_src: u32 = 0;
    for _ in 0..n {
        let delta_src = read_u16(buf, &mut c) as u32;
        let dst_u32 = read_u32(buf, &mut c);
        let delay = read_u16(buf, &mut c);
        let amp = read_u8(buf, &mut c);
        let flags = read_u8(buf, &mut c);
        read_u16(buf, &mut c);
        let src_u32 = prev_src.wrapping_add(delta_src);
        prev_src = src_u32;
        out.push(SpikeEvent { src: NeuronIdx(src_u32 as usize), dst: NeuronIdx(dst_u32 as usize), delay_ticks: delay, amplitude_u8: amp, flags });
    }
    out
}

fn encode_state_arena(arena: &StateArena) -> Vec<u8> {
    let mut out = Vec::new();
    let cap = arena.membrane.len() as u32;
    write_u32(&mut out, cap);
    for i in 0..arena.membrane.len() {
        write_f32(&mut out, arena.membrane[i]);
        write_f32(&mut out, arena.recovery[i]);
        write_f32(&mut out, arena.threshold[i]);
        write_u32(&mut out, arena.refractory[i]);
    }
    out
}

fn decode_state_arena(buf: &[u8]) -> StateArena {
    let mut c = 0usize;
    let cap = read_u32(buf, &mut c) as usize;
    let mut membrane = Vec::with_capacity(cap);
    let mut recovery = Vec::with_capacity(cap);
    let mut threshold = Vec::with_capacity(cap);
    let mut refractory = Vec::with_capacity(cap);
    for _ in 0..cap {
        membrane.push(read_f32(buf, &mut c));
        recovery.push(read_f32(buf, &mut c));
        threshold.push(read_f32(buf, &mut c));
        refractory.push(read_u32(buf, &mut c) as u32);
    }
    StateArena { membrane, recovery, threshold, refractory }
}

fn encode_weight_matrix(wm: &WeightMatrix) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, wm.rows as u32);
    write_u32(&mut out, wm.cols as u32);
    for &v in &wm.data { write_f32(&mut out, v); }
    out
}

fn decode_weight_matrix(buf: &[u8]) -> WeightMatrix {
    let mut c = 0usize;
    let rows = read_u32(buf, &mut c) as usize;
    let cols = read_u32(buf, &mut c) as usize;
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols { data.push(read_f32(buf, &mut c)); }
    WeightMatrix { data, rows, cols }
}

fn encode_shd_sample(sample: &ShdSample) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, sample.spikes.len() as u32);
    for &(time, neuron) in &sample.spikes {
        write_f64(&mut out, time);
        write_u32(&mut out, neuron);
    }
    write_u32(&mut out, sample.label);
    out
}

fn decode_shd_sample(buf: &[u8]) -> ShdSample {
    let mut c = 0usize;
    let n = read_u32(buf, &mut c) as usize;
    let mut spikes = Vec::with_capacity(n);
    for _ in 0..n {
        let time = read_f64(buf, &mut c);
        let neuron = read_u32(buf, &mut c);
        spikes.push((time, neuron));
    }
    let label = read_u32(buf, &mut c);
    ShdSample { spikes, label }
}

fn encode_arc_grid(grid: &ArcGrid) -> Vec<u8> {
    let mut out = Vec::new();
    write_u16(&mut out, grid.width as u16);
    write_u16(&mut out, grid.height as u16);
    for row in &grid.data {
        for &v in row { write_u8(&mut out, v); }
    }
    out
}

fn decode_arc_grid(buf: &[u8]) -> ArcGrid {
    let mut c = 0usize;
    let width = read_u16(buf, &mut c) as usize;
    let height = read_u16(buf, &mut c) as usize;
    let mut data = Vec::with_capacity(height);
    for _ in 0..height {
        let mut row = Vec::with_capacity(width);
        for _ in 0..width { row.push(read_u8(buf, &mut c)); }
        data.push(row);
    }
    ArcGrid { data, width, height }
}

fn encode_hyperbolic_point(p: &HyperbolicPoint) -> Vec<u8> {
    let mut out = Vec::new();
    write_u16(&mut out, p.coords.len() as u16);
    for &v in &p.coords { write_f64(&mut out, v); }
    out
}

fn decode_hyperbolic_point(buf: &[u8]) -> HyperbolicPoint {
    let mut c = 0usize;
    let dim = read_u16(buf, &mut c) as usize;
    let mut coords = Vec::with_capacity(dim);
    for _ in 0..dim { coords.push(read_f64(buf, &mut c)); }
    HyperbolicPoint { coords }
}

fn encode_quaternion(q: &Quaternion) -> Vec<u8> {
    let mut out = Vec::new();
    write_f32(&mut out, q.w);
    write_f32(&mut out, q.x);
    write_f32(&mut out, q.y);
    write_f32(&mut out, q.z);
    out
}

fn decode_quaternion(buf: &[u8]) -> Quaternion {
    let mut c = 0usize;
    let w = read_f32(buf, &mut c);
    let x = read_f32(buf, &mut c);
    let y = read_f32(buf, &mut c);
    let z = read_f32(buf, &mut c);
    Quaternion { w, x, y, z }
}

fn encode_dvs_batch(events: &[DvsEvent]) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, events.len() as u32);
    for e in events {
        write_u8(&mut out, e.x);
        write_u8(&mut out, e.y);
        write_u8(&mut out, e.polarity);
        write_u32(&mut out, e.timestamp_us);
    }
    out
}

fn decode_dvs_batch(buf: &[u8]) -> Vec<DvsEvent> {
    let mut c = 0usize;
    let n = read_u32(buf, &mut c) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let x = read_u8(buf, &mut c);
        let y = read_u8(buf, &mut c);
        let polarity = read_u8(buf, &mut c);
        let timestamp_us = read_u32(buf, &mut c);
        out.push(DvsEvent { x, y, polarity, timestamp_us });
    }
    out
}

fn encode_avalanche_metrics(m: &AvalancheMetrics) -> Vec<u8> {
    let mut out = Vec::new();
    write_u64(&mut out, m.total_spikes);
    write_f32(&mut out, m.mean_activity);
    write_f32(&mut out, m.criticality_index);
    write_f32(&mut out, m.entropy);
    out
}

fn decode_avalanche_metrics(buf: &[u8]) -> AvalancheMetrics {
    let mut c = 0usize;
    let total_spikes = read_u64(buf, &mut c);
    let mean_activity = read_f32(buf, &mut c);
    let criticality_index = read_f32(buf, &mut c);
    let entropy = read_f32(buf, &mut c);
    AvalancheMetrics { total_spikes, mean_activity, criticality_index, entropy }
}

fn encode_lexicon_token(tok: &LexiconToken) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, tok.id as u32);
    let surface_bytes = tok.surface.as_bytes();
    write_u16(&mut out, surface_bytes.len() as u16);
    write_bytes(&mut out, surface_bytes);
    write_u8(&mut out, token_class_to_u8(&tok.class));
    write_f32(&mut out, tok.embedding.w);
    write_f32(&mut out, tok.embedding.x);
    write_f32(&mut out, tok.embedding.y);
    write_f32(&mut out, tok.embedding.z);
    write_u16(&mut out, tok.hyperbolic.coords.len() as u16);
    for &v in &tok.hyperbolic.coords { write_f64(&mut out, v); }
    write_f64(&mut out, tok.salience);
    out
}

fn decode_lexicon_token(buf: &[u8]) -> LexiconToken {
    let mut c = 0usize;
    let id = read_u32(buf, &mut c) as usize;
    let surf_len = read_u16(buf, &mut c) as usize;
    let surface = String::from_utf8(read_bytes(buf, &mut c, surf_len)).unwrap_or_default();
    let class = u8_to_token_class(read_u8(buf, &mut c));
    let w = read_f32(buf, &mut c);
    let x = read_f32(buf, &mut c);
    let y = read_f32(buf, &mut c);
    let z = read_f32(buf, &mut c);
    let embedding = Quaternion { w, x, y, z };
    let h_dim = read_u16(buf, &mut c) as usize;
    let mut coords = Vec::with_capacity(h_dim);
    for _ in 0..h_dim { coords.push(read_f64(buf, &mut c)); }
    let hyperbolic = HyperbolicPoint { coords };
    let salience = read_f64(buf, &mut c);
    LexiconToken { id, surface, class, embedding, hyperbolic, salience }
}

fn encode_shd_ccp(ccp: &SHDCCP) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, ccp.values.len() as u32);
    for &v in &ccp.values { write_f32(&mut out, v); }
    for &idx in &ccp.col_indices { write_u32(&mut out, idx); }
    write_u32(&mut out, ccp.row_ptr.len() as u32);
    for &rp in &ccp.row_ptr { write_u64(&mut out, rp as u64); }
    out
}

fn decode_shd_ccp(buf: &[u8]) -> SHDCCP {
    let mut c = 0usize;
    let n = read_u32(buf, &mut c) as usize;
    let mut values = Vec::with_capacity(n);
    for _ in 0..n { values.push(read_f32(buf, &mut c)); }
    let mut col_indices = Vec::with_capacity(n);
    for _ in 0..n { col_indices.push(read_u32(buf, &mut c)); }
    let rp_len = read_u32(buf, &mut c) as usize;
    let mut row_ptr = Vec::with_capacity(rp_len);
    for _ in 0..rp_len { row_ptr.push(read_u64(buf, &mut c) as usize); }
    SHDCCP { values, col_indices, row_ptr }
}

fn encode_moa_index(idx: &MoAIndex) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, idx.expert_indices.len() as u32);
    for &ei in &idx.expert_indices { write_u32(&mut out, ei); }
    for &s in &idx.scores { write_f32(&mut out, s); }
    out
}

fn decode_moa_index(buf: &[u8]) -> MoAIndex {
    let mut c = 0usize;
    let len = read_u32(buf, &mut c) as usize;
    let mut expert_indices = Vec::with_capacity(len);
    let mut scores = Vec::with_capacity(len);
    for _ in 0..len { expert_indices.push(read_u32(buf, &mut c)); }
    for _ in 0..len { scores.push(read_f32(buf, &mut c)); }
    MoAIndex { expert_indices, scores }
}

fn encode_swarm_config(cfg: &SwarmConfig) -> Vec<u8> {
    let mut out = Vec::new();
    write_f32(&mut out, cfg.decay);
    write_f32(&mut out, cfg.resting_potential);
    write_f32(&mut out, cfg.noise_std);
    out
}

fn decode_swarm_config(buf: &[u8]) -> SwarmConfig {
    let mut c = 0usize;
    let decay = read_f32(buf, &mut c);
    let resting_potential = read_f32(buf, &mut c);
    let noise_std = read_f32(buf, &mut c);
    SwarmConfig { decay, resting_potential, noise_std }
}

fn encode_poincare_ball(ball: &PoincareBall) -> Vec<u8> {
    let mut out = Vec::new();
    write_f64(&mut out, ball.curvature);
    out
}

fn decode_poincare_ball(buf: &[u8]) -> PoincareBall {
    let mut c = 0usize;
    let curvature = read_f64(buf, &mut c);
    PoincareBall { curvature }
}

fn encode_dvs_encoder_config(cfg: &DvsEncoderConfig) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, cfg.window_size_us);
    write_u16(&mut out, cfg.spikes_per_event);
    write_u16(&mut out, cfg.max_delay_ticks);
    write_u8(&mut out, cfg.use_polarity as u8);
    out
}

fn decode_dvs_encoder_config(buf: &[u8]) -> DvsEncoderConfig {
    let mut c = 0usize;
    let window_size_us = read_u32(buf, &mut c);
    let spikes_per_event = read_u16(buf, &mut c);
    let max_delay_ticks = read_u16(buf, &mut c);
    let use_polarity = read_u8(buf, &mut c) != 0;
    DvsEncoderConfig { window_size_us, spikes_per_event, max_delay_ticks, use_polarity }
}

fn encode_conversation_turn(turn: &ConversationTurn) -> Vec<u8> {
    let mut out = Vec::new();
    let role_bytes = turn.role.as_bytes();
    write_u16(&mut out, role_bytes.len() as u16);
    write_bytes(&mut out, role_bytes);
    let content_bytes = turn.text.as_bytes();
    write_u32(&mut out, content_bytes.len() as u32);
    write_bytes(&mut out, content_bytes);
    write_i64(&mut out, turn.timestamp as i64);
    out
}

fn decode_conversation_turn(buf: &[u8]) -> ConversationTurn {
    let mut c = 0usize;
    let role_len = read_u16(buf, &mut c) as usize;
    let role = String::from_utf8(read_bytes(buf, &mut c, role_len)).unwrap_or_default();
    let content_len = read_u32(buf, &mut c) as usize;
    let text = String::from_utf8(read_bytes(buf, &mut c, content_len)).unwrap_or_default();
    let timestamp = read_i64(buf, &mut c) as u64;
    ConversationTurn { role, text, timestamp, tokens: Vec::new(), reward: None }
}

fn write_i64(out: &mut Vec<u8>, v: i64) { out.extend_from_slice(&v.to_le_bytes()); }
fn read_i64(buf: &[u8], cursor: &mut usize) -> i64 {
    let v = i64::from_le_bytes([buf[*cursor], buf[*cursor+1], buf[*cursor+2], buf[*cursor+3], buf[*cursor+4], buf[*cursor+5], buf[*cursor+6], buf[*cursor+7]]);
    *cursor += 8;
    v
}

// ============================================================================
// TokenClass conversion helpers
// ============================================================================

fn token_class_to_u8(class: &TokenClass) -> u8 {
    match class {
        TokenClass::Determiner => 0,
        TokenClass::NounConcrete => 1,
        TokenClass::NounAbstract => 2,
        TokenClass::VerbAction => 3,
        TokenClass::VerbState => 4,
        TokenClass::Adjective => 5,
        TokenClass::Preposition => 6,
        TokenClass::SemanticRole => 7,
        TokenClass::GrammarMarker => 8,
        TokenClass::Punctuation => 9,
        TokenClass::Noise => 10,
    }
}

fn u8_to_token_class(v: u8) -> TokenClass {
    match v {
        0 => TokenClass::Determiner,
        1 => TokenClass::NounConcrete,
        2 => TokenClass::NounAbstract,
        3 => TokenClass::VerbAction,
        4 => TokenClass::VerbState,
        5 => TokenClass::Adjective,
        6 => TokenClass::Preposition,
        7 => TokenClass::SemanticRole,
        8 => TokenClass::GrammarMarker,
        9 => TokenClass::Punctuation,
        _ => TokenClass::Noise,
    }
}

// ============================================================================
// Payload dispatch
// ============================================================================

fn encode_payload(tag: TypeTag, data: &DataType) -> Vec<u8> {
    match (tag, data) {
        (TypeTag::SpikeStream, DataType::SpikeStream(s)) => encode_spike_stream(s.clone()),
        (TypeTag::SpikeStream, DataType::SpikeBuffer(buf)) => encode_spike_stream(
            { buf.indices.iter().map(|&i| SpikeEvent { src: NeuronIdx(i as usize), dst: NeuronIdx(i as usize), delay_ticks: 0, amplitude_u8: 255, flags: 0 }).collect() }
        ),
        (TypeTag::StateArena, DataType::StateArena(a)) => encode_state_arena(a),
        (TypeTag::WeightMatrix, DataType::WeightMatrix(w)) => encode_weight_matrix(w),
        (TypeTag::ShdSample, DataType::ShdSample(s)) => encode_shd_sample(s),
        (TypeTag::ArcGrid, DataType::ArcGrid(g)) => encode_arc_grid(g),
        (TypeTag::HyperbolicPoint, DataType::HyperbolicPoint(p)) => encode_hyperbolic_point(p),
        (TypeTag::Quaternion, DataType::Quaternion(q)) => encode_quaternion(q),
        (TypeTag::DvsEventBatch, DataType::DvsEventBatch(evts)) => encode_dvs_batch(evts),
        (TypeTag::AvalancheMetrics, DataType::AvalancheMetrics(m)) => encode_avalanche_metrics(m),
        (TypeTag::LexiconToken, DataType::LexiconToken(t)) => encode_lexicon_token(t),
        (TypeTag::ShdCcpSparse, DataType::SHDCCP(s)) => encode_shd_ccp(s),
        (TypeTag::MoAIndex, DataType::MoAIndex(m)) => encode_moa_index(m),
        (TypeTag::SwarmConfig, DataType::SwarmConfig(c)) => encode_swarm_config(c),
        (TypeTag::PoincareBall, DataType::PoincareBall(p)) => encode_poincare_ball(p),
        (TypeTag::DvsEncoderConfig, DataType::DvsEncoderConfig(c)) => encode_dvs_encoder_config(c),
        (TypeTag::ConversationTurn, DataType::ConversationTurn(t)) => encode_conversation_turn(&t),
        _ => Vec::new(),
    }
}

fn decode_payload(tag: TypeTag, buf: &[u8]) -> Result<DataType, String> {
    Ok(match tag {
        TypeTag::SpikeStream => DataType::SpikeStream(decode_spike_stream(buf)),
        TypeTag::StateArena => DataType::StateArena(decode_state_arena(buf)),
        TypeTag::WeightMatrix => DataType::WeightMatrix(decode_weight_matrix(buf)),
        TypeTag::ShdSample => DataType::ShdSample(decode_shd_sample(buf)),
        TypeTag::ArcGrid => DataType::ArcGrid(decode_arc_grid(buf)),
        TypeTag::HyperbolicPoint => DataType::HyperbolicPoint(decode_hyperbolic_point(buf)),
        TypeTag::Quaternion => DataType::Quaternion(decode_quaternion(buf)),
        TypeTag::DvsEventBatch => DataType::DvsEventBatch(decode_dvs_batch(buf)),
        TypeTag::AvalancheMetrics => DataType::AvalancheMetrics(decode_avalanche_metrics(buf)),
        TypeTag::LexiconToken => DataType::LexiconToken(decode_lexicon_token(buf)),
        TypeTag::ShdCcpSparse => DataType::SHDCCP(decode_shd_ccp(buf)),
        TypeTag::MoAIndex => DataType::MoAIndex(decode_moa_index(buf)),
        TypeTag::SwarmConfig => DataType::SwarmConfig(decode_swarm_config(buf)),
        TypeTag::PoincareBall => DataType::PoincareBall(decode_poincare_ball(buf)),
        TypeTag::DvsEncoderConfig => DataType::DvsEncoderConfig(decode_dvs_encoder_config(buf)),
        TypeTag::ConversationTurn => DataType::ConversationTurn(decode_conversation_turn(buf)),
    })
}

// ============================================================================
// Public API
// ============================================================================

pub fn encode_datatype(data: &DataType) -> Vec<u8> {
    let tag = data_type_tag(data);
    let payload = encode_payload(tag, data);
    let header = Shd1Header { type_tag: tag, payload_len: payload.len() as u32 };
    let mut out = Vec::with_capacity(12 + payload.len());
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(&payload);
    out
}

pub fn decode_datatype(buf: &[u8]) -> Result<DataType, String> {
    if buf.len() < 12 { return Err("Buffer too short for SHD-CCP header".into()); }
    let header = Shd1Header::decode(buf)?;
    let payload = &buf[12..];
    if payload.len() < header.payload_len as usize { return Err("Payload truncated".into()); }
    decode_payload(header.type_tag, &payload[..header.payload_len as usize])
}

pub fn data_type_tag(data: &DataType) -> TypeTag {
    match data {
        DataType::SpikeStream(_) | DataType::SpikeBuffer(_) => TypeTag::SpikeStream,
        DataType::StateArena(_) => TypeTag::StateArena,
        DataType::WeightMatrix(_) => TypeTag::WeightMatrix,
        DataType::ShdSample(_) => TypeTag::ShdSample,
        DataType::ArcGrid(_) => TypeTag::ArcGrid,
        DataType::HyperbolicPoint(_) => TypeTag::HyperbolicPoint,
        DataType::Quaternion(_) => TypeTag::Quaternion,
        DataType::DvsEventBatch(_) => TypeTag::DvsEventBatch,
        DataType::AvalancheMetrics(_) => TypeTag::AvalancheMetrics,
        DataType::LexiconToken(_) => TypeTag::LexiconToken,
        DataType::SHDCCP(_) => TypeTag::ShdCcpSparse,
        DataType::MoAIndex(_) => TypeTag::MoAIndex,
        DataType::SwarmConfig(_) => TypeTag::SwarmConfig,
        DataType::PoincareBall(_) => TypeTag::PoincareBall,
        DataType::DvsEncoderConfig(_) => TypeTag::DvsEncoderConfig,
        DataType::ConversationTurn(_) => TypeTag::ConversationTurn,
    }
}

pub fn type_tag_name(tag: TypeTag) -> &'static str {
    match tag {
        TypeTag::SpikeStream => "SpikeStream",
        TypeTag::StateArena => "StateArena",
        TypeTag::WeightMatrix => "WeightMatrix",
        TypeTag::ShdSample => "ShdSample",
        TypeTag::ArcGrid => "ArcGrid",
        TypeTag::HyperbolicPoint => "HyperbolicPoint",
        TypeTag::Quaternion => "Quaternion",
        TypeTag::DvsEventBatch => "DvsEventBatch",
        TypeTag::AvalancheMetrics => "AvalancheMetrics",
        TypeTag::LexiconToken => "LexiconToken",
        TypeTag::ShdCcpSparse => "SHDCCP",
        TypeTag::MoAIndex => "MoAIndex",
        TypeTag::SwarmConfig => "SwarmConfig",
        TypeTag::PoincareBall => "PoincareBall",
        TypeTag::DvsEncoderConfig => "DvsEncoderConfig",
        TypeTag::ConversationTurn => "ConversationTurn",
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Quaternion;
    use crate::semantics::token_engine::TokenClass;

    #[test]
    fn spike_stream_roundtrip() {
        let spikes = vec![
            SpikeEvent { src: NeuronIdx(0), dst: NeuronIdx(10), delay_ticks: 3, amplitude_u8: 200, flags: 0 },
            SpikeEvent { src: NeuronIdx(5), dst: NeuronIdx(20), delay_ticks: 1, amplitude_u8: 100, flags: 1 },
        ];
        let dt = DataType::SpikeStream(spikes.clone());
        let encoded = encode_datatype(&dt);
        let decoded = decode_datatype(&encoded).unwrap();
        match decoded {
            DataType::SpikeStream(s) => assert_eq!(s.len(), spikes.len()),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn hyperbolic_point_roundtrip() {
        let hp = HyperbolicPoint { coords: vec![0.1, -0.2, 0.3] };
        let dt = DataType::HyperbolicPoint(hp.clone());
        let encoded = encode_datatype(&dt);
        let decoded = decode_datatype(&encoded).unwrap();
        match decoded {
            DataType::HyperbolicPoint(p) => assert_eq!(p.coords, hp.coords),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn quaternion_roundtrip() {
        let q = Quaternion { w: 1.0, x: 0.5, y: -0.3, z: 0.8 };
        let dt = DataType::Quaternion(q);
        let encoded = encode_datatype(&dt);
        let decoded = decode_datatype(&encoded).unwrap();
        match decoded {
            DataType::Quaternion(p) => {
                assert!((p.w - q.w).abs() < 1e-6);
                assert!((p.x - q.x).abs() < 1e-6);
                assert!((p.y - q.y).abs() < 1e-6);
                assert!((p.z - q.z).abs() < 1e-6);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn arc_grid_roundtrip() {
        let grid = ArcGrid { data: vec![vec![0,1,2], vec![3,4,5]], width: 3, height: 2 };
        let dt = DataType::ArcGrid(grid.clone());
        let encoded = encode_datatype(&dt);
        let decoded = decode_datatype(&encoded).unwrap();
        match decoded {
            DataType::ArcGrid(g) => {
                assert_eq!(g.width, grid.width);
                assert_eq!(g.height, grid.height);
                assert_eq!(g.data, grid.data);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn lexicon_token_roundtrip() {
        let tok = LexiconToken {
            id: 42,
            surface: "hello".to_string(),
            class: TokenClass::NounConcrete,
            embedding: Quaternion { w: 0.9, x: 0.1, y: 0.2, z: 0.3 },
            hyperbolic: HyperbolicPoint { coords: vec![0.5, 0.5] },
            salience: 0.8,
        };
        let dt = DataType::LexiconToken(tok.clone());
        let encoded = encode_datatype(&dt);
        let decoded = decode_datatype(&encoded).unwrap();
        match decoded {
            DataType::LexiconToken(t) => {
                assert_eq!(t.id, tok.id);
                assert_eq!(t.surface, tok.surface);
                assert_eq!(t.class, tok.class);
                assert_eq!(t.salience, tok.salience);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn weight_matrix_roundtrip() {
        let wm = WeightMatrix { data: vec![1.0, 2.0, 3.0, 4.0], rows: 2, cols: 2 };
        let dt = DataType::WeightMatrix(wm.clone());
        let encoded = encode_datatype(&dt);
        let decoded = decode_datatype(&encoded).unwrap();
        match decoded {
            DataType::WeightMatrix(w) => {
                assert_eq!(w.rows, wm.rows);
                assert_eq!(w.cols, wm.cols);
                assert_eq!(w.data, wm.data);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn decode_invalid_magic_returns_err() {
        let buf = [0u8; 4];
        let res = decode_datatype(&buf);
        assert!(res.is_err());
    }

    #[test]
    fn decode_truncated_header_returns_err() {
        let buf = [0u8; 10];
        let res = decode_datatype(&buf);
        assert!(res.is_err());
    }

    #[test]
    fn empty_spike_stream_roundtrip() {
        let dt = DataType::SpikeStream(vec![]);
        let encoded = encode_datatype(&dt);
        let decoded = decode_datatype(&encoded).unwrap();
        match decoded {
            DataType::SpikeStream(s) => assert!(s.is_empty()),
            _ => panic!("Wrong variant"),
        }
    }
}








