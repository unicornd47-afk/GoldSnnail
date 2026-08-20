//! Simple Agent Demo
//!
//! Runs a single timestep through the full GoldWorm AGI pipeline:
//!
//! Sensor → Attention → Working Memory → Compression → World Model → RL Agent → R-STDP
//!
//! Usage:
//!   cargo run --example simple_agent

use goldworm::{
    GeometricBottleneck, HyperbolicPoint,
    Quaternion, QuaternionAttention, RLAgent, RSTDP,
    SpikeBuffer, StateVector, Transition, WorldModel, WorkingMemory,
};
use ndarray::array;

fn main() {
    println!("=== GoldWorm Simple Agent Demo ===\n");

    // 1. Sensor input (self-attention)
    let sensor = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0)];
    let attn = QuaternionAttention::new();
    let attended = attn.forward(&sensor, &sensor, &sensor);
    println!("[1] Attention: {} quaternion(s) attended", attended.len());

    // 2. Working Memory
    let mut mem = WorkingMemory::new(8, 0.9, 1.0);
    mem.set_input_weight(0, 1.0);
    let spikes = mem.step(&attended, 1.0, 0.0);
    let spike_count = spikes.iter().filter(|&&fired| fired).count();
    println!("[2] Working Memory: {} / 8 neurons fired", spike_count);

    // 3. Compression Bottleneck
    let mut bn = GeometricBottleneck::new(8, 3, 0.05, 1.0);
    let phases = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); 8];
    let spike_buf = SpikeBuffer::new(8);
    let compressed = bn.compress(&spike_buf, &phases).unwrap();
    match compressed {
        Some(ref p) => println!("[3] Compression: latent point (norm={:.4})", p.euclidean_norm()),
        None => println!("[3] Compression: suppressed (delta below threshold)"),
    }

    // 4. World Model
    let mut wm = WorldModel::new(3, 6, 1.0);
    let latent = compressed.unwrap_or_else(|| HyperbolicPoint::new(array![0.0, 0.0]).unwrap());
    wm.observe(latent.clone());
    let predicted = wm.predict(&latent).unwrap();
    println!("[4] World Model: predicted latent (norm={:.4})", predicted.euclidean_norm());

    // 5. RL Agent
    let state = StateVector::new(latent.clone(), &spikes);
    let next_state = StateVector::new(predicted, &spikes);
    let mut agent = RLAgent::new(state.dim(), 0.9);
    let action = agent.act(&state);
    println!("[5] RL Agent: action quaternion = ({:.2}, {:.2}, {:.2}, {:.2})", action.w, action.x, action.y, action.z);

    // 6. R-STDP Learning
    let stdp = RSTDP::new(0.01, 20.0, 1.0);
    let pre = HyperbolicPoint::new(array![0.1, 0.0, 0.0]).unwrap();
    let post = HyperbolicPoint::new(array![0.11, 0.01, 0.0]).unwrap();

    let transition = Transition {
        state,
        action,
        reward: 1.0,
        next_state,
    };

    let delta = agent
        .train_step(&transition, &stdp, &pre, &post, 0.0, 5.0, 0.1, 0.1)
        .unwrap();

    println!("[6] R-STDP: TD-error = {:.6}", delta);
    println!("\n=== Pipeline complete ===");
}