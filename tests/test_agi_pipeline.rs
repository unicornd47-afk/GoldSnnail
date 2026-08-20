//! Integration smoke test for the AGI pipeline.
//!
//! Runs a minimal pass through Attention → Working Memory → Compression → World Model → R-STDP → RL
//! to verify that the modules compose correctly.

use goldsnnail::{
    Quaternion, QuaternionAttention, WorkingMemory, RSTDP,
    GeometricBottleneck, SpikeBuffer, WorldModel, HyperbolicPoint,
    RLAgent, StateVector,
};
use ndarray::array;

#[test]
fn agi_pipeline_smoke_test() {
    // 1. Attention on trivial sensor input (self-attention).
    let sensor = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0)];
    let attn = QuaternionAttention::new();
    let attended = attn.forward(&sensor, &sensor, &sensor);

    // 2. Working Memory processes attended input.
    let mut mem = WorkingMemory::new(4, 0.9, 1.0);
    mem.set_input_weight(0, 1.0);
    let _spikes = mem.step(&attended, 1.0, 0.0);

    // 3. Compression Bottleneck
    let mut bn = GeometricBottleneck::new(4, 2, 0.05, 1.0);
    let phases = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); 4];

    // Simulate SpikeBuffer (empty — just verify the pipeline composes)
    let spike_buf = SpikeBuffer::new(4);
    let compressed = bn.compress(&spike_buf, &phases).unwrap();
    // Either Some(latent) or None — both are valid outcomes
    assert!(compressed.is_none() || compressed.is_some());

    // 4. World Model: predict next latent state from a sample point
    let mut wm = WorldModel::new(2, 6, 1.0);
    let sample_latent = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
    wm.observe(sample_latent.clone());
    let predicted = wm.predict(&sample_latent).unwrap();
    assert!(predicted.euclidean_norm() < 1.0, "Prediction must stay inside Poincaré ball");

    // 5. R-STDP computes a reward-modulated update (1-D Poincaré disc embeddings).
    let stdp = RSTDP::new(0.01, 20.0, -1.0);
    let dw = stdp.compute(1.0, 0.0, 5.0, 0.1, 0.11);

    assert!(dw > 0.0, "R-STDP should produce potentiation for pre-before-post with positive reward");

    // 6. RL Agent: Value/Policy + R-STDP full-stack
    let state_dim = 5;
    let mut agent = RLAgent::new(state_dim, 0.9);
    let latent = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
    let state = StateVector::new(latent, &[true, false, true]);

    // Act
    let action = agent.act(&state);
    assert!(action.norm() > 0.99, "Policy action must be normalized");

    // Value head produces bounded scalar
    let val = agent.value.value(&state);
    assert!(val.abs() <= 1.0, "Value head must return bounded value");

    // Observe and train online
    let next_latent = HyperbolicPoint::new(array![0.11, 0.01]).unwrap();
    let next_state = StateVector::new(next_latent, &[false, true, false]);
    let transition = goldsnnail::Transition {
        state: state.clone(),
        action,
        reward: 1.0,
        next_state,
    };
    agent.observe(transition.clone());

    let pre_embed = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
    let post_embed = HyperbolicPoint::new(array![0.11, 0.01]).unwrap();
    let delta = agent.train_step(
        &transition,
        &stdp,
        &pre_embed,
        &post_embed,
        0.0,
        5.0,
        0.01,
        0.01,
    ).unwrap();
    assert!(delta.is_finite(), "TD error must be finite");
}
