//! Surrogate-gradient SNN trainer — Blocks 0–5 of the trainer sprint.
//!
//! Block 0 delivers the differentiable spiking cell (`lif`). Later blocks add
//! the dataset pipeline, the recurrent model, and the BPTT training loop.

pub mod lif;
