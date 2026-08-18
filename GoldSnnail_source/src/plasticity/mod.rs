//! Plasticity — Reward-Modulated STDP & Geometric Learning Rules
//!
//! All updates are computed in-place on flat weight slices. No allocation
//! in the hot path. Distance metrics use the 1-D Poincaré disc from
//! `crate::geometry::poincare`.

pub mod r_stdp;

pub use r_stdp::RSTDP;
