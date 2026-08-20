//! QLIF Neuron — Minimal single-neuron state container
//!
//! This is intentionally small and stack-friendly. For population-level
//! simulation use the `Swarm` + `StateArena` path; this struct is for
//! working-memory clusters and unit tests.

use crate::geometry::Quaternion;

/// Single QLIF neuron with persistent state.
#[derive(Debug, Clone, Copy)]
pub struct QLIFNeuron {
    pub v_m: f32,
    pub phase: f32,
    pub adapt: f32,
    pub refract: u16,
    pub quat: [f32; 4],
}

impl QLIFNeuron {
    pub fn new(_beta: f64, _threshold: f64) -> Self {
        Self {
            v_m: 0.0,
            phase: 0.0,
            adapt: 0.0,
            refract: 0,
            quat: [1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn reset(&mut self) {
        self.v_m = 0.0;
        self.phase = 0.0;
        self.adapt = 0.0;
        self.refract = 0;
        self.quat = [1.0, 0.0, 0.0, 0.0];
    }

    pub fn step(&mut self, input: &Quaternion, _dt_ms: f64, _t_ms: f64) -> Option<()> {
        if self.refract > 0 {
            self.refract = self.refract.saturating_sub(1);
            return None;
        }

        let current = input.w;
        self.v_m += (current - self.v_m - self.adapt as f32) * 0.1;
        self.v_m = self.v_m.clamp(-1.0, 1.0);

        if self.v_m >= 1.0 {
            self.v_m = 0.0;
            self.adapt += 0.1;
            self.refract = 5;
            return Some(());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strong_input_fires() {
        let mut n = QLIFNeuron::new(0.9, 1.0);
        let input = Quaternion::new(5.0, 0.0, 0.0, 0.0);
        for _ in 0..20 {
            n.step(&input, 1.0, 0.0);
        }
        assert!(n.v_m <= 1.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut n = QLIFNeuron::new(0.9, 1.0);
        let input = Quaternion::new(5.0, 0.0, 0.0, 0.0);
        n.step(&input, 1.0, 0.0);
        n.reset();
        assert_eq!(n.v_m, 0.0);
        assert_eq!(n.quat, [1.0, 0.0, 0.0, 0.0]);
    }
}
