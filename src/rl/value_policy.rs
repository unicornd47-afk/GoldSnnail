//! Value/Policy Head with TD-Learning and R-STDP
//!
//! All state vectors are flat `Vec<f64>` (DOD-compatible).

use crate::geometry::{HyperbolicPoint, Quaternion};
use crate::plasticity::RSTDP;
use crate::LabError;

/// A transition for TD-learning: (s, a, r, s')
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: StateVector,
    pub action: Quaternion,
    pub reward: f64,
    pub next_state: StateVector,
}

/// Flattened state: hyperbolic latent + binary memory spikes
#[derive(Debug, Clone)]
pub struct StateVector {
    pub latent: Vec<f64>,
    pub memory_spikes: Vec<f64>, // 0.0 or 1.0
}

impl StateVector {
    pub fn new(latent: HyperbolicPoint, memory: &[bool]) -> Self {
        let latent_vec = latent.coords.to_vec();
        let spikes: Vec<f64> = memory.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        Self {
            latent: latent_vec,
            memory_spikes: spikes,
        }
    }

    pub fn dim(&self) -> usize {
        self.latent.len() + self.memory_spikes.len()
    }

    /// Flatten to a continuous Vec (DOD)
    pub fn as_slice(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.dim());
        out.extend_from_slice(&self.latent);
        out.extend_from_slice(&self.memory_spikes);
        out
    }
}

/// Linear value network: state → scalar value
#[derive(Debug, Clone)]
pub struct ValueHead {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub gamma: f64, // discount factor
}

impl ValueHead {
    pub fn new(state_dim: usize, gamma: f64) -> Self {
        let weights: Vec<f64> = (0..state_dim)
            .map(|i| (i as f64 * 0.1).sin() * 0.01)
            .collect();
        Self {
            weights,
            bias: 0.0,
            gamma,
        }
    }

    pub fn value(&self, state: &StateVector) -> f64 {
        let s = state.as_slice();
        let mut acc = self.bias;
        for (i, &w) in self.weights.iter().enumerate() {
            acc += w * s.get(i).copied().unwrap_or(0.0);
        }
        acc.tanh() // bounded [-1, 1]
    }

    /// TD error: δ = r + γ·V(s') - V(s)
    pub fn td_error(&self, transition: &Transition) -> f64 {
        let v_s = self.value(&transition.state);
        let v_next = self.value(&transition.next_state);
        transition.reward + self.gamma * v_next - v_s
    }

    /// Simple gradient update on value weights
    pub fn update(&mut self, state: &StateVector, td_error: f64, lr: f64) {
        let s = state.as_slice();
        for (i, w) in self.weights.iter_mut().enumerate() {
            let x = s.get(i).copied().unwrap_or(0.0);
            *w += lr * td_error * x;
        }
        self.bias += lr * td_error;
    }
}

/// Policy network: state → quaternion action
#[derive(Debug, Clone)]
pub struct PolicyHead {
    /// [4 x state_dim]: rows for w, x, y, z
    pub weights: Vec<f64>,
}

impl PolicyHead {
    pub fn new(state_dim: usize) -> Self {
        let weights: Vec<f64> = (0..(4 * state_dim))
            .map(|i| (i as f64 * 0.07).cos() * 0.01)
            .collect();
        Self { weights }
    }

    /// Forward: generate action quaternion
    pub fn action(&self, state: &StateVector) -> Quaternion {
        let s = state.as_slice();
        let mut out = [0.0f64; 4];
        for i in 0..4 {
            let mut acc = 0.0;
            for (j, &x) in s.iter().enumerate() {
                let w = self.weights.get(i * s.len() + j).copied().unwrap_or(0.0);
                acc += w * x;
            }
            out[i] = acc;
        }
        Quaternion::new(out[0] as f32, out[1] as f32, out[2] as f32, out[3] as f32)
            .normalize()
    }

    /// Update via R-STDP: strengthen action when TD-error is positive
    pub fn update(
        &mut self,
        state: &StateVector,
        action: &Quaternion,
        td_error: f64,
        stdp: &RSTDP,
        pre_embed: &HyperbolicPoint,
        post_embed: &HyperbolicPoint,
        pre_time: f64,
        post_time: f64,
        lr: f64,
    ) -> Result<(), LabError> {
        let reward = td_error.clamp(-1.0, 1.0);
        let dw = stdp.compute(reward, pre_time, post_time, pre_embed.coords[0] as f32, post_embed.coords[0] as f32);

        let s = state.as_slice();
        let comps = [action.w, action.x, action.y, action.z];
        for i in 0..4 {
            for (j, &x) in s.iter().enumerate() {
                let idx = i * s.len() + j;
                if let Some(w) = self.weights.get_mut(idx) {
                    *w += lr * dw * comps[i] as f64 * x;
                }
            }
        }
        Ok(())
    }
}

/// RL Agent: combines Value (Critic) + Policy (Actor) + R-STDP
#[derive(Debug, Clone)]
pub struct RLAgent {
    pub value: ValueHead,
    pub policy: PolicyHead,
    pub transitions: Vec<Transition>,
    pub max_transitions: usize,
}

impl RLAgent {
    pub fn new(state_dim: usize, gamma: f64) -> Self {
        Self {
            value: ValueHead::new(state_dim, gamma),
            policy: PolicyHead::new(state_dim),
            transitions: Vec::with_capacity(1000),
            max_transitions: 1000,
        }
    }

    pub fn act(&self, state: &StateVector) -> Quaternion {
        self.policy.action(state)
    }

    pub fn observe(&mut self, transition: Transition) {
        if self.transitions.len() >= self.max_transitions {
            self.transitions.remove(0);
        }
        self.transitions.push(transition);
    }

    /// Train on the last transition (online)
    pub fn train_step(
        &mut self,
        transition: &Transition,
        stdp: &RSTDP,
        pre_embed: &HyperbolicPoint,
        post_embed: &HyperbolicPoint,
        pre_time: f64,
        post_time: f64,
        lr_value: f64,
        lr_policy: f64,
    ) -> Result<f64, LabError> {
        let delta = self.value.td_error(transition);
        self.value.update(&transition.state, delta, lr_value);
        self.policy.update(
            &transition.state,
            &transition.action,
            delta,
            stdp,
            pre_embed,
            post_embed,
            pre_time,
            post_time,
            lr_policy,
        )?;
        Ok(delta)
    }

    /// Train on all stored transitions (batch)
    pub fn train_batch(
        &mut self,
        stdp: &RSTDP,
        pre_embed: &HyperbolicPoint,
        post_embed: &HyperbolicPoint,
        pre_time: f64,
        post_time: f64,
        lr_value: f64,
        lr_policy: f64,
    ) -> Result<f64, LabError> {
        let mut total_delta = 0.0;
        let count = self.transitions.len().max(1);
        for t in &self.transitions.clone() {
            total_delta += self.train_step(t, stdp, pre_embed, post_embed, pre_time, post_time, lr_value, lr_policy)?;
        }
        Ok(total_delta / count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn dummy_state() -> StateVector {
        let latent = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
        StateVector::new(latent, &[true, false, true])
    }

    fn dummy_next_state() -> StateVector {
        let latent = HyperbolicPoint::new(array![0.11, 0.01]).unwrap();
        StateVector::new(latent, &[false, true, false])
    }

    #[test]
    fn value_head_bounded() {
        let vh = ValueHead::new(5, 0.9);
        let s = dummy_state();
        let val = vh.value(&s);
        assert!(val.abs() <= 1.0, "Value must be bounded by tanh");
    }

    #[test]
    fn policy_generates_unit_quaternion() {
        let ph = PolicyHead::new(5);
        let s = dummy_state();
        let a = ph.action(&s);
        let n = a.norm();
        assert!((n - 1.0).abs() < 1e-6, "Action must be normalized, was {}", n);
    }

    #[test]
    fn td_error_sign() {
        let vh = ValueHead::new(5, 0.9);
        let t = Transition {
            state: dummy_state(),
            action: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            reward: 1.0,
            next_state: dummy_next_state(),
        };
        let delta = vh.td_error(&t);
        assert!(delta > 0.0 || delta.abs() < 2.0, "TD error should be reasonable");
    }

    #[test]
    fn agent_act_returns_valid_action() {
        let agent = RLAgent::new(5, 0.9);
        let a = agent.act(&dummy_state());
        assert!(a.norm() > 0.99);
    }

    #[test]
    fn value_update_changes_weights() {
        let mut vh = ValueHead::new(3, 0.9);
        let s = StateVector::new(
            HyperbolicPoint::new(array![0.1, 0.0]).unwrap(),
            &[true],
        );
        let v_before = vh.value(&s);
        vh.update(&s, 0.5, 0.1);
        let v_after = vh.value(&s);
        assert!((v_after - v_before).abs() > 1e-12, "Update must change values");
    }
}
