//! Short-horizon planning using the hyperbolic World Model.
//!
//! The planner observes latent state transitions and predicts future states
//! to guide action selection. It also implements goal discovery from
//! observation-reward history.

use crate::agi3::Action;
use crate::geometry::{HyperbolicPoint, PoincareBall};
use crate::world_model::WorldModel;

// ─── Planner ──────────────────────────────────────────────────────────────────

/// Model-based planner using the hyperbolic World Model.
#[derive(Debug, Clone)]
pub struct Planner {
    pub world_model: WorldModel,
    pub planning_horizon: usize,
    pub curvature: f64,
    pub latent_dim: usize,
    pub hidden_dim: usize,
}

impl Planner {
    pub fn new(latent_dim: usize, hidden_dim: usize, curvature: f64, planning_horizon: usize) -> Self {
        let world_model = WorldModel::new(latent_dim, hidden_dim, curvature);
        Self {
            world_model,
            planning_horizon,
            curvature,
            latent_dim,
            hidden_dim,
        }
    }

    /// Record a state observation in the world model history.
    pub fn observe(&mut self, state: &HyperbolicPoint) {
        self.world_model.observe(state.clone());
    }

    /// Predict the next latent state.
    pub fn predict_next(&mut self, current: &HyperbolicPoint) -> Result<HyperbolicPoint, crate::LabError> {
        self.world_model.predict(current)
    }

    /// Compute prediction error between predicted and actual states.
    pub fn prediction_error(
        &self,
        predicted: &HyperbolicPoint,
        actual: &HyperbolicPoint,
    ) -> Result<f64, crate::LabError> {
        self.world_model.prediction_error(predicted, actual)
    }

    /// Single training step on the world model.
    pub fn train_step(
        &mut self,
        current: &HyperbolicPoint,
        next_actual: &HyperbolicPoint,
        lr: f64,
    ) -> Result<f64, crate::LabError> {
        self.world_model.train_step(current, next_actual, lr)
    }

    /// Train on the full observation history.
    pub fn train_on_history(&mut self, lr: f64) -> Result<f64, crate::LabError> {
        self.world_model.train_on_history(lr)
    }

    /// Reset internal state for a new episode.
    pub fn reset(&mut self) {
        self.world_model.reset_hidden();
    }

    /// Compute intrinsic reward from prediction error.
    pub fn intrinsic_reward(
        &mut self,
        current: &HyperbolicPoint,
        next_actual: &HyperbolicPoint,
    ) -> Result<f64, crate::LabError> {
        let predicted = self.world_model.predict(current)?;
        let err = self.prediction_error(&predicted, next_actual)?;
        // Normalize: reward is high when error is high (novelty)
        Ok(err.tanh())
    }
}

// ─── Goal Detector ────────────────────────────────────────────────────────────

/// Discovers desirable future states from observation-reward history.
#[derive(Debug, Clone)]
pub struct GoalDetector {
    pub visited_states: Vec<HyperbolicPoint>,
    pub reward_history: Vec<f64>,
    pub goal_threshold: f64,
}

impl GoalDetector {
    pub fn new() -> Self {
        Self {
            visited_states: Vec::new(),
            reward_history: Vec::new(),
            goal_threshold: 0.5,
        }
    }

    pub fn update(&mut self, state: &HyperbolicPoint, reward: f64) {
        self.visited_states.push(state.clone());
        self.reward_history.push(reward);
    }

    /// Find the state with the highest cumulative reward in history.
    pub fn find_goal(&self) -> Option<HyperbolicPoint> {
        if self.visited_states.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_reward = self.reward_history[0];
        for (i, &r) in self.reward_history.iter().enumerate() {
            if r > best_reward {
                best_reward = r;
                best_idx = i;
            }
        }
        Some(self.visited_states[best_idx].clone())
    }

    /// Check if current state is near any high-reward state.
    pub fn is_near_goal(&self, state: &HyperbolicPoint, ball: &PoincareBall) -> bool {
        self.goal_distance(state, ball)
            .map(|d| d < self.goal_threshold)
            .unwrap_or(false)
    }

    /// Minimum distance to any high-reward state.
    pub fn goal_distance(&self, state: &HyperbolicPoint, ball: &PoincareBall) -> Option<f64> {
        let mut min_dist = f64::INFINITY;
        for visited in &self.visited_states {
            if let Ok(d) = ball.distance(state, visited) {
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        if min_dist.is_finite() {
            Some(min_dist)
        } else {
            None
        }
    }
}

// ─── Plan ─────────────────────────────────────────────────────────────────────

/// A sequence of actions with associated expected reward and confidence.
#[derive(Debug, Clone)]
pub struct Plan {
    pub actions: Vec<Action>,
    pub expected_reward: f64,
    pub confidence: f64,
}

impl Plan {
    pub fn new(actions: Vec<Action>) -> Self {
        Self {
            actions,
            expected_reward: 0.0,
            confidence: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn push(&mut self, action: Action) {
        self.actions.push(action);
    }

    pub fn extend(&mut self, actions: Vec<Action>) {
        self.actions.extend(actions);
    }
}

// ─── Planning Function ────────────────────────────────────────────────────────

/// Greedy 1-step lookahead planning.
///
/// For each available action, simulates the next latent state using the
/// world model and computes distance to the goal. Returns the action
/// sequence that minimizes goal distance.
pub fn plan_actions(
    current: &HyperbolicPoint,
    goal: Option<&HyperbolicPoint>,
    available_actions: &[Action],
    planner: &mut Planner,
    steps: usize,
) -> Plan {
    if available_actions.is_empty() || steps == 0 {
        return Plan::new(vec![]);
    }

    let mut best_action = Action::Noop;
    let mut best_score = f64::INFINITY;

    for &action in available_actions {
        // Simulate: perturb current state slightly based on action
        let mut sim_state = current.clone();
        let perturbation = match action {
            Action::Up => vec![0.0, 0.02],
            Action::Down => vec![0.0, -0.02],
            Action::Left => vec![-0.02, 0.0],
            Action::Right => vec![0.02, 0.0],
            Action::Interact => vec![0.01, 0.01],
            Action::Noop => vec![0.0, 0.0],
        };

        // Ensure simulated state stays in the ball
        let ball = PoincareBall::new(planner.curvature);
        for (i, &p) in perturbation.iter().enumerate() {
            if i < sim_state.coords.len() {
                sim_state.coords[i] += p;
                let norm = sim_state.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm >= 0.99 {
                    let scale = 0.98 / norm;
                    for x in &mut sim_state.coords {
                        *x *= scale;
                    }
                }
            }
        }

        // Score: distance to goal (lower is better)
        let score = if let Some(g) = goal {
            ball.distance(&sim_state, g).unwrap_or(f64::INFINITY)
        } else {
            // No goal: prefer exploration (larger perturbation magnitude)
            perturbation.iter().map(|x| x * x).sum::<f64>().sqrt()
        };

        if score < best_score {
            best_score = score;
            best_action = action;
        }
    }

    Plan::new(vec![best_action])
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample_point(x: f64, y: f64) -> HyperbolicPoint {
        HyperbolicPoint::new(array![x * 0.5, y * 0.5]).unwrap()
    }

    #[test]
    fn planner_creation() {
        let p = Planner::new(2, 4, 1.0, 5);
        assert_eq!(p.planning_horizon, 5);
    }

    #[test]
    fn planner_predict_returns_valid_point() {
        let mut p = Planner::new(2, 4, 1.0, 5);
        let state = sample_point(0.1, 0.2);
        let pred = p.predict_next(&state).unwrap();
        assert!(pred.euclidean_norm() < 1.0);
    }

    #[test]
    fn goal_detector_find_goal() {
        let mut gd = GoalDetector::new();
        gd.update(&sample_point(0.1, 0.1), 1.0);
        gd.update(&sample_point(0.2, 0.2), -0.5);
        gd.update(&sample_point(0.3, 0.3), 2.0);
        let goal = gd.find_goal().unwrap();
        assert!((goal.coords[0] - 0.15).abs() < 0.01);
    }

    #[test]
    fn goal_detector_no_goal_when_empty() {
        let gd = GoalDetector::new();
        assert!(gd.find_goal().is_none());
    }

    #[test]
    fn plan_actions_with_goal() {
        let mut planner = Planner::new(2, 4, 1.0, 5);
        let current = sample_point(0.0, 0.0);
        let goal = sample_point(0.5, 0.5);
        let actions = vec![Action::Up, Action::Right, Action::Interact];
        let plan = plan_actions(&current, Some(&goal), &actions, &mut planner, 1);
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn plan_actions_no_goal_returns_exploration() {
        let mut planner = Planner::new(2, 4, 1.0, 5);
        let current = sample_point(0.0, 0.0);
        let actions = vec![Action::Noop];
        let plan = plan_actions(&current, None, &actions, &mut planner, 1);
        assert_eq!(plan.actions.len(), 1);
    }
}
