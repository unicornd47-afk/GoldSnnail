//! ARC-AGI-3 Interactive Agent
//!
//! The top-level `ArcAgent3` orchestrates perception, memory, planning,
//! exploration, and action selection to interact with ARC-AGI-3 environments.
//!
//! # Agent Loop
//!
//! ```text
//! observe → perceive → encode → reason (plan) → act → learn
//! ```

use crate::agi3::{
    Action, AgentConfig, EpisodeResult, Observation,
};
use crate::agi3::environment::ArcEnvironment;
use crate::agi3::exploration::{ActionSelector, ExplorationEngine, UCBExplorer};
use crate::agi3::memory::WorkingMemory;
use crate::agi3::perception::PerceptionPipeline;
use crate::agi3::planning::{GoalDetector, Planner, plan_actions};
use crate::geometry::PoincareBall;
use crate::rl::{RLAgent, StateVector};
use crate::plasticity::RSTDP;

// ─── Main Agent Struct ────────────────────────────────────────────────────────

/// The ARC-AGI-3 interactive agent.
///
/// Integrates all subsystems:
/// - `PerceptionPipeline` for encoding observations into SNN-compatible spikes
/// - `WorkingMemory` for maintaining temporal context
/// - `Planner` for model-based lookahead using the World Model
/// - `GoalDetector` for discovering desirable states
/// - `RLAgent` for policy/value learning
/// - `ExplorationEngine` for curiosity-driven exploration
#[derive(Debug, Clone)]
pub struct ArcAgent3 {
    pub config: AgentConfig,
    pub perception: PerceptionPipeline,
    pub memory: WorkingMemory,
    pub planner: Planner,
    pub goal_detector: GoalDetector,
    pub rl_agent: RLAgent,
    pub exploration: ExplorationEngine,
    pub action_selector: ActionSelector,
    pub rng: rand::rngs::ThreadRng,
}

impl ArcAgent3 {
    /// Creates a new ARC-AGI-3 agent with default configuration.
    pub fn new(config: AgentConfig) -> Self {
        let perception = PerceptionPipeline::new(
            config.input_dim,
            32,
            16,
            0.75,
            0.06,
        );
        let memory = WorkingMemory::new(50);
        let planner = Planner::new(16, 32, 1.0, config.planning_horizon);
        let goal_detector = GoalDetector::new();
        let state_dim = 16 + 30; // 16D hyperbolic + 30 sensor spikes
        let rl_agent = RLAgent::new(state_dim, config.discount_gamma);
        let exploration = ExplorationEngine::new(
            config.exploration_beta,
            config.curiosity_weight,
        );
        let action_selector = ActionSelector::UCB(UCBExplorer::new());

        Self {
            config,
            perception,
            memory,
            planner,
            goal_detector,
            rl_agent,
            exploration,
            action_selector,
            rng: rand::thread_rng(),
        }
    }

    /// Runs a single episode in the given environment.
    pub fn run_episode(&mut self, env: &mut dyn ArcEnvironment) -> EpisodeResult {
        let mut total_reward = 0.0;
        let mut steps = 0u64;
        let mut solved = false;

        // Reset all subsystems
        self.memory.clear();
        self.planner.reset();
        self.goal_detector = GoalDetector::new();
        self.exploration.reset();
        if let ActionSelector::UCB(ref mut explorer) = self.action_selector {
            *explorer = UCBExplorer::new();
        }

        // Reset environment and get first observation
        let obs = env.reset().expect("Environment reset failed");
        let episode_id = obs.episode_id.clone();

        // Process first observation
        let mut h_state = self.perception.process_observation(&obs)
            .expect("Perception pipeline failed");
        let spikes = self.perception.grid_to_spikes(
            &crate::vision::ArcGrid::from_2d(&obs.frame)
        );

        self.planner.observe(&h_state);
        self.memory.push(obs.clone(), Action::Noop, 0.0, h_state.clone(), spikes);

        let ball = PoincareBall::new(1.0);

        for step in 0..self.config.max_episode_steps {
            steps = step + 1;

            // Build state vector for RL
            let sensor_activations = self.perception.sensor_activations();
            let state_vec = StateVector::new(
                crate::geometry::HyperbolicPoint { coords: h_state.coords.clone() },
                &sensor_activations.iter().map(|&v| v > 0.0).collect::<Vec<_>>(),
            );

            // Check if we're near a goal
            let _near_goal = self.goal_detector.is_near_goal(&h_state, &ball);

            // Plan next action
            let goal = self.goal_detector.find_goal();
            let available_actions = vec![
                Action::Noop, Action::Up, Action::Down,
                Action::Left, Action::Right, Action::Interact,
            ];
            let plan = plan_actions(
                &h_state,
                goal.as_ref(),
                &available_actions,
                &mut self.planner,
                self.config.planning_horizon,
            );

            // Select action using RL policy + exploration
            let action_values = self.compute_action_values(&state_vec, &plan, &h_state, &ball);
            let action_idx = self.action_selector.select(&action_values);
            let action = match action_idx {
                0 => Action::Noop,
                1 => Action::Up,
                2 => Action::Down,
                3 => Action::Left,
                4 => Action::Right,
                5 => Action::Interact,
                _ => Action::Noop,
            };

            // Take environment step
            let step_result = env.step(action).expect("Environment step failed");
            let extrinsic = step_result.reward;
            total_reward += extrinsic;

            // Compute intrinsic reward (curiosity)
            let next_h_state = self.perception.process_observation(&step_result.observation)
                .expect("Perception pipeline failed on next obs");
            let prediction_err = self.planner.intrinsic_reward(&h_state, &next_h_state)
                .unwrap_or(0.0);
            let intrinsic = self.exploration.intrinsic_reward(&next_h_state, prediction_err);

            let total_step_reward = extrinsic + self.config.curiosity_weight * intrinsic;

            // Record in memory
            let next_spikes = self.perception.grid_to_spikes(
                &crate::vision::ArcGrid::from_2d(&step_result.observation.frame)
            );
            self.memory.push(
                step_result.observation.clone(),
                action,
                total_step_reward,
                next_h_state.clone(),
                next_spikes,
            );

            // Update goal detector
            self.goal_detector.update(&next_h_state, extrinsic);

            // Train world model on transition
            let _ = self.planner.train_step(&h_state, &next_h_state, 0.01);

            // RL learning
            let next_sensor = self.perception.sensor_activations();
            let next_state_vec = StateVector::new(
                crate::geometry::HyperbolicPoint { coords: next_h_state.coords.clone() },
                &next_sensor.iter().map(|&v| v > 0.0).collect::<Vec<_>>(),
            );

            let transition = crate::rl::Transition {
                state: state_vec,
                action: crate::geometry::Quaternion::new(
                    1.0, 0.0, 0.0, 0.0
                ),
                reward: total_step_reward,
                next_state: next_state_vec,
            };

            let _ = self.rl_agent.train_step(
                &transition,
                &RSTDP::new(0.01, 20.0, 1.0),
                &h_state,
                &next_h_state,
                0.0,
                1.0,
                0.01,
                0.005,
            );

            // Update action selector
            self.action_selector.update(action_idx, total_step_reward);

            // Prepare for next iteration
            h_state = next_h_state;

            // Check termination
            if step_result.done {
                solved = total_reward > 5.0;
                break;
            }

            if extrinsic > 5.0 {
                solved = true;
                break;
            }
        }

        EpisodeResult {
            episode_id,
            total_reward,
            steps,
            solved,
            final_state: self.memory.recent_observation().cloned(),
        }
    }

    /// Compute action values for all 6 actions using RL + planning signals.
    fn compute_action_values(
        &self,
        state: &StateVector,
        plan: &crate::agi3::planning::Plan,
        h_state: &crate::geometry::HyperbolicPoint,
        ball: &PoincareBall,
    ) -> [f64; 6] {
        let mut values = [0.0f64; 6];

        // Base values from RL policy head (quaternion → scalar per action)
        let quat_action = self.rl_agent.act(state);
        let base = quat_action.norm();

        for i in 0..6 {
            // RL base value
            values[i] = base as f64 * 0.3;

            // Planning signal: if this action is in the plan, boost it
            if plan.actions.contains(&match i {
                0 => Action::Noop,
                1 => Action::Up,
                2 => Action::Down,
                3 => Action::Left,
                4 => Action::Right,
                5 => Action::Interact,
                _ => Action::Noop,
            }) {
                values[i] += plan.expected_reward * 0.5;
            }

            // Goal proximity signal
            if let Some(goal) = self.goal_detector.find_goal() {
                let sim_state = match i {
                    1 => {
                        let mut s = h_state.clone();
                        s.coords[1] += 0.02;
                        s
                    }
                    2 => {
                        let mut s = h_state.clone();
                        s.coords[1] -= 0.02;
                        s
                    }
                    3 => {
                        let mut s = h_state.clone();
                        s.coords[0] -= 0.02;
                        s
                    }
                    4 => {
                        let mut s = h_state.clone();
                        s.coords[0] += 0.02;
                        s
                    }
                    5 => {
                        let mut s = h_state.clone();
                        s.coords[0] += 0.01;
                        s.coords[1] += 0.01;
                        s
                    }
                    _ => h_state.clone(),
                };
                let dist = ball.distance(&sim_state, &goal).unwrap_or(1.0);
                values[i] -= dist * 0.2;
            }

            // Small noise for exploration
            values[i] += (rand::random::<f64>() - 0.5) * 0.05;
        }

        values
    }
}

// ─── Convenience Constructor ─────────────────────────────────────────────────

impl Default for ArcAgent3 {
    fn default() -> Self {
        Self::new(AgentConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agi3::planning::Plan;
    use crate::geometry::HyperbolicPoint;
    use crate::vision::ArcGrid;

    #[test]
    fn agent_creation() {
        let agent = ArcAgent3::default();
        assert_eq!(agent.config.max_episode_steps, 200);
    }

    #[test]
    fn agent_run_episode_returns_result() {
        let mut agent = ArcAgent3::default();
        let mut env = crate::agi3::environment::GridEnvironment::new(
            "test",
            ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap(),
            ArcGrid::from_data(vec![vec![0, 1], vec![2, 0]]).unwrap(),
            10,
        );
        let result = agent.run_episode(&mut env);
        assert_eq!(result.episode_id, "test");
        assert!(result.steps <= 10);
    }

    #[test]
    fn agent_run_episode_solves_simple_env() {
        let g = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let mut solved = 0;
        for _ in 0..20 {
            let mut agent = ArcAgent3::default();
            let mut env = crate::agi3::environment::GridEnvironment::new(
                "identity",
                g.clone(),
                g.clone(),
                20,
            );
            let result = agent.run_episode(&mut env);
            if result.solved {
                solved += 1;
            }
        }
        assert!(solved > 0, "Agent never solved identity env in 20 episodes (solved {})", solved);
    }

    #[test]
    fn compute_action_values_returns_six_values() {
        let agent = ArcAgent3::default();
        let state = StateVector::new(
            HyperbolicPoint { coords: vec![0.1, 0.0] },
            &[true, false],
        );
        let plan = Plan::new(vec![]);
        let ball = PoincareBall::new(1.0);
        let h_state = HyperbolicPoint { coords: vec![0.1, 0.0] };
        let values = agent.compute_action_values(&state, &plan, &h_state, &ball);
        assert_eq!(values.len(), 6);
    }
}
