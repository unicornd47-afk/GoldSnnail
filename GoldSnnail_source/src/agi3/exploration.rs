//! Curiosity-driven exploration and intrinsic motivation.
//!
//! Uses prediction error (curiosity) and count-based novelty bonuses to
//! encourage the agent to visit novel states.

use crate::agi3::Action;
use crate::geometry::HyperbolicPoint;

// ─── Exploration Engine ───────────────────────────────────────────────────────

/// Curiosity and novelty-driven exploration engine.
#[derive(Debug, Clone)]
pub struct ExplorationEngine {
    pub visit_counts: std::collections::HashMap<String, usize>,
    pub curiosity_weight: f64,
    pub novelty_weight: f64,
    pub total_intrinsic_reward: f64,
    pub discretization_bins: usize,
}

impl ExplorationEngine {
    pub fn new(curiosity_weight: f64, novelty_weight: f64) -> Self {
        Self {
            visit_counts: std::collections::HashMap::new(),
            curiosity_weight,
            novelty_weight,
            total_intrinsic_reward: 0.0,
            discretization_bins: 10,
        }
    }

    pub fn reset(&mut self) {
        self.visit_counts.clear();
        self.total_intrinsic_reward = 0.0;
    }

    /// Discretizes a hyperbolic point into a string key.
    pub fn state_key(&self, state: &HyperbolicPoint) -> String {
        let bins = self.discretization_bins;
        state
            .coords
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-0.99, 0.99);
                let bin = ((clamped + 0.99) / 1.98 * (bins as f64)).floor() as usize;
                bin.min(bins - 1).to_string()
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Returns how many times this state has been visited.
    pub fn visit_count(&self, state: &HyperbolicPoint) -> usize {
        let key = self.state_key(state);
        *self.visit_counts.get(&key).unwrap_or(&0)
    }

    /// Records a visit to this state.
    pub fn record_visit(&mut self, state: &HyperbolicPoint) {
        let key = self.state_key(state);
        *self.visit_counts.entry(key).or_insert(0) += 1;
    }

    /// Computes total intrinsic reward for this step.
    ///
    /// Combines curiosity (prediction error) and novelty (visit count bonus).
    pub fn intrinsic_reward(&mut self, state: &HyperbolicPoint, prediction_error: f64) -> f64 {
        let r_curiosity = self.curiosity_weight * prediction_error;
        let count = self.visit_count(state);
        let r_novelty = self.novelty_weight * (1.0 / (1.0 + count as f64));
        let total = r_curiosity + r_novelty;

        self.record_visit(state);
        self.total_intrinsic_reward += total;
        total
    }

    /// Total intrinsic reward accumulated this episode.
    pub fn total_intrinsic(&self) -> f64 {
        self.total_intrinsic_reward
    }

    /// Resets episode accumulator but keeps visit counts.
    pub fn reset_episode(&mut self) {
        self.total_intrinsic_reward = 0.0;
    }
}

// ─── UCB Action Selector ──────────────────────────────────────────────────────

/// UCB1-based action selector for exploration.
#[derive(Debug, Clone)]
pub struct UCBExplorer {
    pub action_counts: std::collections::HashMap<u8, usize>,
    pub action_values: std::collections::HashMap<u8, f64>,
    pub exploration_constant: f64,
}

impl UCBExplorer {
    pub fn new() -> Self {
        Self {
            action_counts: std::collections::HashMap::new(),
            action_values: std::collections::HashMap::new(),
            exploration_constant: 1.414,
        }
    }

    /// Map Action to u8 index.
    pub fn action_index(action: Action) -> u8 {
        match action {
            Action::Noop => 0,
            Action::Up => 1,
            Action::Down => 2,
            Action::Left => 3,
            Action::Right => 4,
            Action::Interact => 5,
        }
    }

    /// Select an action using UCB1.
    pub fn select_action(&self, available_actions: &[Action]) -> Action {
        let total_visits: usize = self.action_counts.values().sum();

        let mut best_action = available_actions[0];
        let mut best_score = f64::NEG_INFINITY;

        for &action in available_actions {
            let idx = Self::action_index(action);
            let count = self.action_counts.get(&idx).copied().unwrap_or(0);
            let value = self.action_values.get(&idx).copied().unwrap_or(0.0);

            let score = if count == 0 {
                f64::INFINITY // Explore untried actions first
            } else {
                value + self.exploration_constant * ((total_visits as f64).ln() / (count as f64)).sqrt()
            };

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        best_action
    }

    /// Update action value estimate.
    pub fn update(&mut self, action: Action, reward: f64) {
        let idx = Self::action_index(action);
        let count = self.action_counts.entry(idx).or_insert(0);
        *count += 1;

        let old_value = self.action_values.entry(idx).or_insert(0.0);
        *old_value += (reward - *old_value) / (*count as f64);
    }
}

// ─── Action Selector ──────────────────────────────────────────────────────────

/// Strategy for selecting actions from value estimates.
#[derive(Debug, Clone)]
pub enum ActionSelector {
    Greedy,
    UCB(UCBExplorer),
    EpsilonGreedy { epsilon: f64 },
}

impl ActionSelector {
    pub fn select(&self, action_values: &[f64; 6]) -> usize {
        match self {
            ActionSelector::Greedy => {
                let mut best = 0;
                let mut best_val = action_values[0];
                for (i, &v) in action_values.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best = i;
                    }
                }
                best
            }
            ActionSelector::UCB(explorer) => {
                let actions = vec![
                    Action::Noop, Action::Up, Action::Down,
                    Action::Left, Action::Right, Action::Interact,
                ];
                let chosen = explorer.select_action(&actions);
                UCBExplorer::action_index(chosen) as usize
            }
            ActionSelector::EpsilonGreedy { epsilon } => {
                if rand::random::<f64>() < *epsilon {
                    rand::random::<usize>() % 6
                } else {
                    let mut best = 0;
                    let mut best_val = action_values[0];
                    for (i, &v) in action_values.iter().enumerate() {
                        if v > best_val {
                            best_val = v;
                            best = i;
                        }
                    }
                    best
                }
            }
        }
    }

    pub fn update(&mut self, action_idx: usize, reward: f64) {
        if let ActionSelector::UCB(explorer) = self {
            let action = match action_idx {
                0 => Action::Noop,
                1 => Action::Up,
                2 => Action::Down,
                3 => Action::Left,
                4 => Action::Right,
                5 => Action::Interact,
                _ => Action::Noop,
            };
            explorer.update(action, reward);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exploration_engine_creation() {
        let engine = ExplorationEngine::new(0.1, 0.05);
        assert_eq!(engine.curiosity_weight, 0.1);
    }

    #[test]
    fn exploration_engine_intrinsic_reward() {
        let mut engine = ExplorationEngine::new(0.1, 0.05);
        let state = HyperbolicPoint { coords: vec![0.1, 0.2] };
        let r = engine.intrinsic_reward(&state, 0.5);
        assert!(r > 0.0);
        // Second visit should have lower novelty bonus
        let r2 = engine.intrinsic_reward(&state, 0.5);
        assert!(r2 < r);
    }

    #[test]
    fn ucb_selects_unvisited_action() {
        let explorer = UCBExplorer::new();
        let actions = vec![Action::Noop, Action::Up, Action::Interact];
        let chosen = explorer.select_action(&actions);
        // Should select one of the available actions
        assert!(actions.contains(&chosen));
    }

    #[test]
    fn ucb_update_changes_values() {
        let mut explorer = UCBExplorer::new();
        explorer.update(Action::Up, 1.0);
        explorer.update(Action::Up, 0.5);
        assert_eq!(*explorer.action_values.get(&1).unwrap(), 0.75);
    }

    #[test]
    fn action_selector_greedy() {
        let selector = ActionSelector::Greedy;
        let values = [0.1, 0.5, 0.3, 0.2, 0.4, 0.0];
        assert_eq!(selector.select(&values), 1);
    }

    #[test]
    fn action_selector_epsilon_greedy() {
        let selector = ActionSelector::EpsilonGreedy { epsilon: 0.0 };
        let values = [0.1, 0.9, 0.3, 0.2, 0.4, 0.0];
        assert_eq!(selector.select(&values), 1);
    }
}
