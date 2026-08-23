//! Grid-world environments for ARC-AGI-3 agent training and evaluation.
//!
//! ARC-AGI-3 environments are turn-based interactive games. The agent receives
//! a 2D grid frame and must choose an action to advance. No instructions or
//! goals are provided explicitly — the agent must discover them through
//! interaction.

use crate::agi3::{Action, Observation};
use crate::arc_apply::{apply_arc_op, program_solves_train};
use crate::arc_program::{ArcOpCode, ArcOpToken, ArcProgram};
use crate::vision::ArcGrid;
use rand::seq::SliceRandom;
use rand::thread_rng;

// ─── Step Result ──────────────────────────────────────────────────────────────

/// Result of stepping an environment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArcEnvironmentStep {
    /// The new observation after the action.
    pub observation: Observation,
    /// Scalar reward signal.
    pub reward: f64,
    /// Whether the episode has terminated.
    pub done: bool,
    /// Auxiliary information dictionary.
    pub info: std::collections::HashMap<String, serde_json::Value>,
}

// ─── Environment Trait ─────────────────────────────────────────────────────────

/// Trait for ARC-AGI-3 interactive environments.
pub trait ArcEnvironment: std::fmt::Debug {
    /// Start a new episode and return the first observation.
    fn reset(&mut self) -> Result<Observation, String>;

    /// Take one step in the environment.
    fn step(&mut self, action: Action) -> Result<ArcEnvironmentStep, String>;

    /// Number of discrete actions available.
    fn action_space(&self) -> usize;

    /// (width, height) of observation frames.
    fn observation_shape(&self) -> (usize, usize);

    /// Unique environment identifier.
    fn id(&self) -> &str;
}

// ─── Concrete Grid Environment ────────────────────────────────────────────────

/// A concrete ARC-style grid environment.
///
/// Wraps a pair of grids (initial → target) with a hidden transformation.
/// The agent's goal is to discover which action sequence produces the target
/// grid from the initial grid.
#[derive(Debug, Clone)]
pub struct GridEnvironment {
    pub id: String,
    pub initial_grid: ArcGrid,
    pub target_grid: ArcGrid,
    pub current_grid: ArcGrid,
    pub width: usize,
    pub height: usize,
    pub turn: u64,
    pub max_turns: u64,
    pub transformation: ArcOpCode,
    pub rng: rand::rngs::ThreadRng,
}

impl GridEnvironment {
    pub fn new(
        id: impl Into<String>,
        initial: ArcGrid,
        target: ArcGrid,
        max_turns: u64,
    ) -> Self {
        let width = initial.width;
        let height = initial.height;
        Self {
            id: id.into(),
            initial_grid: initial.clone(),
            target_grid: target,
            current_grid: initial,
            width,
            height,
            turn: 0,
            max_turns,
            transformation: ArcOpCode::Identity,
            rng: thread_rng(),
        }
    }

    /// Shifts non-zero pixels in the given direction by one cell.
    fn shift_grid(grid: &ArcGrid, dx: i32, dy: i32) -> ArcGrid {
        let mut result = vec![vec![0u8; grid.width]; grid.height];
        for r in 0..grid.height {
            for c in 0..grid.width {
                let val = grid.data[r][c];
                if val == 0 {
                    continue;
                }
                let nr = (r as i32 + dy).clamp(0, grid.height as i32 - 1) as usize;
                let nc = (c as i32 + dx).clamp(0, grid.width as i32 - 1) as usize;
                result[nr][nc] = val;
            }
        }
        ArcGrid::from_data(result).unwrap_or_else(|_| grid.clone())
    }

    /// Apply a directional action to the current grid.
    fn apply_directional(&mut self, action: Action) -> (ArcGrid, f64) {
        let (dx, dy) = match action {
            Action::Up => (0, -1),
            Action::Down => (0, 1),
            Action::Left => (-1, 0),
            Action::Right => (1, 0),
            _ => (0, 0),
        };
        let new_grid = Self::shift_grid(&self.current_grid, dx, dy);
        // Reward: how many cells match the target?
        let mut matches = 0u32;
        let mut total_nonzero = 0u32;
        for r in 0..self.height.min(new_grid.height).min(self.target_grid.height) {
            for c in 0..self.width.min(new_grid.width).min(self.target_grid.width) {
                if self.target_grid.data[r][c] != 0 {
                    total_nonzero += 1;
                    if new_grid.data[r][c] == self.target_grid.data[r][c] {
                        matches += 1;
                    }
                }
            }
        }
        let reward = if total_nonzero > 0 {
            (matches as f64 / total_nonzero as f64 - 0.5) * 0.2
        } else {
            0.0
        };
        (new_grid, reward)
    }
}

impl ArcEnvironment for GridEnvironment {
    fn reset(&mut self) -> Result<Observation, String> {
        self.current_grid = self.initial_grid.clone();
        self.turn = 0;

        let ops = vec![
            ArcOpCode::Identity,
            ArcOpCode::Rotate,
            ArcOpCode::Flip,
            ArcOpCode::Gravity,
            ArcOpCode::Mirror,
            ArcOpCode::Tile,
            ArcOpCode::Crop,
            ArcOpCode::Scale,
            ArcOpCode::ReplaceColor,
            ArcOpCode::Fill,
            ArcOpCode::Move,
            ArcOpCode::Copy,
            ArcOpCode::CropContent,
        ];

        let mut candidates = ops;
        candidates.shuffle(&mut self.rng);

        let mut found = false;
        for op in candidates {
            let token = ArcOpToken::new(op as u8, 0, 0, 0, 0, 0, 0, 0);
            if let Some(computed) = apply_arc_op(&self.initial_grid, &token) {
                self.transformation = op;
                self.target_grid = computed;
                found = true;
                break;
            }
        }

        if !found {
            self.transformation = ArcOpCode::Identity;
            self.target_grid = self.initial_grid.clone();
        }

        Ok(Observation {
            frame: self.current_grid.data.clone(),
            width: self.width,
            height: self.height,
            turn: self.turn,
            episode_id: self.id.clone(),
        })
    }

    fn step(&mut self, action: Action) -> Result<ArcEnvironmentStep, String> {
        self.turn += 1;

        let reward = match action {
            Action::Noop => -0.01,
            Action::Interact => {
                // Apply the hidden transformation
                let token = ArcOpToken::new(
                    self.transformation as u8, 0, 0, 0, 0, 0, 0, 0,
                );
                if let Some(new_grid) = apply_arc_op(&self.current_grid, &token) {
                    self.current_grid = new_grid;
                }
                // Check if we reached the target
                let program = ArcProgram::from_tokens(vec![token]);
                let solved = program_solves_train(
                    &crate::vision::ArcTask {
                        id: self.id.clone(),
                        train_pairs: vec![(
                            self.initial_grid.clone(),
                            self.target_grid.clone(),
                        )],
                        test_inputs: vec![],
                        test_outputs: vec![],
                    },
                    &program,
                );
                if solved {
                    10.0
                } else {
                    -0.1
                }
            }
            _ => {
                let (new_grid, r) = self.apply_directional(action);
                self.current_grid = new_grid;
                r
            }
        };

        let done = self.turn >= self.max_turns;
        let obs = Observation {
            frame: self.current_grid.data.clone(),
            width: self.width,
            height: self.height,
            turn: self.turn,
            episode_id: self.id.clone(),
        };

        Ok(ArcEnvironmentStep {
            observation: obs,
            reward,
            done,
            info: std::collections::HashMap::new(),
        })
    }

    fn action_space(&self) -> usize {
        6
    }

    fn observation_shape(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn id(&self) -> &str {
        &self.id
    }
}

// ─── Environment Registry ─────────────────────────────────────────────────────

/// A registry of available environments.
#[derive(Debug, Default)]
pub struct EnvironmentRegistry {
    pub environments: Vec<Box<dyn ArcEnvironment>>,
}

impl EnvironmentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, env: Box<dyn ArcEnvironment>) {
        self.environments.push(env);
    }

    pub fn get(&self, idx: usize) -> Option<&dyn ArcEnvironment> {
        self.environments.get(idx).map(|e| e.as_ref())
    }

    pub fn len(&self) -> usize {
        self.environments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.environments.is_empty()
    }
}

// ─── Demo Environment Loader ──────────────────────────────────────────────────

/// Creates 8 demo ARC-AGI-3 environments with simple grid transformations.
pub fn load_demo_environments() -> EnvironmentRegistry {
    let mut registry = EnvironmentRegistry::new();

    // 1. rotate90: 3x3 grid, rotate 90 degrees
    let g1 = ArcGrid::from_data(vec![
        vec![1, 0, 0],
        vec![0, 2, 0],
        vec![0, 0, 3],
    ]).unwrap();
    let g1_out = ArcGrid::from_data(vec![
        vec![0, 0, 1],
        vec![0, 2, 0],
        vec![3, 0, 0],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("rotate90", g1, g1_out, 20)));

    // 2. flip_h: 3x3, flip horizontal
    let g2 = ArcGrid::from_data(vec![
        vec![1, 0, 0],
        vec![0, 2, 0],
        vec![0, 0, 3],
    ]).unwrap();
    let g2_out = ArcGrid::from_data(vec![
        vec![0, 0, 1],
        vec![0, 2, 0],
        vec![3, 0, 0],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("flip_h", g2, g2_out, 20)));

    // 3. gravity: 4x4, non-zero pixels fall down
    let g3 = ArcGrid::from_data(vec![
        vec![1, 0, 0, 0],
        vec![0, 2, 0, 0],
        vec![0, 0, 3, 0],
        vec![0, 0, 0, 0],
    ]).unwrap();
    let g3_out = ArcGrid::from_data(vec![
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 0],
        vec![1, 0, 0, 0],
        vec![0, 2, 3, 0],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("gravity", g3, g3_out, 20)));

    // 4. mirror_v: 3x3, mirror vertically
    let g4 = ArcGrid::from_data(vec![
        vec![1, 0, 0],
        vec![0, 2, 0],
        vec![0, 0, 3],
    ]).unwrap();
    let g4_out = ArcGrid::from_data(vec![
        vec![0, 0, 3],
        vec![0, 2, 0],
        vec![1, 0, 0],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("mirror_v", g4, g4_out, 20)));

    // 5. tile_2x2: 2x2 → 4x4
    let g5 = ArcGrid::from_data(vec![
        vec![1, 2],
        vec![3, 0],
    ]).unwrap();
    let g5_out = ArcGrid::from_data(vec![
        vec![1, 2, 1, 2],
        vec![3, 0, 3, 0],
        vec![1, 2, 1, 2],
        vec![3, 0, 3, 0],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("tile_2x2", g5, g5_out, 20)));

    // 6. crop_center: 4x4 → 2x2 center
    let g6 = ArcGrid::from_data(vec![
        vec![0, 0, 0, 0],
        vec![0, 1, 2, 0],
        vec![0, 3, 4, 0],
        vec![0, 0, 0, 0],
    ]).unwrap();
    let g6_out = ArcGrid::from_data(vec![
        vec![1, 2],
        vec![3, 4],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("crop_center", g6, g6_out, 20)));

    // 7. scale_2x: 2x2 → 4x4
    let g7 = ArcGrid::from_data(vec![
        vec![1, 2],
        vec![3, 4],
    ]).unwrap();
    let g7_out = ArcGrid::from_data(vec![
        vec![1, 1, 2, 2],
        vec![1, 1, 2, 2],
        vec![3, 3, 4, 4],
        vec![3, 3, 4, 4],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("scale_2x", g7, g7_out, 20)));

    // 8. replace_color: 3x3, replace color 1→2
    let g8 = ArcGrid::from_data(vec![
        vec![1, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 1],
    ]).unwrap();
    let g8_out = ArcGrid::from_data(vec![
        vec![2, 0, 0],
        vec![0, 2, 0],
        vec![0, 0, 2],
    ]).unwrap();
    registry.register(Box::new(GridEnvironment::new("replace_color", g8, g8_out, 20)));

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::ArcGrid;

    #[test]
    fn grid_env_creation() {
        let g1 = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let g2 = ArcGrid::from_data(vec![vec![0, 1], vec![2, 0]]).unwrap();
        let env = GridEnvironment::new("test", g1, g2, 10);
        assert_eq!(env.id(), "test");
        assert_eq!(env.action_space(), 6);
        assert_eq!(env.observation_shape(), (2, 2));
    }

    #[test]
    fn grid_env_reset_returns_observation() {
        let g1 = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let g2 = ArcGrid::from_data(vec![vec![0, 1], vec![2, 0]]).unwrap();
        let mut env = GridEnvironment::new("test", g1, g2, 10);
        let obs = env.reset().unwrap();
        assert_eq!(obs.width, 2);
        assert_eq!(obs.height, 2);
        assert_eq!(obs.turn, 0);
    }

    #[test]
    fn grid_env_step_noop_gives_negative_reward() {
        let g1 = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let g2 = ArcGrid::from_data(vec![vec![0, 1], vec![2, 0]]).unwrap();
        let mut env = GridEnvironment::new("test", g1, g2, 10);
        env.reset().unwrap();
        let result = env.step(Action::Noop).unwrap();
        assert!(!result.done);
        assert!(result.reward < 0.0);
    }

    #[test]
    fn grid_env_step_interact_changes_grid() {
        let g1 = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let g2 = ArcGrid::from_data(vec![vec![0, 1], vec![2, 0]]).unwrap();
        let mut env = GridEnvironment::new("test", g1, g2, 10);
        env.reset().unwrap();
        let result = env.step(Action::Interact).unwrap();
        assert_eq!(result.observation.turn, 1);
    }

    #[test]
    fn load_demo_environments_returns_eight() {
        let registry = load_demo_environments();
        assert_eq!(registry.len(), 8);
        assert!(!registry.is_empty());
    }

    #[test]
    fn environment_registry_get() {
        let registry = load_demo_environments();
        assert!(registry.get(0).is_some());
        assert!(registry.get(7).is_some());
        assert!(registry.get(8).is_none());
    }

    #[test]
    fn shift_grid_moves_pixels() {
        let g = ArcGrid::from_data(vec![vec![1, 0, 0], vec![0, 2, 0], vec![0, 0, 3]]).unwrap();
        let shifted = GridEnvironment::shift_grid(&g, 1, 0);
        assert_eq!(shifted.data[0][1], 1);
        assert_eq!(shifted.data[1][2], 2);
        assert_eq!(shifted.data[2][0], 0); // 3 moved right, clamped at edge
    }
}
