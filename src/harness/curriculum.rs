//! Curriculum Generator — Task Sequencing & Difficulty Progression
//!
//! Automatically sequences training tasks from simple to complex,
//! mixing ARC tasks with other benchmarks to prevent task-specific overfitting.

use crate::vision::arc_loader::{ArcGrid, ArcTask, ArcDataset};

/// A wrapped curriculum task with metadata.
#[derive(Debug, Clone)]
pub struct CurriculumTask {
    pub id: String,
    pub train_pairs: Vec<(ArcGrid, ArcGrid)>,
    pub difficulty: f64,
    pub width: usize,
    pub height: usize,
    pub color_count: usize,
}

impl CurriculumTask {
    /// Creates a curriculum task from an ARC task with estimated difficulty.
    pub fn from_arc_task(task: ArcTask) -> Self {
        let width = task.train_pairs.first().map(|(i, _)| i.width).unwrap_or(0);
        let height = task.train_pairs.first().map(|(i, _)| i.height).unwrap_or(0);
        let color_count = task.train_pairs.iter()
            .flat_map(|(i, o)| i.unique_colors().into_iter().chain(o.unique_colors()))
            .collect::<std::collections::HashSet<_>>()
            .len();

        let difficulty = Self::estimate_difficulty(width, height, color_count, task.train_pairs.len());
        Self {
            id: task.id,
            train_pairs: task.train_pairs,
            difficulty,
            width,
            height,
            color_count,
        }
    }

    /// Estimates task difficulty based on grid size, color count, and pair count.
    fn estimate_difficulty(width: usize, height: usize, colors: usize, pairs: usize) -> f64 {
        let size_factor = (width * height) as f64 / 100.0;
        let color_factor = colors as f64 / 5.0;
        let pair_factor = (pairs as f64).ln();
        (size_factor + color_factor + pair_factor).max(0.1)
    }
}

/// Curriculum stage containing tasks of similar difficulty.
#[derive(Debug, Clone, Default)]
pub struct CurriculumStage {
    pub tasks: Vec<CurriculumTask>,
    pub mastered: bool,
    pub mastery_threshold: f64,
}

impl CurriculumStage {
    pub fn new(tasks: Vec<CurriculumTask>, mastery_threshold: f64) -> Self {
        Self {
            tasks,
            mastered: false,
            mastery_threshold,
        }
    }
}

/// Curriculum generator for autonomous task sequencing.
#[derive(Debug, Clone, Default)]
pub struct Curriculum {
    pub stages: Vec<CurriculumStage>,
    pub current_stage: usize,
    pub current_task_idx: usize,
    pub success_threshold: f64,
    pub total_tasks_presented: u64,
}

impl Curriculum {
    /// Creates an empty curriculum.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a curriculum from an ARC dataset, sorted by estimated difficulty.
    pub fn from_arc_dataset(dataset: &ArcDataset, stages: usize) -> Self {
        let mut tasks: Vec<CurriculumTask> = dataset.tasks
            .iter()
            .map(|t| CurriculumTask::from_arc_task(t.clone()))
            .collect();

        // Sort by difficulty
        tasks.sort_by(|a, b| a.difficulty.partial_cmp(&b.difficulty).unwrap());

        // Split into stages
        let per_stage = (tasks.len() / stages).max(1);
        let mut curriculum_stages = Vec::new();
        for chunk in tasks.chunks(per_stage) {
            curriculum_stages.push(CurriculumStage::new(chunk.to_vec(), 0.8));
        }

        Self {
            stages: curriculum_stages,
            current_stage: 0,
            current_task_idx: 0,
            success_threshold: 0.8,
            total_tasks_presented: 0,
        }
    }

    /// Returns the next task to train on, cycling within the current stage.
    pub fn next_task(&mut self) -> Option<&CurriculumTask> {
        if self.stages.is_empty() {
            return None;
        }

        let stage = &self.stages[self.current_stage];
        if stage.tasks.is_empty() {
            return None;
        }

        let task = &stage.tasks[self.current_task_idx];
        self.current_task_idx = (self.current_task_idx + 1) % stage.tasks.len();
        self.total_tasks_presented += 1;
        Some(task)
    }

    /// Advances to the next curriculum stage if the current one is mastered.
    pub fn advance(&mut self, performance: f64) -> bool {
        if self.stages.is_empty() {
            return false;
        }

        let stage = &mut self.stages[self.current_stage];
        if performance >= stage.mastery_threshold && !stage.mastered {
            stage.mastered = true;
            if self.current_stage + 1 < self.stages.len() {
                self.current_stage += 1;
                self.current_task_idx = 0;
                return true;
            }
        }
        false
    }

    /// Returns the number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Returns the current stage index.
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Returns progress through the curriculum (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.stages.is_empty() {
            return 1.0;
        }
        let stage_progress = self.current_stage as f64 / self.stages.len() as f64;
        let task_progress = if self.stages[self.current_stage].tasks.is_empty() {
            0.0
        } else {
            self.current_task_idx as f64 / self.stages[self.current_stage].tasks.len() as f64
        };
        (stage_progress + task_progress / self.stages.len() as f64).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::arc_loader::ArcTask;

    fn make_task(id: &str, width: usize, height: usize) -> ArcTask {
        let mut task = ArcTask::new(id);
        task.train_pairs.push((
            ArcGrid::from_data(vec![vec![0; width]; height]).unwrap(),
            ArcGrid::from_data(vec![vec![0; width]; height]).unwrap(),
        ));
        task
    }

    #[test]
    fn curriculum_task_difficulty() {
        let task = CurriculumTask::from_arc_task(make_task("t1", 10, 10));
        assert!(task.difficulty > 0.0);
    }

    #[test]
    fn curriculum_next_task_cycles() {
        let dataset = ArcDataset {
            tasks: vec![make_task("t1", 5, 5), make_task("t2", 5, 5)],
        };
        let mut curriculum = Curriculum::from_arc_dataset(&dataset, 1);
        let t1 = curriculum.next_task().is_some();
        let t2 = curriculum.next_task().is_some();
        assert!(t1);
        assert!(t2);
    }

    #[test]
    fn curriculum_advance_stage() {
        let dataset = ArcDataset {
            tasks: vec![make_task("t1", 5, 5), make_task("t2", 10, 10)],
        };
        let mut curriculum = Curriculum::from_arc_dataset(&dataset, 2);
        assert_eq!(curriculum.current_stage(), 0);
        curriculum.advance(0.9);
        assert_eq!(curriculum.current_stage(), 1);
    }
}
