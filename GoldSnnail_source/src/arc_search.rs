//! ARC Compositional Solver — Brute-Force Search
//!
//! This module searches the compositional program space for ARC task solutions.
//! It enumerates all programs up to a given depth and returns the first one
//! that solves all training pairs.
//!
//! # Search Strategy
//!
//! 1. **Depth-first enumeration:** Programs are generated in depth-first order.
//! 2. **Pruning:** If a partial program fails on any training pair, deeper
//!    extensions are skipped immediately.
//! 3. **Color map first:** Color map inference is attempted before operation
//!    search, as it often solves tasks alone.
//! 4. **Early exit:** The first solving program is returned; no exhaustive
//!    enumeration beyond the first solution.
//!
//! # Complexity
//!
//! | Depth | Candidates (8 ops) | With Pruning |
//! |-------|-------------------|--------------|
//! | 1     | 8                 | ~8           |
//! | 2     | 64                | ~20          |
//! | 3     | 512               | ~50-100      |
//!
//! Pruning typically reduces the search space by 5-10x because most random
//! programs fail on the first training pair.

use crate::arc_apply::{apply_program, program_solves_train};
use crate::arc_program::{ArcOpCode, ArcOpToken, ArcProgram};
use crate::vision::{ArcGrid, ArcTask};
use std::collections::HashSet;

// ─── Search Configuration ────────────────────────────────────────────────────

/// Configuration for the ARC program search.
#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    /// Maximum program length (depth) to search.
    pub max_depth: usize,
    /// Whether to attempt color map inference before operation search.
    pub try_color_map: bool,
    /// Whether to allow Identity in programs longer than 1.
    pub allow_identity_in_composite: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            try_color_map: true,
            allow_identity_in_composite: false,
        }
    }
}

// ─── Search Result ───────────────────────────────────────────────────────────

/// Result of a program search for an ARC task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// The solving program, if found.
    pub program: Option<ArcProgram>,
    /// Number of candidates evaluated before finding a solution.
    pub candidates_evaluated: usize,
    /// Whether the search hit the candidate limit without finding a solution.
    pub timeout: bool,
}

impl SearchResult {
    /// Creates a successful result.
    pub fn found(program: ArcProgram, candidates: usize) -> Self {
        Self {
            program: Some(program),
            candidates_evaluated: candidates,
            timeout: false,
        }
    }

    /// Creates a failed result (no solution found).
    pub fn not_found(candidates: usize, timeout: bool) -> Self {
        Self {
            program: None,
            candidates_evaluated: candidates,
            timeout,
        }
    }

    /// Returns true if a solving program was found.
    pub fn is_some(&self) -> bool {
        self.program.is_some()
    }
}

// ─── Grid Signature & Pre-Filter ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct GridSignature {
    dims: (usize, usize),
    colors: Vec<u8>,
    color_counts: Vec<(u8, usize)>,
    object_count: usize,
}

impl GridSignature {
    fn from_grid(grid: &ArcGrid) -> Self {
        let mut counts = [0usize; 10];
        for row in &grid.data {
            for &c in row {
                counts[c as usize] += 1;
            }
        }
        let colors: Vec<u8> = counts.iter().enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(c, _)| c as u8)
            .collect();
        let color_counts: Vec<(u8, usize)> = counts.iter().enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(c, &count)| (c as u8, count))
            .collect();
        
        // Count 4-connected components of non-zero pixels
        let object_count = count_components(grid);
        
        Self {
            dims: (grid.width, grid.height),
            colors,
            color_counts,
            object_count,
        }
    }
}

fn count_components(grid: &ArcGrid) -> usize {
    if grid.height == 0 || grid.width == 0 {
        return 0;
    }
    let mut visited = vec![vec![false; grid.width]; grid.height];
    let mut count = 0;
    for r in 0..grid.height {
        for c in 0..grid.width {
            if visited[r][c] || grid.data[r][c] == 0 {
                continue;
            }
            count += 1;
            let mut stack = vec![(r, c)];
            while let Some((cr, cc)) = stack.pop() {
                if visited[cr][cc] { continue; }
                visited[cr][cc] = true;
                if cr > 0 && !visited[cr-1][cc] && grid.data[cr-1][cc] == grid.data[r][c] {
                    stack.push((cr-1, cc));
                }
                if cr + 1 < grid.height && !visited[cr+1][cc] && grid.data[cr+1][cc] == grid.data[r][c] {
                    stack.push((cr+1, cc));
                }
                if cc > 0 && !visited[cr][cc-1] && grid.data[cr][cc-1] == grid.data[r][c] {
                    stack.push((cr, cc-1));
                }
                if cc + 1 < grid.width && !visited[cr][cc+1] && grid.data[cr][cc+1] == grid.data[r][c] {
                    stack.push((cr, cc+1));
                }
            }
        }
    }
    count
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plausibility {
    Plausible,
    Implausible,
    Unknown,
}

fn op_is_plausible(op_code: u8, input_sig: &GridSignature, output_sig: &GridSignature) -> Plausibility {
    match op_code {
        0 => Plausibility::Plausible, // Identity: always plausible
        1 | 2 => { // Rotate90/270: dims must swap
            if input_sig.dims.0 == output_sig.dims.1 && input_sig.dims.1 == output_sig.dims.0 {
                Plausibility::Plausible
            } else {
                Plausibility::Implausible
            }
        }
        3 | 4 | 5 | 6 | 7 => { // Rotate180, Flip, Gravity, Mirror: dims stay same
            if input_sig.dims == output_sig.dims {
                Plausibility::Plausible
            } else {
                Plausibility::Implausible
            }
        }
        8 => { // Tile: output must be >= input in both dims
            if output_sig.dims.0 >= input_sig.dims.0 && output_sig.dims.1 >= input_sig.dims.1 {
                Plausibility::Plausible
            } else {
                Plausibility::Implausible
            }
        }
        9 => { // Crop: output must be <= input in both dims
            if output_sig.dims.0 <= input_sig.dims.0 && output_sig.dims.1 <= input_sig.dims.1 {
                Plausibility::Plausible
            } else {
                Plausibility::Implausible
            }
        }
        10 => { // ReplaceColor: palette must change
            if input_sig.colors != output_sig.colors {
                Plausibility::Plausible
            } else {
                Plausibility::Implausible
            }
        }
        11 => { // Scale: output must be >= input in both dims
            if output_sig.dims.0 >= input_sig.dims.0 && output_sig.dims.1 >= input_sig.dims.1 {
                Plausibility::Plausible
            } else {
                Plausibility::Implausible
            }
        }
        _ => Plausibility::Unknown,
    }
}

// ─── Candidate Generation ────────────────────────────────────────────────────

/// Returns (width, height) of the first available grid in a task.
fn grid_dims(task: &ArcTask) -> (usize, usize) {
    if let Some((input, _)) = task.train_pairs.first() {
        (input.width, input.height)
    } else if let Some(input) = task.test_inputs.first() {
        (input.width, input.height)
    } else {
        (10, 10)
    }
}

// ─── Parameter Inference ─────────────────────────────────────────────────────

/// Tries to infer Tile parameters (n, m) from training pairs.
fn infer_tile_params(task: &ArcTask) -> Option<(u8, u8)> {
    let (input, output) = task.train_pairs.first()?;
    if input.height == 0 || input.width == 0 {
        return None;
    }
    if output.height % input.height != 0 || output.width % input.width != 0 {
        return None;
    }
    let n = (output.width / input.width) as u8;
    let m = (output.height / input.height) as u8;
    if n >= 1 && n <= 4 && m >= 1 && m <= 4 {
        Some((n, m))
    } else {
        None
    }
}

/// Tries to infer Crop parameters (x, y, w, h) from training pairs.
fn infer_crop_params(task: &ArcTask) -> Option<(u8, u8, u8, u8)> {
    let (input, output) = task.train_pairs.first()?;
    // Find bbox of non-zero pixels in output
    let mut min_r = output.height;
    let mut max_r = 0;
    let mut min_c = output.width;
    let mut max_c = 0;
    for r in 0..output.height {
        for c in 0..output.width {
            if output.data[r][c] != 0 {
                min_r = min_r.min(r);
                max_r = max_r.max(r);
                min_c = min_c.min(c);
                max_c = max_c.max(c);
            }
        }
    }
    if min_r > max_r || min_c > max_c {
        return None;
    }
    let w = (max_c - min_c + 1) as u8;
    let h = (max_r - min_r + 1) as u8;
    // Verify the input contains this region at the same position
    if min_r < input.height && min_c < input.width {
        Some((min_c as u8, min_r as u8, w, h))
    } else {
        None
    }
}

/// Tries to infer ReplaceColor parameters (src, dst) from training pairs.
fn infer_replace_color_params(task: &ArcTask) -> Option<(u8, u8)> {
    let (input, output) = task.train_pairs.first()?;
    let mut mapping: Option<(u8, u8)> = None;
    for (i_cell, o_cell) in input.data.iter().flatten().zip(output.data.iter().flatten()) {
        if *i_cell != *o_cell && *i_cell != 0 && *o_cell != 0 {
            if mapping.is_some() && mapping.unwrap() != (*i_cell, *o_cell) {
                return None; // Inconsistent mapping
            }
            mapping = Some((*i_cell, *o_cell));
        }
    }
    mapping
}

/// Tries to infer Scale factor from training pairs.
fn infer_scale_params(task: &ArcTask) -> Option<u8> {
    let (input, output) = task.train_pairs.first()?;
    if input.height == 0 || input.width == 0 {
        return None;
    }
    if output.height % input.height != 0 || output.width % input.width != 0 {
        return None;
    }
    let factor_h = output.height / input.height;
    let factor_w = output.width / input.width;
    if factor_h == factor_w && factor_h >= 1 && factor_h <= 3 {
        Some(factor_h as u8)
    } else {
        None
    }
}

/// Generates all valid parameterized tokens for a given op code.
/// For new ops (8-11), tries inferred parameters first, then falls back to enumeration.
fn candidates_for_op(op_code: u8, task: &ArcTask) -> Vec<ArcOpToken> {
    let (w, h) = grid_dims(task);
    match op_code {
        // Old ops: zero-param (preserves existing search behavior)
        0..=7 => vec![ArcOpToken::new(op_code, 0, 0, 0, 0, 0, 0, 0)],
        // Tile: inferred params first, then enumerated
        8 => {
            let mut tokens = Vec::new();
            // Try inferred params first
            if let Some((n, m)) = infer_tile_params(task) {
                tokens.push(ArcOpToken::new(8, n, m, 0, 0, 0, 0, 0));
            }
            // Then enumerate small range
            for n in 1..=3 {
                for m in 1..=3 {
                    if h * m as usize <= 30 && w * n as usize <= 30 {
                        let token = ArcOpToken::new(8, n, m, 0, 0, 0, 0, 0);
                        if !tokens.contains(&token) {
                            tokens.push(token);
                        }
                    }
                }
            }
            tokens
        }
        // Crop: inferred params first, then limited enumeration
        9 => {
            let mut tokens = Vec::new();
            // Try inferred params first
            if let Some((x, y, w, h)) = infer_crop_params(task) {
                tokens.push(ArcOpToken::new(9, x, y, w, h, 0, 0, 0));
            }
            // Then enumerate a few heuristic crops (center, top-left, etc.)
            let mut heuristic_crops = Vec::new();
            if w > 2 && h > 2 {
                heuristic_crops.push((1, 1, (w - 2) as u8, (h - 2) as u8)); // center crop
            }
            if w >= 2 && h >= 2 {
                heuristic_crops.push((0, 0, (w / 2) as u8, (h / 2) as u8)); // top-left half
                heuristic_crops.push((w / 2, h / 2, (w / 2) as u8, (h / 2) as u8)); // bottom-right half
            }
            for (x, y, cw, ch) in heuristic_crops {
                if cw > 0 && ch > 0 && x + cw as usize <= w && y + ch as usize <= h {
                    let token = ArcOpToken::new(9, x as u8, y as u8, cw, ch, 0, 0, 0);
                    if !tokens.contains(&token) {
                        tokens.push(token);
                    }
                }
            }
            tokens
        }
        // ReplaceColor: inferred params first, then limited enumeration
        10 => {
            let mut tokens = Vec::new();
            // Try inferred params first
            if let Some((src, dst)) = infer_replace_color_params(task) {
                tokens.push(ArcOpToken::new(10, src, dst, 0, 0, 0, 0, 0));
            }
            // Then enumerate only the most common color replacements
            if let Some((input, _)) = task.train_pairs.first() {
                let mut color_counts = [0usize; 10];
                for row in &input.data {
                    for &c in row {
                        if c != 0 {
                            color_counts[c as usize] += 1;
                        }
                    }
                }
                let mut colors: Vec<_> = color_counts.iter().enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .map(|(c, _)| c as u8)
                    .collect();
                colors.sort_by(|a, b| color_counts[*b as usize].cmp(&color_counts[*a as usize]));
                // Try top 3 colors → all other colors
                for &src in colors.iter().take(3) {
                    for dst in 0..=9u8 {
                        if src != dst {
                            let token = ArcOpToken::new(10, src, dst, 0, 0, 0, 0, 0);
                            if !tokens.contains(&token) {
                                tokens.push(token);
                            }
                        }
                    }
                }
            }
            tokens
        }
        // Scale: inferred params first, then factor 2 or 3
        11 => {
            let mut tokens = Vec::new();
            // Try inferred params first
            if let Some(factor) = infer_scale_params(task) {
                tokens.push(ArcOpToken::new(11, factor, 0, 0, 0, 0, 0, 0));
            }
            // Then try factor 2 or 3 if it fits
            for factor in [2u8, 3] {
                if h * factor as usize <= 30 && w * factor as usize <= 30 {
                    let token = ArcOpToken::new(11, factor, 0, 0, 0, 0, 0, 0);
                    if !tokens.contains(&token) {
                        tokens.push(token);
                    }
                }
            }
            tokens
        }
        // CropContent: always one candidate (no params)
        12 => {
            vec![ArcOpToken::new(12, 0, 0, 0, 0, 0, 0, 0)]
        }
        _ => vec![],
    }
}

// ─── Core Search ─────────────────────────────────────────────────────────────

/// Searches for a program that solves all training pairs of an ARC task.
///
/// Uses depth-first enumeration with pruning. Returns the first solving
/// program found, or `None` if no solution exists within the search budget.
///
/// # Arguments
///
/// * `task` — The ARC task to solve
/// * `config` — Search configuration
///
/// # Returns
///
/// A `SearchResult` containing the solving program (if found) and statistics.
pub fn search_program(task: &ArcTask, config: SearchConfig) -> SearchResult {
    let max_candidates = 100_000; // Safety bound
    
    // Pre-compute grid signatures from first train pair
    let (input_sig, output_sig) = if let Some((input, output)) = task.train_pairs.first() {
        (GridSignature::from_grid(input), GridSignature::from_grid(output))
    } else {
        return SearchResult::not_found(0, false);
    };
    
    // Tiered budget allocation
    let depth1_budget = (max_candidates * 6) / 10; // 60%
    let depth2_budget = (max_candidates * 3) / 10; // 30%
    let _depth3_budget = max_candidates - depth1_budget - depth2_budget; // 10%
    
    let mut candidates = 0;
    
    if config.max_depth == 0 {
        return SearchResult::not_found(0, false);
    }

    // ─── Try color map alone ──────────────────────────────────────────────────
    if config.try_color_map {
        if let Some(mapping) = crate::vision::dsl_solver::infer_color_map(task) {
            let program = ArcProgram::from_tokens(vec![
                ArcOpToken::new(0, 0, 0, 0, 0, 0, 0, 0) // Identity with color map
            ]);
            candidates += 1;
            if program_solves_train_with_color_map(task, &program, &mapping) {
                return SearchResult::found(program, candidates);
            }
        }
    }

    // ─── Try color map + single operation ────────────────────────────────────
    if config.try_color_map {
        if let Some(mapping) = crate::vision::dsl_solver::infer_color_map(task) {
            for op_code in 1..12 {
                if op_is_plausible(op_code, &input_sig, &output_sig) != Plausibility::Implausible {
                    let tokens = candidates_for_op(op_code, task);
                    for token in tokens {
                        let program = ArcProgram::from_tokens(vec![token]);
                        candidates += 1;
                        if program_solves_train_with_color_map(task, &program, &mapping) {
                            return SearchResult::found(program, candidates);
                        }
                        if candidates >= depth1_budget {
                            break;
                        }
                    }
                }
                if candidates >= depth1_budget {
                    break;
                }
            }
        }
    }

    // ─── Try single operations (Depth 1) ────────────────────────────────────
    for op_code in 0..12 {
        if op_code == 0 && !config.allow_identity_in_composite {
            continue;
        }
        if op_is_plausible(op_code, &input_sig, &output_sig) == Plausibility::Implausible {
            continue;
        }
        let tokens = candidates_for_op(op_code, task);
        for token in tokens {
            let program = ArcProgram::from_tokens(vec![token]);
            candidates += 1;
            if program_solves_train(task, &program) {
                return SearchResult::found(program, candidates);
            }
            if candidates >= depth1_budget {
                break;
            }
        }
        if candidates >= depth1_budget {
            break;
        }
    }
    
    // ─── Try depth 2 and beyond ───────────────────────────────────────────────
    if config.max_depth >= 2 && candidates < depth1_budget + depth2_budget {
        let mut partial = Vec::new();
        if search_depth_first(task, &mut partial, config.max_depth, &mut candidates, max_candidates, depth1_budget + depth2_budget, &input_sig, &output_sig) {
            let program = ArcProgram::from_tokens(partial);
            return SearchResult::found(program, candidates);
        }
    }

    SearchResult::not_found(candidates, candidates >= max_candidates)
}

/// Depth-first search with pruning.
///
/// Extends `partial` program in-place. Returns `true` if a solution is found.
fn search_depth_first(
    task: &ArcTask,
    partial: &mut Vec<ArcOpToken>,
    max_depth: usize,
    candidates: &mut usize,
    max_candidates: usize,
    budget: usize,
    input_sig: &GridSignature,
    output_sig: &GridSignature,
) -> bool {
    if partial.len() >= max_depth {
        return false;
    }

    for op_code in 0..12 {
        if op_code == 0 {
            continue; // Skip identity in composite programs
        }
        
        // Plausibility check for first op
        if partial.is_empty() && op_is_plausible(op_code, input_sig, output_sig) == Plausibility::Implausible {
            continue;
        }
        
        let tokens = candidates_for_op(op_code, task);
        for token in tokens {
            partial.push(token);
            *candidates += 1;

            // Fast-fail pruning: check only first train pair first
            if partial_solves_train_prefix(task, partial) {
                // Check full program
                let program = ArcProgram::from_tokens(partial.clone());
                if program_solves_train(task, &program) {
                    return true;
                }
                // Recurse deeper
                if search_depth_first(task, partial, max_depth, candidates, max_candidates, budget, input_sig, output_sig) {
                    return true;
                }
            }

            partial.pop();

            if *candidates >= max_candidates || *candidates >= budget {
                return false;
            }
        }
    }

    false
}

/// Checks if applying the partial program to all training inputs still
/// produces valid grids (not None). This is the pruning condition.
///
/// A partial program is "viable" if it hasn't failed on any training pair yet.
fn partial_solves_train_prefix(task: &ArcTask, partial: &[ArcOpToken]) -> bool {
    let program = ArcProgram {
        tokens: partial.to_vec(),
    };

    for (input, _expected) in &task.train_pairs {
        if apply_program(input, &program).is_none() {
            return false;
        }
    }

    true
}

/// Checks if a program solves all training pairs, with optional color map.
fn program_solves_train_with_color_map(
    task: &ArcTask,
    program: &ArcProgram,
    mapping: &[Option<u8>],
) -> bool {
    for (input, expected) in &task.train_pairs {
        let colored = crate::vision::dsl_solver::apply_color_map(input, mapping);
        if let Some(output) = apply_program(&colored, program) {
            if output != *expected {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

// ─── Convenience Wrappers ────────────────────────────────────────────────────

/// Searches for a solving program using default configuration.
///
/// Equivalent to `search_program(task, SearchConfig::default())`.
pub fn find_program(task: &ArcTask) -> Option<ArcProgram> {
    search_program(task, SearchConfig::default()).program
}

/// Searches for a solving program with a specific maximum depth.
pub fn find_program_with_depth(task: &ArcTask, max_depth: usize) -> Option<ArcProgram> {
    let mut config = SearchConfig::default();
    config.max_depth = max_depth;
    search_program(task, config).program
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_identity_task() {
        let task = ArcTask {
            id: "test".to_string(),
            train_pairs: vec![(
                ArcGrid::from_data(vec![vec![1, 0], vec![0, 1]]).unwrap(),
                ArcGrid::from_data(vec![vec![1, 0], vec![0, 1]]).unwrap(),
            )],
            test_inputs: vec![],
            test_outputs: vec![],
        };

        let result = search_program(&task, SearchConfig::default());
        assert!(result.is_some());
        assert_eq!(result.program.unwrap().len(), 1);
    }

    #[test]
    fn test_search_rotate90_task() {
        let task = ArcTask {
            id: "test".to_string(),
            train_pairs: vec![(
                ArcGrid::from_data(vec![vec![1, 0], vec![0, 0]]).unwrap(),
                ArcGrid::from_data(vec![vec![0, 1], vec![0, 0]]).unwrap(),
            )],
            test_inputs: vec![],
            test_outputs: vec![],
        };

        let result = search_program(&task, SearchConfig { max_depth: 2, ..Default::default() });
        assert!(result.is_some());
        assert_eq!(result.candidates_evaluated, 2); // Identity + Rotate90
    }

    #[test]
    fn test_search_no_solution() {
        let task = ArcTask {
            id: "test".to_string(),
            train_pairs: vec![(
                ArcGrid::from_data(vec![vec![1, 0], vec![0, 0]]).unwrap(),
                ArcGrid::from_data(vec![vec![0, 0], vec![0, 1]]).unwrap(),
            )],
            test_inputs: vec![],
            test_outputs: vec![],
        };

        let result = search_program(&task, SearchConfig { max_depth: 1, ..Default::default() });
        assert!(!result.is_some());
    }

    #[test]
    fn test_search_depth_limit() {
        let task = ArcTask {
            id: "test".to_string(),
            train_pairs: vec![(
                ArcGrid::from_data(vec![vec![1, 0], vec![0, 0]]).unwrap(),
                ArcGrid::from_data(vec![vec![0, 1], vec![0, 0]]).unwrap(),
            )],
            test_inputs: vec![],
            test_outputs: vec![],
        };

        // With max_depth=0, no search should happen
        let result = search_program(&task, SearchConfig { max_depth: 0, ..Default::default() });
        assert!(!result.is_some());
    }
}
