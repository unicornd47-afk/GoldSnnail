//! ARC Compositional Solver — Apply Engine
//!
//! This module applies `ArcOpToken` sequences to `ArcGrid`s. It is the
//! execution engine for the compositional solver: pure functional grid
//! transformations, no ML, no gradients, no embedding.
//!
//! # Design Principles
//!
//! - **Deterministic:** Same input + same program = same output
//! - **No allocation in hot path:** All operations reuse grid buffers where possible
//! - **Elastic failure:** Operations that cannot be applied (e.g., out-of-bounds)
//!   return `None` rather than panicking
//! - **DOD-compliant:** Works with flat `Vec<Vec<u8>>` grids

use crate::arc_program::{ArcOpCode, ArcOpToken, ArcProgram};
use crate::vision::ArcGrid;

// ─── Single Operation Apply ──────────────────────────────────────────────────

/// Applies a single `ArcOpToken` to an `ArcGrid`.
///
/// Returns `None` if the operation cannot be applied (e.g., out-of-bounds
/// parameters, invalid operation code).
///
/// # Elastic Failure
///
/// No panics. Invalid operations or parameters return `None`. The caller
/// decides how to handle failure (skip, fallback, abort).
pub fn apply_arc_op(grid: &ArcGrid, token: &ArcOpToken) -> Option<ArcGrid> {
    let op = token.op()?;
    match op {
        ArcOpCode::Identity => apply_identity(grid),
        ArcOpCode::Rotate => apply_rotate(grid, token.param(0)),
        ArcOpCode::Flip => apply_flip(grid, token.param(0)),
        ArcOpCode::Move => apply_move(grid, token.param(0), token.param(1)),
        ArcOpCode::Fill => apply_fill(
            grid,
            token.param(0),
            token.param(1),
            token.param(2),
            token.param(3),
            token.param(4),
        ),
        ArcOpCode::Copy => apply_copy(
            grid,
            token.param(0),
            token.param(1),
            token.param(2),
            token.param(3),
            token.param(4),
            token.param(5),
        ),
        ArcOpCode::Gravity => apply_gravity(grid, token.param(0)),
        ArcOpCode::Mirror => apply_mirror(grid, token.param(0), token.param(1)),
        ArcOpCode::Tile => apply_tile(grid, token.param(0), token.param(1)),
        ArcOpCode::Crop => apply_crop(grid, token.param(0), token.param(1), token.param(2), token.param(3)),
        ArcOpCode::ReplaceColor => apply_replace_color(grid, token.param(0), token.param(1)),
        ArcOpCode::Scale => apply_scale(grid, token.param(0)),
        ArcOpCode::CropContent => apply_crop_content(grid),
    }
}

// ─── Program Apply ───────────────────────────────────────────────────────────

/// Applies a complete `ArcProgram` to an `ArcGrid`.
///
/// Operations are applied sequentially. If any operation fails, the entire
/// program returns `None`.
///
/// # Example
///
/// ```
/// use goldworm::arc_program::{ArcOpToken, ArcProgram};
/// use goldworm::arc_apply::apply_program;
/// use goldworm::vision::ArcGrid;
///
/// let grid = ArcGrid::from_data(vec![vec![0, 1], vec![0, 1]]).unwrap();
/// let program = ArcProgram::from_tokens(vec![
///     ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0), // Rotate 90°
/// ]);
/// let result = apply_program(&grid, &program);
/// assert!(result.is_some());
/// ```
pub fn apply_program(grid: &ArcGrid, program: &ArcProgram) -> Option<ArcGrid> {
    let mut current = grid.clone();
    for token in &program.tokens {
        current = apply_arc_op(&current, token)?;
    }
    Some(current)
}

/// Checks if a program solves all training pairs for an ARC task.
///
/// Returns `true` if applying the program to every training input produces
/// the expected training output.
pub fn program_solves_train(task: &crate::vision::ArcTask, program: &ArcProgram) -> bool {
    for (input, expected) in &task.train_pairs {
        if let Some(output) = apply_program(input, program) {
            if output != *expected {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

// ─── Primitive Operations ────────────────────────────────────────────────────

fn apply_identity(grid: &ArcGrid) -> Option<ArcGrid> {
    Some(grid.clone())
}

fn apply_rotate(grid: &ArcGrid, angle: u8) -> Option<ArcGrid> {
    match angle {
        0 => rotate90(grid),
        1 => Some(rotate180(grid)),
        2 => rotate270(grid),
        _ => None,
    }
}

fn apply_flip(grid: &ArcGrid, axis: u8) -> Option<ArcGrid> {
    match axis {
        0 => Some(flip_horizontal(grid)),
        1 => Some(flip_vertical(grid)),
        _ => None,
    }
}

fn apply_move(grid: &ArcGrid, dx: u8, dy: u8) -> Option<ArcGrid> {
    let dx = dx as i32;
    let dy = dy as i32;
    let new_width = grid.width;
    let new_height = grid.height;
    let mut data = vec![vec![0u8; new_width]; new_height];

    for (r, row) in grid.data.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            if val == 0 {
                continue; // Skip background pixels
            }
            let new_r = r as i32 + dy;
            let new_c = c as i32 + dx;
            if new_r >= 0 && new_r < new_height as i32 && new_c >= 0 && new_c < new_width as i32 {
                data[new_r as usize][new_c as usize] = val;
            }
            // Out-of-bounds pixels are clipped (dropped) instead of failing.
        }
    }

    Some(ArcGrid::from_data(data).unwrap())
}

fn apply_fill(
    grid: &ArcGrid,
    color: u8,
    x: u8,
    y: u8,
    w: u8,
    h: u8,
) -> Option<ArcGrid> {
    let x = x as usize;
    let y = y as usize;
    let w = w as usize;
    let h = h as usize;

    if x >= grid.width || y >= grid.height {
        return None;
    }

    let mut data = grid.data.clone();
    for r in y..(y + h).min(grid.height) {
        for c in x..(x + w).min(grid.width) {
            data[r][c] = color;
        }
    }

    Some(ArcGrid::from_data(data).unwrap())
}

fn apply_copy(
    grid: &ArcGrid,
    src_x: u8,
    src_y: u8,
    dst_x: u8,
    dst_y: u8,
    w: u8,
    h: u8,
) -> Option<ArcGrid> {
    let src_x = src_x as usize;
    let src_y = src_y as usize;
    let dst_x = dst_x as usize;
    let dst_y = dst_y as usize;
    let w = w as usize;
    let h = h as usize;

    if src_x + w > grid.width || src_y + h > grid.height {
        return None;
    }
    if dst_x + w > grid.width || dst_y + h > grid.height {
        return None;
    }

    let mut data = grid.data.clone();
    for dr in 0..h {
        for dc in 0..w {
            data[dst_y + dr][dst_x + dc] = grid.data[src_y + dr][src_x + dc];
        }
    }

    Some(ArcGrid::from_data(data).unwrap())
}

fn apply_gravity(grid: &ArcGrid, direction: u8) -> Option<ArcGrid> {
    let mut data = grid.data.clone();

    match direction {
        0 => {
            // Gravity down: pixels fall to the bottom
            for c in 0..grid.width {
                let mut write_r = grid.height;
                for r in (0..grid.height).rev() {
                    if grid.data[r][c] != 0 {
                        write_r -= 1;
                        data[write_r][c] = grid.data[r][c];
                    }
                }
                for r in 0..write_r {
                    data[r][c] = 0;
                }
            }
        }
        1 => {
            // Gravity up: pixels fall to the top
            for c in 0..grid.width {
                let mut write_r = 0;
                for r in 0..grid.height {
                    if grid.data[r][c] != 0 {
                        data[write_r][c] = grid.data[r][c];
                        write_r += 1;
                    }
                }
                for r in write_r..grid.height {
                    data[r][c] = 0;
                }
            }
        }
        2 => {
            // Gravity left: pixels fall to the left
            for r in 0..grid.height {
                let mut write_c = 0;
                for c in 0..grid.width {
                    if grid.data[r][c] != 0 {
                        data[r][write_c] = grid.data[r][c];
                        write_c += 1;
                    }
                }
                for c in write_c..grid.width {
                    data[r][c] = 0;
                }
            }
        }
        3 => {
            // Gravity right: pixels fall to the right
            for r in 0..grid.height {
                let mut write_c = grid.width;
                for c in (0..grid.width).rev() {
                    if grid.data[r][c] != 0 {
                        write_c -= 1;
                        data[r][write_c] = grid.data[r][c];
                    }
                }
                for c in 0..write_c {
                    data[r][c] = 0;
                }
            }
        }
        _ => return None,
    }

    Some(ArcGrid::from_data(data).unwrap())
}

fn apply_mirror(grid: &ArcGrid, axis_x: u8, axis_y: u8) -> Option<ArcGrid> {
    let axis_x = axis_x as usize;
    let axis_y = axis_y as usize;
    let mut data = grid.data.clone();

    for r in 0..grid.height {
        for c in 0..grid.width {
            let mirror_r = axis_y.abs_diff(r);
            let mirror_c = axis_x.abs_diff(c);
            if mirror_r < grid.height && mirror_c < grid.width {
                data[r][c] = grid.data[mirror_r][mirror_c];
            }
        }
    }

    Some(ArcGrid::from_data(data).unwrap())
}

// ─── New Primitives ──────────────────────────────────────────────────────────

fn apply_tile(grid: &ArcGrid, n: u8, m: u8) -> Option<ArcGrid> {
    if n == 0 || m == 0 || n > 4 || m > 4 {
        return None;
    }
    let h = grid.height;
    let w = grid.width;
    if h == 0 || w == 0 {
        return None;
    }
    let new_h = h * m as usize;
    let new_w = w * n as usize;
    if new_h > 30 || new_w > 30 {
        return None;
    }

    let mut data = vec![vec![0u8; new_w]; new_h];
    for ty in 0..m as usize {
        for tx in 0..n as usize {
            for y in 0..h {
                for x in 0..w {
                    data[ty * h + y][tx * w + x] = grid.data[y][x];
                }
            }
        }
    }
    ArcGrid::from_data(data).ok()
}

fn apply_crop(grid: &ArcGrid, x: u8, y: u8, w: u8, h: u8) -> Option<ArcGrid> {
    let x = x as usize;
    let y = y as usize;
    let w = w as usize;
    let h = h as usize;

    if x + w > grid.width || y + h > grid.height || w == 0 || h == 0 {
        return None;
    }

    let mut data = vec![vec![0u8; w]; h];
    for dy in 0..h {
        for dx in 0..w {
            data[dy][dx] = grid.data[y + dy][x + dx];
        }
    }
    ArcGrid::from_data(data).ok()
}

fn apply_replace_color(grid: &ArcGrid, src: u8, dst: u8) -> Option<ArcGrid> {
    if src == dst {
        return None;
    }
    let mut data = grid.data.clone();
    for row in &mut data {
        for cell in row {
            if *cell == src {
                *cell = dst;
            }
        }
    }
    ArcGrid::from_data(data).ok()
}

fn apply_scale(grid: &ArcGrid, factor: u8) -> Option<ArcGrid> {
    if factor == 0 || factor > 3 {
        return None;
    }
    let h = grid.height;
    let w = grid.width;
    if h == 0 || w == 0 {
        return None;
    }
    let new_h = h * factor as usize;
    let new_w = w * factor as usize;
    if new_h > 30 || new_w > 30 {
        return None;
    }

    let mut data = vec![vec![0u8; new_w]; new_h];
    for y in 0..h {
        for x in 0..w {
            for dy in 0..factor as usize {
                for dx in 0..factor as usize {
                    data[y * factor as usize + dy][x * factor as usize + dx] = grid.data[y][x];
                }
            }
        }
    }
    ArcGrid::from_data(data).ok()
}

fn apply_crop_content(grid: &ArcGrid) -> Option<ArcGrid> {
    if grid.height == 0 || grid.width == 0 {
        return Some(grid.clone());
    }
    
    let bg = most_common_color(grid);
    let mut min_r = grid.height;
    let mut max_r = 0;
    let mut min_c = grid.width;
    let mut max_c = 0;
    
    for r in 0..grid.height {
        for c in 0..grid.width {
            if grid.data[r][c] != bg {
                min_r = min_r.min(r);
                max_r = max_r.max(r);
                min_c = min_c.min(c);
                max_c = max_c.max(c);
            }
        }
    }
    
    if min_r > max_r || min_c > max_c {
        return Some(grid.clone());
    }
    
    let mut data = Vec::with_capacity(max_r - min_r + 1);
    for r in min_r..=max_r {
        data.push(grid.data[r][min_c..=max_c].to_vec());
    }
    ArcGrid::from_data(data).ok()
}

fn most_common_color(grid: &ArcGrid) -> u8 {
    let mut counts = [0usize; 10];
    for row in &grid.data {
        for &c in row {
            counts[c as usize] += 1;
        }
    }
    let mut max_color = 0;
    let mut max_count = 0;
    for (color, &count) in counts.iter().enumerate() {
        if count > max_count {
            max_count = count;
            max_color = color;
        }
    }
    max_color as u8
}

// ─── Reused Grid Primitives ──────────────────────────────────────────────────

/// Rotates a grid 90° clockwise.
fn rotate90(grid: &ArcGrid) -> Option<ArcGrid> {
    if grid.width == 0 || grid.height == 0 {
        return None;
    }
    let new_width = grid.height;
    let new_height = grid.width;
    let mut data = vec![vec![0u8; new_width]; new_height];
    for r in 0..grid.height {
        for c in 0..grid.width {
            data[grid.width - 1 - c][r] = grid.data[r][c];
        }
    }
    ArcGrid::from_data(data).ok()
}

/// Rotates a grid 180°.
fn rotate180(grid: &ArcGrid) -> ArcGrid {
    let mut data = grid.data.clone();
    for row in &mut data {
        row.reverse();
    }
    data.reverse();
    ArcGrid::from_data(data).unwrap()
}

/// Rotates a grid 270° clockwise (90° counter-clockwise).
fn rotate270(grid: &ArcGrid) -> Option<ArcGrid> {
    if grid.width == 0 || grid.height == 0 {
        return None;
    }
    let new_width = grid.height;
    let new_height = grid.width;
    let mut data = vec![vec![0u8; new_width]; new_height];
    for r in 0..grid.height {
        for c in 0..grid.width {
            data[c][grid.height - 1 - r] = grid.data[r][c];
        }
    }
    ArcGrid::from_data(data).ok()
}

/// Flips a grid horizontally (left-right mirror).
fn flip_horizontal(grid: &ArcGrid) -> ArcGrid {
    let mut data = grid.data.clone();
    for row in &mut data {
        row.reverse();
    }
    ArcGrid::from_data(data).unwrap()
}

/// Flips a grid vertically (top-bottom mirror).
fn flip_vertical(grid: &ArcGrid) -> ArcGrid {
    let mut data = grid.data.clone();
    data.reverse();
    ArcGrid::from_data(data).unwrap()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let result = apply_identity(&grid).unwrap();
        assert_eq!(result, grid);
    }

    #[test]
    fn test_rotate90() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        // rotate90 in dsl_solver.rs is counter-clockwise: [[2,4],[1,3]]
        assert_eq!(result.data, vec![vec![2, 4], vec![1, 3]]);
    }

    #[test]
    fn test_rotate180() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(1, 1, 0, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![4, 3], vec![2, 1]]);
    }

    #[test]
    fn test_rotate270() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(1, 2, 0, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        // rotate270 in dsl_solver.rs is clockwise: [[3,1],[4,2]]
        assert_eq!(result.data, vec![vec![3, 1], vec![4, 2]]);
    }

    #[test]
    fn test_flip_horizontal() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(2, 0, 0, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![2, 1], vec![4, 3]]);
    }

    #[test]
    fn test_flip_vertical() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(2, 1, 0, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![3, 4], vec![1, 2]]);
    }

    #[test]
    fn test_move() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 0]]).unwrap();
        let token = ArcOpToken::new(3, 1, 0, 0, 0, 0, 0, 0); // dx=1, dy=0
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![0, 1], vec![0, 0]]);
    }

    #[test]
    fn test_fill() {
        let grid = ArcGrid::from_data(vec![vec![0, 0], vec![0, 0]]).unwrap();
        let token = ArcOpToken::new(4, 5, 0, 0, 2, 2, 0, 0); // color=5, x=0, y=0, w=2, h=2
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![5, 5], vec![5, 5]]);
    }

    #[test]
    fn test_copy() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(5, 0, 0, 1, 1, 1, 1, 0); // copy (0,0)->(1,1), 1x1
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![1, 2], vec![3, 1]]);
    }

    #[test]
    fn test_gravity_down() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let token = ArcOpToken::new(6, 0, 0, 0, 0, 0, 0, 0); // gravity down
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![0, 0], vec![1, 2]]);
    }

    #[test]
    fn test_gravity_up() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let token = ArcOpToken::new(6, 1, 0, 0, 0, 0, 0, 0); // gravity up
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![1, 2], vec![0, 0]]);
    }

    #[test]
    fn test_gravity_left() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let token = ArcOpToken::new(6, 2, 0, 0, 0, 0, 0, 0); // gravity left
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![1, 0], vec![2, 0]]);
    }

    #[test]
    fn test_gravity_right() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 2]]).unwrap();
        let token = ArcOpToken::new(6, 3, 0, 0, 0, 0, 0, 0); // gravity right
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![0, 1], vec![0, 2]]);
    }

    #[test]
    fn test_mirror() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(7, 1, 1, 0, 0, 0, 0, 0); // mirror at (1,1)
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![4, 3], vec![2, 1]]);
    }

    #[test]
    fn test_tile_2x2() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(8, 2, 2, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![
            vec![1, 2, 1, 2],
            vec![3, 4, 3, 4],
            vec![1, 2, 1, 2],
            vec![3, 4, 3, 4],
        ]);
    }

    #[test]
    fn test_crop() {
        let grid = ArcGrid::from_data(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]).unwrap();
        let token = ArcOpToken::new(9, 1, 1, 2, 2, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![5, 6], vec![8, 9]]);
    }

    #[test]
    fn test_replace_color() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![2, 1]]).unwrap();
        let token = ArcOpToken::new(10, 1, 9, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![vec![9, 2], vec![2, 9]]);
    }

    #[test]
    fn test_scale_2x() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let token = ArcOpToken::new(11, 2, 0, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        assert_eq!(result.data, vec![
            vec![1, 1, 2, 2],
            vec![1, 1, 2, 2],
            vec![3, 3, 4, 4],
            vec![3, 3, 4, 4],
        ]);
    }

    #[test]
    fn test_crop_content() {
        let grid = ArcGrid::from_data(vec![
            vec![0, 0, 0, 0, 0],
            vec![0, 1, 1, 0, 0],
            vec![0, 1, 1, 0, 0],
            vec![0, 0, 0, 2, 0],
            vec![0, 0, 0, 0, 0],
        ]).unwrap();
        let token = ArcOpToken::new(12, 0, 0, 0, 0, 0, 0, 0);
        let result = apply_arc_op(&grid, &token).unwrap();
        // Background is 0, content bbox is rows 1..=3, cols 1..=3
        assert_eq!(result.data, vec![
            vec![1, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 2],
        ]);
    }

    #[test]
    fn test_apply_program_sequence() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 0]]).unwrap();
        let program = ArcProgram::from_tokens(vec![
            ArcOpToken::new(3, 1, 0, 0, 0, 0, 0, 0), // Move dx=1
            ArcOpToken::new(4, 5, 0, 0, 2, 2, 0, 0), // Fill color=5, w=2, h=2
        ]);
        let result = apply_program(&grid, &program).unwrap();
        assert_eq!(result.data, vec![vec![5, 5], vec![5, 5]]);
    }

    #[test]
    fn test_apply_program_short_circuit() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 0]]).unwrap();
        let program = ArcProgram::from_tokens(vec![
            ArcOpToken::new(3, 1, 0, 0, 0, 0, 0, 0), // Move dx=1
            ArcOpToken::new(3, 10, 0, 0, 0, 0, 0, 0), // Move dx=10 (clipped off-grid)
        ]);
        let result = apply_program(&grid, &program).unwrap();
        assert_eq!(result.data, vec![vec![0, 0], vec![0, 0]]);
    }
}
