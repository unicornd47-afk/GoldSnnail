//! ARC Compositional Solver — Grid Parser
//!
//! This module parses ARC grids into structured object graphs for analysis
//! and program generation. It extracts connected components, color clusters,
//! and spatial features that can guide program search.
//!
//! # Design Principles
//!
//! - **Non-destructive:** Parsing does not modify the input grid
//! - **DOD-compliant:** Uses flat arrays and indices, no pointer chasing
//! - **Elastic failure:** Returns empty results for invalid inputs, never panics
//! - **Composable:** Output feeds into program search heuristics

use crate::vision::ArcGrid;

// ─── Connected Component ─────────────────────────────────────────────────────

/// A connected component of non-zero pixels in an ARC grid.
///
/// Uses 4-connectivity (up, down, left, right).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    /// The color of this component.
    pub color: u8,
    /// The pixels in this component as (row, col) pairs.
    pub pixels: Vec<(usize, usize)>,
    /// Bounding box: (min_row, max_row, min_col, max_col).
    pub bbox: (usize, usize, usize, usize),
}

impl Component {
    /// Returns the number of pixels in this component.
    pub fn size(&self) -> usize {
        self.pixels.len()
    }

    /// Returns the width of the bounding box.
    pub fn width(&self) -> usize {
        self.bbox.3 - self.bbox.2 + 1
    }

    /// Returns the height of the bounding box.
    pub fn height(&self) -> usize {
        self.bbox.1 - self.bbox.0 + 1
    }

    /// Returns true if this component touches the grid border.
    pub fn touches_border(&self, grid_width: usize, grid_height: usize) -> bool {
        self.bbox.0 == 0
            || self.bbox.1 == grid_height - 1
            || self.bbox.2 == 0
            || self.bbox.3 == grid_width - 1
    }
}

/// Extracts all connected components from a grid.
///
/// Uses 4-connectivity. Components with color 0 (background) are ignored.
///
/// # Returns
///
/// A vector of `Component` structs, sorted by size (largest first).
///
/// # Example
///
/// ```
/// use goldsnnail::arc_parser::extract_components;
/// use goldsnnail::vision::ArcGrid;
///
/// let grid = ArcGrid::from_data(vec![
///     vec![1, 1, 0],
///     vec![0, 1, 0],
///     vec![0, 0, 2],
/// ]).unwrap();
/// let components = extract_components(&grid);
/// assert_eq!(components.len(), 2);
/// assert_eq!(components[0].color, 1); // Largest component
/// assert_eq!(components[0].size(), 3);
/// assert_eq!(components[1].color, 2);
/// assert_eq!(components[1].size(), 1);
/// ```
pub fn extract_components(grid: &ArcGrid) -> Vec<Component> {
    if grid.height == 0 || grid.width == 0 {
        return Vec::new();
    }

    let mut visited = vec![vec![false; grid.width]; grid.height];
    let mut components: Vec<Component> = Vec::new();

    for r in 0..grid.height {
        for c in 0..grid.width {
            if visited[r][c] || grid.data[r][c] == 0 {
                continue;
            }

            let color = grid.data[r][c];
            let mut pixels = Vec::new();
            let mut stack = vec![(r, c)];

            while let Some((cr, cc)) = stack.pop() {
                if visited[cr][cc] {
                    continue;
                }
                visited[cr][cc] = true;
                pixels.push((cr, cc));

                let neighbors = [
                    cr.checked_sub(1).map(|nr| (nr, cc)),
                    Some((cr + 1, cc)).filter(|_| cr + 1 < grid.height),
                    cc.checked_sub(1).map(|nc| (cr, nc)),
                    Some((cr, cc + 1)).filter(|_| cc + 1 < grid.width),
                ];
                for n in neighbors.iter().flatten() {
                    if !visited[n.0][n.1] && grid.data[n.0][n.1] == color {
                        stack.push(*n);
                    }
                }
            }

            // Compute bounding box
            let mut min_r = grid.height;
            let mut max_r = 0;
            let mut min_c = grid.width;
            let mut max_c = 0;
            for &(pr, pc) in &pixels {
                min_r = min_r.min(pr);
                max_r = max_r.max(pr);
                min_c = min_c.min(pc);
                max_c = max_c.max(pc);
            }

            components.push(Component {
                color,
                pixels,
                bbox: (min_r, max_r, min_c, max_c),
            });
        }
    }

    // Sort by size (largest first)
    components.sort_by(|a, b| b.size().cmp(&a.size()));
    components
}

// ─── Color Statistics ────────────────────────────────────────────────────────

/// Statistics about color distribution in a grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorStats {
    /// Count of each color (0-9).
    pub counts: [usize; 10],
    /// Total number of non-background pixels.
    pub total_non_bg: usize,
    /// The most common non-zero color.
    pub dominant_color: u8,
}

impl ColorStats {
    /// Computes color statistics for a grid.
    pub fn compute(grid: &ArcGrid) -> Self {
        let mut counts = [0usize; 10];
        for row in &grid.data {
            for &c in row {
                counts[c as usize] += 1;
            }
        }

        let total_non_bg = counts.iter().skip(1).sum();
        let dominant_color = counts
            .iter()
            .enumerate()
            .skip(1)
            .max_by_key(|&(_, &count)| count)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        Self {
            counts,
            total_non_bg,
            dominant_color,
        }
    }

    /// Returns the count for a specific color.
    pub fn count(&self, color: u8) -> usize {
        self.counts[color as usize]
    }
}

// ─── Object Graph ────────────────────────────────────────────────────────────

/// A structured representation of an ARC grid as a graph of objects.
///
/// This is the output of the parser, used by the program search to understand
/// the task structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectGraph {
    /// The original grid dimensions.
    pub width: usize,
    pub height: usize,
    /// Connected components sorted by size.
    pub components: Vec<Component>,
    /// Color statistics.
    pub colors: ColorStats,
    /// The most common color (background).
    pub background: u8,
}

impl ObjectGraph {
    /// Parses an `ArcGrid` into an `ObjectGraph`.
    pub fn parse(grid: &ArcGrid) -> Self {
        let colors = ColorStats::compute(grid);
        let background = most_common_color(grid);
        let components = extract_components(grid);

        Self {
            width: grid.width,
            height: grid.height,
            components,
            colors,
            background,
        }
    }

    /// Returns the number of non-background components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Returns the largest component, if any.
    pub fn largest_component(&self) -> Option<&Component> {
        self.components.first()
    }

    /// Returns true if the grid is empty (all background).
    pub fn is_empty(&self) -> bool {
        self.colors.total_non_bg == 0
    }

    /// Returns true if the grid has exactly one component.
    pub fn is_single_component(&self) -> bool {
        self.components.len() == 1
    }
}

// ─── Helper Functions ────────────────────────────────────────────────────────

/// Returns the most common color in a grid.
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_components_simple() {
        let grid = ArcGrid::from_data(vec![
            vec![1, 1, 0],
            vec![0, 1, 0],
            vec![0, 0, 2],
        ]).unwrap();
        let components = extract_components(&grid);
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].color, 1);
        assert_eq!(components[0].size(), 3);
        assert_eq!(components[1].color, 2);
        assert_eq!(components[1].size(), 1);
    }

    #[test]
    fn test_extract_components_empty() {
        let grid = ArcGrid::from_data(vec![
            vec![0, 0],
            vec![0, 0],
        ]).unwrap();
        let components = extract_components(&grid);
        assert!(components.is_empty());
    }

    #[test]
    fn test_component_bbox() {
        let grid = ArcGrid::from_data(vec![
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
        ]).unwrap();
        let components = extract_components(&grid);
        // Diagonal pixels are not 4-connected, so there are 3 separate components
        assert_eq!(components.len(), 3);
        assert_eq!(components[0].bbox, (0, 0, 0, 0));
        assert_eq!(components[0].width(), 1);
        assert_eq!(components[0].height(), 1);
    }

    #[test]
    fn test_component_touches_border() {
        let grid = ArcGrid::from_data(vec![
            vec![1, 0, 0],
            vec![0, 0, 0],
            vec![0, 0, 1],
        ]).unwrap();
        let components = extract_components(&grid);
        assert!(components[0].touches_border(3, 3));
        assert!(components[1].touches_border(3, 3));
    }

    #[test]
    fn test_color_stats() {
        let grid = ArcGrid::from_data(vec![
            vec![1, 1, 2],
            vec![0, 1, 0],
        ]).unwrap();
        let stats = ColorStats::compute(&grid);
        assert_eq!(stats.count(0), 2);
        assert_eq!(stats.count(1), 3);
        assert_eq!(stats.count(2), 1);
        assert_eq!(stats.total_non_bg, 4);
        assert_eq!(stats.dominant_color, 1);
    }

    #[test]
    fn test_object_graph_parse() {
        let grid = ArcGrid::from_data(vec![
            vec![1, 1, 0],
            vec![0, 1, 0],
            vec![0, 0, 2],
        ]).unwrap();
        let graph = ObjectGraph::parse(&grid);
        assert_eq!(graph.width, 3);
        assert_eq!(graph.height, 3);
        assert_eq!(graph.component_count(), 2);
        assert_eq!(graph.background, 0);
        assert!(!graph.is_empty());
        assert!(!graph.is_single_component());
    }
}
