//! Brute-Force DSL Solver für ARC-AGI-1
//!
//! Programm-Synthese auf Train-Paaren: Für jede Task wird das erste Programm
//! gesucht, das ALLE Train-Paare löst. Dann wird es auf Test-Inputs angewendet.
//!
//! DSL-Primitive:
//! - Identity, Rotate90/180/270, FlipH, FlipV
//! - ColorMap (inferiert aus Train-Paaren)
//! - Invert, FillBackground
//! - CropBorder1, PadBorder1, Tile2x2

use crate::vision::{ArcGrid, ArcTask};
use std::collections::HashMap;

// ─── Operationen ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipH,
    FlipV,
    Invert,
    FillBackground,
    CropBorder1,
    PadBorder1,
    Tile2x2,
    Scale2x,
    Scale3x,
    MirrorH,
    MirrorV,
    CropCenter,
    CropToContent,
    DiagonalFlip,
    Outline,
    ExtractLargestComponent,
    ExtractSecondLargestComponent,
    ExtractSmallestComponent,
    RemoveIsolatedPixels,
    RemoveEmptyRows,
    RemoveEmptyCols,
    FillEnclosed,
}

impl Op {
    /// Wendet die Operation auf ein Grid an.
    /// Gibt `None` zurück, wenn die Operation fehlschlägt (z.B. Größenfehler).
    pub fn apply(&self, grid: &ArcGrid) -> Option<ArcGrid> {
        match self {
            Op::Identity => Some(identity(grid)),
            Op::Rotate90 => rotate90(grid),
            Op::Rotate180 => Some(rotate180(grid)),
            Op::Rotate270 => rotate270(grid),
            Op::FlipH => Some(flip_horizontal(grid)),
            Op::FlipV => Some(flip_vertical(grid)),
            Op::Invert => Some(invert_colors(grid)),
            Op::FillBackground => Some(fill_background(grid)),
            Op::CropBorder1 => Some(crop_border1(grid)),
            Op::PadBorder1 => Some(pad_border1(grid)),
            Op::Tile2x2 => Some(tile_2x2(grid)),
            Op::Scale2x => Some(scale_2x(grid)),
            Op::Scale3x => Some(scale_3x(grid)),
            Op::MirrorH => Some(mirror_horizontal(grid)),
            Op::MirrorV => Some(mirror_vertical(grid)),
            Op::CropCenter => Some(crop_center(grid)),
            Op::CropToContent => Some(crop_to_content(grid)),
            Op::DiagonalFlip => diagonal_flip(grid),
            Op::Outline => Some(outline(grid)),
            Op::ExtractLargestComponent => Some(extract_largest_component(grid)),
            Op::ExtractSecondLargestComponent => Some(extract_second_largest_component(grid)),
            Op::ExtractSmallestComponent => Some(extract_smallest_component(grid)),
            Op::RemoveIsolatedPixels => Some(remove_isolated_pixels(grid)),
            Op::RemoveEmptyRows => Some(remove_empty_rows(grid)),
            Op::RemoveEmptyCols => Some(remove_empty_cols(grid)),
            Op::FillEnclosed => Some(fill_enclosed(grid)),
        }
    }

    /// Name für Debug-Ausgaben
    pub fn name(&self) -> &'static str {
        match self {
            Op::Identity => "Id",
            Op::Rotate90 => "R90",
            Op::Rotate180 => "R180",
            Op::Rotate270 => "R270",
            Op::FlipH => "FH",
            Op::FlipV => "FV",
            Op::Invert => "Inv",
            Op::FillBackground => "FillBG",
            Op::CropBorder1 => "Crop1",
            Op::PadBorder1 => "Pad1",
            Op::Tile2x2 => "Tile2x2",
            Op::Scale2x => "Scale2x",
            Op::Scale3x => "Scale3x",
            Op::MirrorH => "MirrorH",
            Op::MirrorV => "MirrorV",
            Op::CropCenter => "CropCtr",
            Op::CropToContent => "CropContent",
            Op::DiagonalFlip => "DiagFlip",
            Op::Outline => "Outline",
            Op::ExtractLargestComponent => "ExtractLC",
            Op::ExtractSecondLargestComponent => "Extract2LC",
            Op::ExtractSmallestComponent => "ExtractSC",
            Op::RemoveIsolatedPixels => "RmIsolated",
            Op::RemoveEmptyRows => "RmEmptyR",
            Op::RemoveEmptyCols => "RmEmptyC",
            Op::FillEnclosed => "FillEnclosed",
        }
    }
}

/// Ein Programm = Sequenz von Operationen + optionale ColorMap
#[derive(Debug, Clone)]
pub struct Program {
    pub ops: Vec<Op>,
    pub color_map: Option<Vec<Option<u8>>>,
}

impl Program {
    pub fn new(ops: Vec<Op>) -> Self {
        Self { ops, color_map: None }
    }

    pub fn with_color_map(ops: Vec<Op>, mapping: Vec<Option<u8>>) -> Self {
        Self { ops, color_map: Some(mapping) }
    }

    /// Wendet das Programm auf ein Grid an
    pub fn apply(&self, grid: &ArcGrid) -> Option<ArcGrid> {
        let mut current = grid.clone();
        
        // Wende ColorMap zuerst an (falls vorhanden)
        if let Some(ref mapping) = self.color_map {
            current = apply_color_map(&current, mapping);
        }
        
        for op in &self.ops {
            current = op.apply(&current)?;
        }
        Some(current)
    }

    /// Prüft, ob das Programm alle Train-Paare löst
    pub fn solves_train(&self, task: &ArcTask) -> bool {
        for (input, expected) in &task.train_pairs {
            if let Some(output) = self.apply(input) {
                if !grids_equal(&output, expected) {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

// ─── Grid-Operationen ───────────────────────────────────────────────────────

fn identity(grid: &ArcGrid) -> ArcGrid {
    grid.clone()
}

fn rotate180(grid: &ArcGrid) -> ArcGrid {
    let mut data = grid.data.clone();
    for row in &mut data {
        row.reverse();
    }
    data.reverse();
    ArcGrid::from_data(data).unwrap()
}

fn flip_horizontal(grid: &ArcGrid) -> ArcGrid {
    let mut data = grid.data.clone();
    for row in &mut data {
        row.reverse();
    }
    ArcGrid::from_data(data).unwrap()
}

fn flip_vertical(grid: &ArcGrid) -> ArcGrid {
    let mut data = grid.data.clone();
    data.reverse();
    ArcGrid::from_data(data).unwrap()
}

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

fn invert_colors(grid: &ArcGrid) -> ArcGrid {
    let data: Vec<Vec<u8>> = grid.data.iter()
        .map(|row| row.iter().map(|&c| 9 - c).collect())
        .collect();
    ArcGrid::from_data(data).unwrap()
}

fn fill_background(grid: &ArcGrid) -> ArcGrid {
    let bg = most_common_color(grid);
    let data = vec![vec![bg; grid.width]; grid.height];
    ArcGrid::from_data(data).unwrap()
}

fn crop_border1(grid: &ArcGrid) -> ArcGrid {
    if grid.height <= 2 || grid.width <= 2 {
        return grid.clone();
    }
    let mut data = Vec::with_capacity(grid.height - 2);
    for r in 1..grid.height - 1 {
        data.push(grid.data[r][1..grid.width - 1].to_vec());
    }
    ArcGrid::from_data(data).unwrap()
}

fn pad_border1(grid: &ArcGrid) -> ArcGrid {
    let bg = most_common_color(grid);
    let mut data = vec![vec![bg; grid.width + 2]; grid.height + 2];
    for r in 0..grid.height {
        for c in 0..grid.width {
            data[r + 1][c + 1] = grid.data[r][c];
        }
    }
    ArcGrid::from_data(data).unwrap()
}

fn tile_2x2(grid: &ArcGrid) -> ArcGrid {
    let mut data = vec![vec![0u8; grid.width * 2]; grid.height * 2];
    for r in 0..grid.height {
        for c in 0..grid.width {
            let val = grid.data[r][c];
            data[r][c] = val;
            data[r][c + grid.width] = val;
            data[r + grid.height][c] = val;
            data[r + grid.height][c + grid.width] = val;
        }
    }
    ArcGrid::from_data(data).unwrap()
}

fn scale_2x(grid: &ArcGrid) -> ArcGrid {
    let mut data = vec![vec![0u8; grid.width * 2]; grid.height * 2];
    for r in 0..grid.height {
        for c in 0..grid.width {
            let val = grid.data[r][c];
            data[r * 2][c * 2] = val;
            data[r * 2][c * 2 + 1] = val;
            data[r * 2 + 1][c * 2] = val;
            data[r * 2 + 1][c * 2 + 1] = val;
        }
    }
    ArcGrid::from_data(data).unwrap()
}

fn scale_3x(grid: &ArcGrid) -> ArcGrid {
    let mut data = vec![vec![0u8; grid.width * 3]; grid.height * 3];
    for r in 0..grid.height {
        for c in 0..grid.width {
            let val = grid.data[r][c];
            for dr in 0..3 {
                for dc in 0..3 {
                    data[r * 3 + dr][c * 3 + dc] = val;
                }
            }
        }
    }
    ArcGrid::from_data(data).unwrap()
}

fn mirror_horizontal(grid: &ArcGrid) -> ArcGrid {
    if grid.width <= 1 {
        return grid.clone();
    }
    let mut data = grid.data.clone();
    for row in &mut data {
        let w = row.len();
        for c in 0..w {
            let src = if c < w / 2 { c } else { w - 1 - c };
            row[c] = row[src];
        }
    }
    ArcGrid::from_data(data).unwrap()
}

fn mirror_vertical(grid: &ArcGrid) -> ArcGrid {
    if grid.height <= 1 {
        return grid.clone();
    }
    let mut data = grid.data.clone();
    let h = data.len();
    for r in 0..h {
        let src = if r < h / 2 { r } else { h - 1 - r };
        data[r] = data[src].clone();
    }
    ArcGrid::from_data(data).unwrap()
}

fn crop_center(grid: &ArcGrid) -> ArcGrid {
    if grid.height <= 2 || grid.width <= 2 {
        return grid.clone();
    }
    let new_h = grid.height - 2;
    let new_w = grid.width - 2;
    let mut data = Vec::with_capacity(new_h);
    for r in 1..grid.height - 1 {
        data.push(grid.data[r][1..grid.width - 1].to_vec());
    }
    ArcGrid::from_data(data).unwrap()
}

// --- Objektbasierte Operationen ---

/// Isoliert die größte 4-connectivity-Komponente (ungleich 0).
fn extract_largest_component(grid: &ArcGrid) -> ArcGrid {
    if grid.height == 0 || grid.width == 0 {
        return grid.clone();
    }
    let mut visited = vec![vec![false; grid.width]; grid.height];
    let mut largest_size = 0usize;
    let mut largest_color = 0u8;
    let mut largest_pixels = Vec::new();

    for r in 0..grid.height {
        for c in 0..grid.width {
            if visited[r][c] || grid.data[r][c] == 0 {
                continue;
            }
            let color = grid.data[r][c];
            let mut component = Vec::new();
            let mut stack = vec![(r, c)];

            while let Some((cr, cc)) = stack.pop() {
                if visited[cr][cc] {
                    continue;
                }
                visited[cr][cc] = true;
                component.push((cr, cc));

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

            if component.len() > largest_size {
                largest_size = component.len();
                largest_color = color;
                largest_pixels = component;
            }
        }
    }

    let mut data = vec![vec![0u8; grid.width]; grid.height];
    for (r, c) in largest_pixels {
        data[r][c] = largest_color;
    }
    ArcGrid::from_data(data).unwrap()
}

/// Entfernt Pixel, die keinen 4-connectivity-Nachbarn derselben Farbe haben.
fn remove_isolated_pixels(grid: &ArcGrid) -> ArcGrid {
    let mut data = grid.data.clone();

    for r in 0..grid.height {
        for c in 0..grid.width {
            let color = grid.data[r][c];
            if color == 0 {
                continue;
            }
            let has_neighbor = [
                r > 0 && grid.data[r - 1][c] == color,
                r + 1 < grid.height && grid.data[r + 1][c] == color,
                c > 0 && grid.data[r][c - 1] == color,
                c + 1 < grid.width && grid.data[r][c + 1] == color,
            ]
            .iter()
            .any(|&x| x);

            if !has_neighbor {
                data[r][c] = 0;
            }
        }
    }
    ArcGrid::from_data(data).unwrap()
}

/// Füllt Hintergrund-Löcher (0), die von Nicht-Hintergrund umschlossen sind.
fn fill_enclosed(grid: &ArcGrid) -> ArcGrid {
    if grid.height == 0 || grid.width == 0 {
        return grid.clone();
    }
    let mut visited = vec![vec![false; grid.width]; grid.height];
    let mut stack = Vec::new();

    for r in 0..grid.height {
        for &c in &[0, grid.width - 1] {
            if grid.data[r][c] == 0 && !visited[r][c] {
                stack.push((r, c));
            }
        }
    }
    for c in 0..grid.width {
        for &r in &[0, grid.height - 1] {
            if grid.data[r][c] == 0 && !visited[r][c] {
                stack.push((r, c));
            }
        }
    }

    while let Some((r, c)) = stack.pop() {
        if visited[r][c] {
            continue;
        }
        visited[r][c] = true;

        let neighbors = [
            r.checked_sub(1).map(|nr| (nr, c)),
            Some((r + 1, c)).filter(|_| r + 1 < grid.height),
            c.checked_sub(1).map(|nc| (r, nc)),
            Some((r, c + 1)).filter(|_| c + 1 < grid.width),
        ];
        for n in neighbors.iter().flatten() {
            if !visited[n.0][n.1] && grid.data[n.0][n.1] == 0 {
                stack.push(*n);
            }
        }
    }

    let mut data = grid.data.clone();
    for r in 0..grid.height {
        for c in 0..grid.width {
            if grid.data[r][c] == 0 && !visited[r][c] {
                let mut color_counts = [0u8; 10];
                let neighbors = [
                    r.checked_sub(1).map(|nr| grid.data[nr][c]),
                    Some((r + 1 < grid.height).then(|| grid.data[r + 1][c])).flatten(),
                    c.checked_sub(1).map(|nc| grid.data[r][nc]),
                    Some((c + 1 < grid.width).then(|| grid.data[r][c + 1])).flatten(),
                ];
                for n in neighbors.iter().flatten() {
                    if *n != 0 {
                        color_counts[*n as usize] += 1;
                    }
                }
                let best_color = color_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &v)| v)
                    .map(|(i, _)| i as u8)
                    .unwrap_or(1);
                data[r][c] = best_color;
            }
        }
    }
    ArcGrid::from_data(data).unwrap()
}

/// Crop to bounding box of non-background pixels.
fn crop_to_content(grid: &ArcGrid) -> ArcGrid {
    if grid.height == 0 || grid.width == 0 {
        return grid.clone();
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
        return ArcGrid::from_data(vec![vec![bg; 1]; 1]).unwrap();
    }

    let mut data = Vec::with_capacity(max_r - min_r + 1);
    for r in min_r..=max_r {
        data.push(grid.data[r][min_c..=max_c].to_vec());
    }
    ArcGrid::from_data(data).unwrap()
}

/// Transpose the grid (swap rows and columns).
fn diagonal_flip(grid: &ArcGrid) -> Option<ArcGrid> {
    if grid.width == 0 || grid.height == 0 {
        return Some(grid.clone());
    }
    let mut data = vec![vec![0u8; grid.height]; grid.width];
    for r in 0..grid.height {
        for c in 0..grid.width {
            data[c][r] = grid.data[r][c];
        }
    }
    ArcGrid::from_data(data).ok()
}

/// Keep only the outline (border pixels) of non-zero connected components.
fn outline(grid: &ArcGrid) -> ArcGrid {
    if grid.height == 0 || grid.width == 0 {
        return grid.clone();
    }
    let mut data = grid.data.clone();

    for r in 0..grid.height {
        for c in 0..grid.width {
            if data[r][c] == 0 {
                continue;
            }
            let has_bg_neighbor = [
                r > 0 && grid.data[r - 1][c] == 0,
                r + 1 < grid.height && grid.data[r + 1][c] == 0,
                c > 0 && grid.data[r][c - 1] == 0,
                c + 1 < grid.width && grid.data[r][c + 1] == 0,
            ]
            .iter()
            .any(|&x| x);

            if !has_bg_neighbor {
                data[r][c] = 0;
            }
        }
    }
    ArcGrid::from_data(data).unwrap()
}

/// Extracts the second-largest connected component (by pixel count).
fn extract_second_largest_component(grid: &ArcGrid) -> ArcGrid {
    if grid.height == 0 || grid.width == 0 {
        return grid.clone();
    }
    let mut visited = vec![vec![false; grid.width]; grid.height];
    let mut components: Vec<(usize, u8, Vec<(usize, usize)>)> = Vec::new();

    for r in 0..grid.height {
        for c in 0..grid.width {
            if visited[r][c] || grid.data[r][c] == 0 {
                continue;
            }
            let color = grid.data[r][c];
            let mut component = Vec::new();
            let mut stack = vec![(r, c)];

            while let Some((cr, cc)) = stack.pop() {
                if visited[cr][cc] {
                    continue;
                }
                visited[cr][cc] = true;
                component.push((cr, cc));

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

            components.push((component.len(), color, component));
        }
    }

    components.sort_by(|a, b| b.0.cmp(&a.0));

    if components.len() < 2 {
        let mut data = vec![vec![0u8; grid.width]; grid.height];
        return ArcGrid::from_data(data).unwrap();
    }

    let (_, color, pixels) = &components[1];
    let mut data = vec![vec![0u8; grid.width]; grid.height];
    for (r, c) in pixels {
        data[*r][*c] = *color;
    }
    ArcGrid::from_data(data).unwrap()
}

/// Extracts the smallest non-zero connected component.
fn extract_smallest_component(grid: &ArcGrid) -> ArcGrid {
    if grid.height == 0 || grid.width == 0 {
        return grid.clone();
    }
    let mut visited = vec![vec![false; grid.width]; grid.height];
    let mut best_size = usize::MAX;
    let mut best_color = 0u8;
    let mut best_pixels = Vec::new();

    for r in 0..grid.height {
        for c in 0..grid.width {
            if visited[r][c] || grid.data[r][c] == 0 {
                continue;
            }
            let color = grid.data[r][c];
            let mut component = Vec::new();
            let mut stack = vec![(r, c)];

            while let Some((cr, cc)) = stack.pop() {
                if visited[cr][cc] {
                    continue;
                }
                visited[cr][cc] = true;
                component.push((cr, cc));

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

            if component.len() < best_size {
                best_size = component.len();
                best_color = color;
                best_pixels = component;
            }
        }
    }

    let mut data = vec![vec![0u8; grid.width]; grid.height];
    for (r, c) in best_pixels {
        data[r][c] = best_color;
    }
    ArcGrid::from_data(data).unwrap()
}

/// Remove rows that consist entirely of background color.
fn remove_empty_rows(grid: &ArcGrid) -> ArcGrid {
    let bg = most_common_color(grid);
    let mut data = Vec::new();
    for row in &grid.data {
        if row.iter().any(|&c| c != bg) {
            data.push(row.clone());
        }
    }
    if data.is_empty() {
        data.push(vec![bg; grid.width]);
    }
    ArcGrid::from_data(data).unwrap()
}

/// Remove columns that consist entirely of background color.
fn remove_empty_cols(grid: &ArcGrid) -> ArcGrid {
    if grid.height == 0 || grid.width == 0 {
        return grid.clone();
    }
    let bg = most_common_color(grid);
    let mut keep_cols = Vec::new();
    for c in 0..grid.width {
        let mut has_content = false;
        for r in 0..grid.height {
            if grid.data[r][c] != bg {
                has_content = true;
                break;
            }
        }
        if has_content {
            keep_cols.push(c);
        }
    }

    if keep_cols.is_empty() {
        return ArcGrid::from_data(vec![vec![bg; 1]; grid.height]).unwrap();
    }

    let mut data = Vec::with_capacity(grid.height);
    for row in &grid.data {
        data.push(keep_cols.iter().map(|&c| row[c]).collect());
    }
    ArcGrid::from_data(data).unwrap()
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

// ─── ColorMap-Inferenz ──────────────────────────────────────────────────────

/// Inferiert eine konsistente Farbzuordnung aus Train-Paaren.
/// Berücksichtigt nur Paare mit gleicher Größe.
pub fn infer_color_map(task: &ArcTask) -> Option<Vec<Option<u8>>> {
    let mut mapping: Vec<Option<u8>> = vec![None; 10];
    let mut used_output = vec![false; 10];

    for (input, output) in &task.train_pairs {
        if input.width != output.width || input.height != output.height {
            return None; // Größenänderung → keine einfache ColorMap
        }
        for (in_cell, out_cell) in input.data.iter().flatten().zip(output.data.iter().flatten()) {
            let in_c = *in_cell as usize;
            let out_c = *out_cell as usize;
            match mapping[in_c] {
                None => {
                    if used_output[out_c] {
                        return None; // Konflikt: Output-Farbe bereits vergeben
                    }
                    mapping[in_c] = Some(out_c as u8);
                    used_output[out_c] = true;
                }
                Some(expected) => {
                    if expected != out_c as u8 {
                        return None; // Inkonsistente Zuordnung
                    }
                }
            }
        }
    }

    // Prüfe, dass mindestens eine Mapping verwendet wurde
    if mapping.iter().all(|&m| m.is_none()) {
        return None;
    }

    Some(mapping)
}

/// Wendet eine ColorMap auf ein Grid an
pub fn apply_color_map(grid: &ArcGrid, mapping: &[Option<u8>]) -> ArcGrid {
    let data: Vec<Vec<u8>> = grid.data.iter()
        .map(|row| row.iter().map(|&c| mapping[c as usize].unwrap_or(c)).collect())
        .collect();
    ArcGrid::from_data(data).unwrap()
}

// ─── Grid-Vergleich ─────────────────────────────────────────────────────────

pub fn grids_equal(a: &ArcGrid, b: &ArcGrid) -> bool {
    a == b
}

// ─── Brute-Force-Suche ──────────────────────────────────────────────────────

/// Durchsucht alle Programme bis zur gegebenen Länge.
/// Gibt das erste gefundene Programm zurück, das alle Train-Paare löst.
pub fn find_solving_program(task: &ArcTask, max_length: usize) -> Option<Program> {
    let base_ops = vec![
        Op::Identity,
        Op::Rotate90,
        Op::Rotate180,
        Op::Rotate270,
        Op::FlipH,
        Op::FlipV,
        Op::Invert,
        Op::FillBackground,
        Op::CropBorder1,
        Op::PadBorder1,
        Op::Tile2x2,
        Op::Scale2x,
        Op::Scale3x,
        Op::MirrorH,
        Op::MirrorV,
        Op::CropCenter,
        Op::CropToContent,
        Op::DiagonalFlip,
        Op::Outline,
        Op::ExtractLargestComponent,
        Op::ExtractSecondLargestComponent,
        Op::ExtractSmallestComponent,
        Op::RemoveIsolatedPixels,
        Op::RemoveEmptyRows,
        Op::RemoveEmptyCols,
        Op::FillEnclosed,
    ];

    // Prüfe ColorMap alleine
    if let Some(mapping) = infer_color_map(task) {
        let program = Program::with_color_map(vec![Op::Identity], mapping);
        if program.solves_train(task) {
            return Some(program);
        }
    }

    // Prüfe einzelne Operationen (Länge 1)
    for &op in &base_ops {
        let program = Program::new(vec![op]);
        if program.solves_train(task) {
            return Some(program);
        }
    }

    // Prüfe ColorMap + eine weitere Operation
    if let Some(mapping) = infer_color_map(task) {
        for &op in &base_ops {
            if op == Op::Identity {
                continue;
            }
            let program = Program::with_color_map(vec![op], mapping.clone());
            if program.solves_train(task) {
                return Some(program);
            }
        }
    }

    if max_length >= 2 {
        // Paare von Operationen
        for &op1 in &base_ops {
            for &op2 in &base_ops {
                if op1 == Op::Identity && op2 == Op::Identity {
                    continue;
                }
                let program = Program::new(vec![op1, op2]);
                if program.solves_train(task) {
                    return Some(program);
                }
            }
        }

        // ColorMap + Paar von Operationen
        if let Some(mapping) = infer_color_map(task) {
            for &op1 in &base_ops {
                for &op2 in &base_ops {
                    if op1 == Op::Identity && op2 == Op::Identity {
                        continue;
                    }
                    let program = Program::with_color_map(vec![op1, op2], mapping.clone());
                    if program.solves_train(task) {
                        return Some(program);
                    }
                }
            }
        }
    }

    if max_length >= 3 {
        // Tripel
        for &op1 in &base_ops {
            for &op2 in &base_ops {
                for &op3 in &base_ops {
                    if op1 == Op::Identity && op2 == Op::Identity && op3 == Op::Identity {
                        continue;
                    }
                    let program = Program::new(vec![op1, op2, op3]);
                    if program.solves_train(task) {
                        return Some(program);
                    }
                }
            }
        }

        // ColorMap + Tripel
        if let Some(mapping) = infer_color_map(task) {
            for &op1 in &base_ops {
                for &op2 in &base_ops {
                    for &op3 in &base_ops {
                        if op1 == Op::Identity && op2 == Op::Identity && op3 == Op::Identity {
                            continue;
                        }
                        let program = Program::with_color_map(vec![op1, op2, op3], mapping.clone());
                        if program.solves_train(task) {
                            return Some(program);
                        }
                    }
                }
            }
        }
    }

    None
}

// ─── Evaluation ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SolverResult {
    pub total: usize,
    pub solved: usize,       // Train-Paare durch Programm gelöst
    pub correct: usize,      // Test-Output exakt korrekt
    pub programs: Vec<(String, usize)>,  // (Programm-Beschreibung, Häufigkeit)
}

pub fn evaluate_solver(tasks: &[ArcTask], max_length: usize, n_tasks: usize) -> SolverResult {
    let mut result = SolverResult {
        total: n_tasks.min(tasks.len()),
        solved: 0,
        correct: 0,
        programs: Vec::new(),
    };

    let mut program_counts: HashMap<String, usize> = HashMap::new();

    for task in tasks.iter().take(n_tasks) {
        if let Some(program) = find_solving_program(task, max_length) {
            result.solved += 1;
            let desc = if program.color_map.is_some() {
                format!("CMap->{}", program.ops.iter().map(|op| op.name()).collect::<Vec<_>>().join("->"))
            } else {
                program.ops.iter().map(|op| op.name()).collect::<Vec<_>>().join("->")
            };
            *program_counts.entry(desc).or_insert(0) += 1;

            // Wende auf alle Test-Inputs an
            for (i, test_input) in task.test_inputs.iter().enumerate() {
                if let Some(output) = program.apply(test_input) {
                    if let Some(Some(expected)) = task.test_outputs.get(i) {
                        if grids_equal(&output, expected) {
                            result.correct += 1;
                        }
                    }
                }
            }
        }
    }

    result.programs = program_counts.into_iter().collect();
    result.programs.sort_by(|a, b| b.1.cmp(&a.1));

    result
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_solves_identity_task() {
        let task = ArcTask {
            id: "test".to_string(),
            train_pairs: vec![(
                ArcGrid::from_data(vec![vec![1, 0], vec![0, 1]]).unwrap(),
                ArcGrid::from_data(vec![vec![1, 0], vec![0, 1]]).unwrap(),
            )],
            test_inputs: vec![],
            test_outputs: vec![],
        };
        let program = Program::new(vec![Op::Identity]);
        assert!(program.solves_train(&task));
    }

    #[test]
    fn test_rotate180() {
        let grid = ArcGrid::from_data(vec![vec![1, 0], vec![0, 1]]).unwrap();
        let rotated = rotate180(&grid);
        assert_eq!(rotated.data, vec![vec![1, 0], vec![0, 1]]); // Symmetrisch
    }

    #[test]
    fn test_flip_horizontal() {
        let grid = ArcGrid::from_data(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let flipped = flip_horizontal(&grid);
        assert_eq!(flipped.data, vec![vec![2, 1], vec![4, 3]]);
    }
}
