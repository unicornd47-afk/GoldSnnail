//! ARC Compositional Solver — Program DSL
//!
//! This module defines the compact program representation for ARC-AGI-2
//! compositional solving. Each operation is encoded as an 8-byte token
//! (`ArcOpToken`), and a program is a sequence of such tokens.
//!
//! # Design
//!
//! The 8 primitive operations are inspired by the D8 Lattice / Containment
//! structure from the SHD-CCP protocol. Each token is exactly 8 bytes,
//! fitting in L1 cache. A depth-3 program is 24 bytes.
//!
//! ## Byte Layout
//!
//! ```text
//! [0] = op_code (0-7)
//! [1] = param_a (angle, axis, dx, color, direction, ...)
//! [2] = param_b (dy, src_x, ...)
//! [3] = param_c (0, src_y, ...)
//! [4] = param_d (0, dst_x, x, ...)
//! [5] = param_e (0, dst_y, y, ...)
//! [6] = param_f (0, w, ...)
//! [7] = param_g (0, h, ...)
//! ```
//!
//! ## Operations
//!
//! | Code | Name | Params | Description |
//! |------|------|--------|-------------|
//! | 0 | Identity | — | No-op |
//! | 1 | Rotate | angle (0=90°, 1=180°, 2=270°) | Rotate grid |
//! | 2 | Flip | axis (0=H, 1=V) | Flip horizontal/vertical |
//! | 3 | Move | dx, dy | Translate grid |
//! | 4 | Fill | color, x, y, w, h | Fill rectangle |
//! | 5 | Copy | src_x, src_y, dst_x, dst_y, w, h | Copy region |
//! | 6 | Gravity | direction (0=down, 1=up, 2=left, 3=right) | Gravity fall |
//! | 7 | Mirror | axis_x, axis_y | Mirror across point/axis |
//!
//! # Integration
//!
//! - `ArcOpToken` serializes to `DataTensor::U8[8]` (Phase 5)
//! - `ArcProgram` is a `Vec<ArcOpToken>` — directly streamable via MCP
//! - `ArcApplyEngine` consumes `ArcProgram` and produces `ArcGrid`

use crate::vision::ArcGrid;

// ─── Operation Codes ─────────────────────────────────────────────────────────

/// Operation codes for the ARC compositional DSL.
///
/// Each code corresponds to a primitive grid transformation.
/// The full token is 8 bytes: `[op_code, p1, p2, p3, p4, p5, p6, p7]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArcOpCode {
    Identity = 0,
    Rotate = 1,
    Flip = 2,
    Move = 3,
    Fill = 4,
    Copy = 5,
    Gravity = 6,
    Mirror = 7,
    Tile = 8,
    Crop = 9,
    ReplaceColor = 10,
    Scale = 11,
    CropContent = 12,
}

impl ArcOpCode {
    /// Returns the byte representation of this op code.
    pub fn as_byte(&self) -> u8 {
        *self as u8
    }

    /// Creates an `ArcOpCode` from a byte value.
    ///
    /// Returns `None` if the byte does not correspond to a valid operation.
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Identity),
            1 => Some(Self::Rotate),
            2 => Some(Self::Flip),
            3 => Some(Self::Move),
            4 => Some(Self::Fill),
            5 => Some(Self::Copy),
            6 => Some(Self::Gravity),
            7 => Some(Self::Mirror),
            8 => Some(Self::Tile),
            9 => Some(Self::Crop),
            10 => Some(Self::ReplaceColor),
            11 => Some(Self::Scale),
            12 => Some(Self::CropContent),
            _ => None,
        }
    }

    /// Returns the human-readable name of this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Identity => "Identity",
            Self::Rotate => "Rotate",
            Self::Flip => "Flip",
            Self::Move => "Move",
            Self::Fill => "Fill",
            Self::Copy => "Copy",
            Self::Gravity => "Gravity",
            Self::Mirror => "Mirror",
            Self::Tile => "Tile",
            Self::Crop => "Crop",
            Self::ReplaceColor => "ReplaceColor",
            Self::Scale => "Scale",
            Self::CropContent => "CropContent",
        }
    }
}

// ─── Operation Token ─────────────────────────────────────────────────────────

/// An 8-byte token representing a single ARC operation.
///
/// This is the fundamental unit of the compositional solver. Each token
/// encodes an operation code and up to 7 parameters.
///
/// # Memory Layout
///
/// ```text
/// [0] = op_code (0-7)
/// [1] = param_a
/// [2] = param_b
/// [3] = param_c
/// [4] = param_d
/// [5] = param_e
/// [6] = param_f
/// [7] = param_g
/// ```
///
/// # Example
///
/// ```
/// use goldworm::arc_program::ArcOpToken;
///
/// // Rotate 90°: [1, 0, 0, 0, 0, 0, 0, 0]
/// let token = ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0);
/// assert_eq!(token.bytes[0], 1); // Rotate
/// assert_eq!(token.bytes[1], 0); // 90°
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArcOpToken(pub [u8; 8]);

impl ArcOpToken {
    /// Creates a new operation token from 8 parameter bytes.
    ///
    /// # Arguments
    ///
    /// * `op_code` — The operation code (0-7)
    /// * `p1` through `p7` — Operation parameters
    pub fn new(op_code: u8, p1: u8, p2: u8, p3: u8, p4: u8, p5: u8, p6: u8, p7: u8) -> Self {
        Self([op_code, p1, p2, p3, p4, p5, p6, p7])
    }

    /// Creates a new operation token from an op code and a slice of parameters.
    ///
    /// Missing parameters are filled with zeros.
    pub fn from_parts(op_code: u8, params: &[u8]) -> Self {
        let mut bytes = [0u8; 8];
        bytes[0] = op_code;
        let len = params.len().min(7);
        bytes[1..1 + len].copy_from_slice(&params[..len]);
        Self(bytes)
    }

    /// Returns the operation code.
    pub fn op_code(&self) -> u8 {
        self.0[0]
    }

    /// Returns the operation code as an `ArcOpCode` enum.
    pub fn op(&self) -> Option<ArcOpCode> {
        ArcOpCode::from_byte(self.0[0])
    }

    /// Returns the parameter bytes (excluding op code).
    pub fn params(&self) -> &[u8; 7] {
        unsafe { &*(&self.0[1..] as *const [u8] as *const [u8; 7]) }
    }

    /// Returns a specific parameter by index (0-6).
    pub fn param(&self, index: usize) -> u8 {
        self.0[1 + index]
    }

    /// Returns the raw 8-byte array.
    pub fn bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// Creates an `ArcOpToken` from a raw byte array.
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }
}

// ─── Program ─────────────────────────────────────────────────────────────────

/// A sequence of ARC operation tokens representing a complete program.
///
/// A program is a compositional solution to an ARC task. It is applied
/// sequentially to the input grid to produce the output grid.
///
/// # Example
///
/// ```
/// use goldworm::arc_program::{ArcOpToken, ArcProgram};
///
/// // Rotate 90° then fill background with color 1
/// let program = ArcProgram::from_tokens(vec![
///     ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0), // Rotate 90°
///     ArcOpToken::new(4, 1, 0, 0, 0, 0, 0, 0), // Fill with color 1
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArcProgram {
    pub tokens: Vec<ArcOpToken>,
}

impl ArcProgram {
    /// Creates a new empty program.
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Creates a program from a vector of tokens.
    pub fn from_tokens(tokens: Vec<ArcOpToken>) -> Self {
        Self { tokens }
    }

    /// Pushes a token onto the program.
    pub fn push(&mut self, token: ArcOpToken) {
        self.tokens.push(token);
    }

    /// Returns the number of operations in the program.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the program is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the total byte size of the program (8 bytes per token).
    pub fn byte_size(&self) -> usize {
        self.tokens.len() * 8
    }

    /// Returns a description of the program for debugging.
    ///
    /// Example: `"Rotate(90°) -> Fill(color=1)"`
    pub fn describe(&self) -> String {
        self.tokens
            .iter()
            .map(|token| {
                if let Some(op) = token.op() {
                    format!("{}({:?})", op.name(), token.params())
                } else {
                    format!("Unknown({})", token.op_code())
                }
            })
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// Applies the program to a grid using the apply engine.
    ///
    /// This is a convenience wrapper around `arc_apply::apply_program`.
    pub fn apply_to(&self, grid: &ArcGrid) -> Option<ArcGrid> {
        crate::arc_apply::apply_program(grid, self)
    }

    /// Checks if the program solves all training pairs for a task.
    ///
    /// This is a convenience wrapper around `arc_apply::program_solves_train`.
    pub fn solves_train(&self, task: &crate::vision::ArcTask) -> bool {
        crate::arc_apply::program_solves_train(task, self)
    }
}

impl Default for ArcProgram {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Serialization ───────────────────────────────────────────────────────────

/// Serializes an `ArcProgram` to a flat `Vec<u8>` suitable for
/// JSON export or DataTensor conversion.
///
/// # Example
///
/// ```
/// use goldworm::arc_program::{ArcOpToken, ArcProgram, serialize_program};
///
/// let program = ArcProgram::from_tokens(vec![
///     ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0),
/// ]);
/// let bytes = serialize_program(&program);
/// assert_eq!(bytes.len(), 8);
/// ```
pub fn serialize_program(program: &ArcProgram) -> Vec<u8> {
    program
        .tokens
        .iter()
        .flat_map(|token| token.0.iter().copied())
        .collect()
}

/// Deserializes an `ArcProgram` from a flat `Vec<u8>`.
///
/// Returns `None` if the byte slice length is not a multiple of 8.
pub fn deserialize_program(bytes: &[u8]) -> Option<ArcProgram> {
    if bytes.len() % 8 != 0 {
        return None;
    }
    let tokens = bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            ArcOpToken(arr)
        })
        .collect();
    Some(ArcProgram { tokens })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_op_code_roundtrip() {
        for code in 0..8 {
            let op = ArcOpCode::from_byte(code).unwrap();
            assert_eq!(op.as_byte(), code);
        }
        assert!(ArcOpCode::from_byte(255).is_none());
    }

    #[test]
    fn arc_op_token_creation() {
        let token = ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(token.op_code(), 1);
        assert_eq!(token.op(), Some(ArcOpCode::Rotate));
        assert_eq!(token.param(0), 0);
    }

    #[test]
    fn arc_op_token_from_parts() {
        let token = ArcOpToken::from_parts(4, &[1, 2, 3]);
        assert_eq!(token.bytes(), &[4, 1, 2, 3, 0, 0, 0, 0]);
    }

    #[test]
    fn arc_program_describe() {
        let program = ArcProgram::from_tokens(vec![
            ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0),
            ArcOpToken::new(4, 1, 0, 0, 0, 0, 0, 0),
        ]);
        assert_eq!(program.describe(), "Rotate([0, 0, 0, 0, 0, 0, 0]) -> Fill([1, 0, 0, 0, 0, 0, 0])");
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let program = ArcProgram::from_tokens(vec![
            ArcOpToken::new(1, 0, 0, 0, 0, 0, 0, 0),
            ArcOpToken::new(4, 1, 0, 0, 0, 0, 0, 0),
        ]);
        let bytes = serialize_program(&program);
        let recovered = deserialize_program(&bytes).unwrap();
        assert_eq!(program, recovered);
    }
}
