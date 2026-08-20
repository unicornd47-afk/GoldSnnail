//! LongLongMoA (Mixture of Agents) Flat Index Table
//!
//! The routing decision reduces to a flat index lookup. No deep dispatch trees,
//! no boxed closures. The "LongLong" name indicates two-level routing:
//! a coarse expert selection (level 1) and a fine sub-expert pinning (level 2).
//!
//! # Architecture
//!
//! ```text
//! Token i → [score_i : f32 × E] → top-k expert indices → ExpertIdx[k]
//! ```
//!
//! Scores are kept in a flat 2-D array `[num_tokens × num_experts]`.
//! Top-k selection is O(E) with a partial-max scan — no heap sort, no
//! allocation in the hot path.

/// Flat mapping from input token index to expert index.
///
/// `table[i]` = the primary expert responsible for token `i`.
pub struct MoaIndexTable {
    /// Primary expert assignment per token. `table[i] = ExpertIdx`.
    pub table: Vec<usize>,

    /// Number of tokens this table covers.
    pub num_tokens: usize,

    /// Number of available experts `E`.
    pub num_experts: usize,
}

impl MoaIndexTable {
    /// Creates a zero-initialised table (all tokens → expert 0).
    pub fn new(num_tokens: usize, num_experts: usize) -> Self {
        assert!(num_experts > 0, "MoaIndexTable: num_experts must be > 0");
        Self {
            table: vec![0; num_tokens],
            num_tokens,
            num_experts,
        }
    }

    /// Returns the primary expert index for token `i`.
    #[inline(always)]
    pub fn expert_of(&self, token: usize) -> usize {
        self.table[token]
    }
}

// ============================================================================
// LongLongMoA — two-level routing with score matrix
// ============================================================================

/// Two-level MoA router with a flat score matrix.
///
/// Level 1: coarse expert selection via top-k scan over `scores`.
/// Level 2: sub-expert pinning (stored in `sub_table`).
///
/// All buffers are flat `Vec` — no nested `Vec<Vec<_>>`.
pub struct LongLongMoaRouter {
    /// Score matrix, row-major: `scores[t * num_experts + e]` = score for
    /// token `t` routing to expert `e`. Length = `num_tokens × num_experts`.
    pub scores: Vec<f32>,

    /// Primary top-1 expert assignment per token. Length = `num_tokens`.
    pub primary: Vec<usize>,

    /// Secondary (top-2) expert assignment per token. Length = `num_tokens`.
    pub secondary: Vec<usize>,

    /// Sub-expert index within the primary expert. Length = `num_tokens`.
    pub sub_table: Vec<usize>,

    /// Accumulated token load per expert. Length = `num_experts`.
    /// Useful for load-balancing auxiliary loss in training.
    pub expert_load: Vec<u32>,

    /// Number of tokens.
    pub num_tokens: usize,

    /// Number of experts `E`.
    pub num_experts: usize,
}

impl LongLongMoaRouter {
    /// Creates a router for `num_tokens` tokens and `num_experts` experts.
    ///
    /// All scores are initialised to 0.0; primary/secondary to expert 0.
    pub fn new(num_tokens: usize, num_experts: usize) -> Self {
        assert!(num_experts >= 2, "LongLongMoaRouter: need at least 2 experts");
        Self {
            scores: vec![0.0_f32; num_tokens * num_experts],
            primary: vec![0; num_tokens],
            secondary: vec![1; num_tokens],
            sub_table: vec![0; num_tokens],
            expert_load: vec![0; num_experts],
            num_tokens,
            num_experts,
        }
    }

    /// Re-routes all tokens based on the current `scores` matrix.
    ///
    /// Performs an in-place top-2 partial-max scan (O(N·E), no allocation).
    /// Expert loads are recomputed from scratch.
    pub fn route(&mut self) {
        let e = self.num_experts;
        self.expert_load.fill(0);

        for t in 0..self.num_tokens {
            let row = &self.scores[t * e..(t + 1) * e];

            // Top-2 partial scan.
            let (mut best1, mut best2) = (0usize, 1usize);
            for (i, &s) in row.iter().enumerate() {
                if s > row[best1] {
                    best2 = best1;
                    best1 = i;
                } else if i != best1 && s > row[best2] {
                    best2 = i;
                }
            }

            self.primary[t] = best1;
            self.secondary[t] = best2;
            self.expert_load[best1] = self.expert_load[best1].saturating_add(1);
        }
    }

    /// Applies softmax in-place to the score row for token `t`.
    ///
    /// Elastic: uses the numerically stable `max`-subtracted variant.
    /// Values that would overflow `exp` are clamped to the safety limit.
    pub fn softmax_row(&mut self, t: usize) {
        let e = self.num_experts;
        let row = &mut self.scores[t * e..(t + 1) * e];

        // Subtract max for numerical stability.
        let max_s = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_s = if max_s.is_finite() { max_s } else { 0.0 };

        let mut sum = 0.0_f32;
        for s in row.iter_mut() {
            *s = (*s - max_s).min(80.0).exp(); // clamp exponent to prevent overflow
            sum += *s;
        }
        let inv_sum = if sum > 1e-12 { 1.0 / sum } else { 1.0 / e as f32 };
        for s in row.iter_mut() {
            *s *= inv_sum;
        }
    }

    /// Returns the load-imbalance ratio: `max_load / mean_load`.
    ///
    /// Ideally close to 1.0. A high ratio indicates routing collapse.
    pub fn load_imbalance(&self) -> f32 {
        let total: u32 = self.expert_load.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let mean = total as f32 / self.num_experts as f32;
        let max = *self.expert_load.iter().max().unwrap_or(&0) as f32;
        max / mean.max(1e-6)
    }
}
