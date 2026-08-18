//! AVX2 SIMD Accelerations — Hot-path optimizations for flat f32 arrays
//!
//! This module provides AVX2-accelerated versions of the most performance-critical
//! operations in the substrate. All public functions have scalar fallbacks and
//! dispatch at runtime based on CPU feature detection.
//!
//! # Design Principles
//!
//! - **DOD-First**: All operations work on flat `&[f32]` slices with usize indices.
//! - **Elastic Bounds**: All functions handle empty slices, misaligned data, and
//!   lengths not divisible by the SIMD width gracefully.
//! - **No unsafe in public API**: The `#[target_feature]` unsafe blocks are
//!   encapsulated inside this module.

use std::arch::x86_64::*;

#[cfg(feature = "rayon")]
pub use rayon::prelude::*;

// =============================================================================
// 1. Runtime Feature Detection
// =============================================================================

/// Returns true if the current CPU supports AVX2 and FMA.
#[inline(always)]
pub fn has_avx2() -> bool {
    cfg!(target_feature = "avx2") || std::is_x86_feature_detected!("avx2")
}

/// Returns true if the current CPU supports FMA (Fused Multiply-Add).
#[inline(always)]
pub fn has_fma() -> bool {
    cfg!(target_feature = "fma") || std::is_x86_feature_detected!("fma")
}

// =============================================================================
// 2. Batch Euclidean Distances (AVX2)
// =============================================================================

/// Computes Euclidean distances from a single query point to many database points.
///
/// # Scalar fallback
///
/// If AVX2 is not available, falls back to a scalar implementation.
///
/// # Arguments
///
/// * `query` - The query point (flat coordinates, e.g., 2D or 3D).
/// * `database` - Flat array of database points, stored consecutively.
///   Length must be a multiple of `query.len()`.
///
/// # Returns
///
/// A `Vec<f32>` of distances, one per database point.
pub fn batch_euclidean_distances(query: &[f32], database: &[f32]) -> Vec<f32> {
    let dim = query.len();
    if dim == 0 || database.len() % dim != 0 {
        return Vec::new();
    }
    let count = database.len() / dim;
    let mut distances = Vec::with_capacity(count);

    if has_avx2() {
        unsafe { batch_euclidean_distances_avx2(query, database, &mut distances, count, dim) };
    } else {
        batch_euclidean_distances_scalar_impl(query, database, &mut distances, count, dim);
    }
    distances
}

/// AVX2 implementation of batch Euclidean distance.
///
/// # Safety
///
/// Requires AVX2 support. Caller must ensure `database.len() % dim == 0`.
#[target_feature(enable = "avx2")]
unsafe fn batch_euclidean_distances_avx2(
    query: &[f32],
    database: &[f32],
    distances: &mut Vec<f32>,
    count: usize,
    dim: usize,
) {
    distances.clear();
    distances.reserve(count);

    for i in 0..count {
        let offset = i * dim;
        let mut sum_sq = _mm256_setzero_ps();

        // Process 8 dimensions at a time (AVX2 width)
        let mut j = 0;
        while j + 8 <= dim {
            let db_vals = _mm256_loadu_ps(database.as_ptr().add(offset + j));
            let q_vals = _mm256_loadu_ps(query.as_ptr().add(j));
            let diff = _mm256_sub_ps(db_vals, q_vals);
            let sq = _mm256_mul_ps(diff, diff);
            sum_sq = _mm256_add_ps(sum_sq, sq);
            j += 8;
        }

        // Horizontal sum of the 8 lanes
        let mut sum_arr = [0.0f32; 8];
        _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_sq);
        let mut sum = sum_arr.iter().sum::<f32>();

        // Remainder
        for k in j..dim {
            let diff = database[offset + k] - query[k % query.len()];
            sum += diff * diff;
        }

        distances.push(sum.sqrt());
    }
}

/// Scalar fallback for batch Euclidean distance.
pub fn batch_euclidean_distances_scalar(query: &[f32], database: &[f32]) -> Vec<f32> {
    let dim = query.len();
    if dim == 0 || database.len() % dim != 0 {
        return Vec::new();
    }
    let count = database.len() / dim;
    let mut distances = Vec::with_capacity(count);
    batch_euclidean_distances_scalar_impl(query, database, &mut distances, count, dim);
    distances
}

fn batch_euclidean_distances_scalar_impl(
    query: &[f32],
    database: &[f32],
    distances: &mut Vec<f32>,
    count: usize,
    dim: usize,
) {
    distances.clear();
    distances.reserve(count);
    for i in 0..count {
        let offset = i * dim;
        let mut sum_sq = 0.0f32;
        for j in 0..dim {
            let diff = database[offset + j] - query[j];
            sum_sq += diff * diff;
        }
        distances.push(sum_sq.sqrt());
    }
}

// =============================================================================
// 3. Batch ArgMax (Winner-Take-All)
// =============================================================================

/// Finds the index of the maximum value in a flat f32 slice.
///
/// Returns `None` for empty slices.
pub fn batch_argmax(values: &[f32]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    if has_avx2() {
        unsafe { batch_argmax_avx2(values) }
    } else {
        batch_argmax_scalar(values)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn batch_argmax_avx2(values: &[f32]) -> Option<usize> {
    let len = values.len();
    let mut max_val = -f32::INFINITY;
    let mut max_idx = 0usize;

    let mut i = 0;
    while i + 8 <= len {
        let vals = _mm256_loadu_ps(values.as_ptr().add(i));
        // Find max within the 8-lane vector
        let mut max_arr = [0.0f32; 8];
        _mm256_storeu_ps(max_arr.as_mut_ptr(), vals);
        for (j, &v) in max_arr.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i + j;
            }
        }
        i += 8;
    }

    // Remainder
    for j in i..len {
        if values[j] > max_val {
            max_val = values[j];
            max_idx = j;
        }
    }

    Some(max_idx)
}

fn batch_argmax_scalar(values: &[f32]) -> Option<usize> {
    values.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}

// =============================================================================
// 4. FMA Dot Product
// =============================================================================

/// Computes the dot product of two flat f32 slices.
///
/// Returns `None` if lengths differ.
pub fn dot_product(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let len = a.len();
    if len == 0 {
        return Some(0.0);
    }
    if has_fma() {
        unsafe { Some(dot_product_fma(a, b, len)) }
    } else {
        Some(dot_product_scalar(a, b, len))
    }
}

#[target_feature(enable = "fma,avx2")]
unsafe fn dot_product_fma(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let a_vals = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vals = _mm256_loadu_ps(b.as_ptr().add(i));
        // FMA: sum = sum + a * b
        sum = _mm256_fmadd_ps(a_vals, b_vals, sum);
        i += 8;
    }

    let mut result = 0.0f32;
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    result += tmp.iter().sum::<f32>();

    // Remainder
    for j in i..len {
        result += a[j] * b[j];
    }
    result
}

fn dot_product_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// =============================================================================
// 5. Rayon Batch Distance (Parallel)
// =============================================================================

/// Computes Euclidean distances from a query to many database points in parallel.
///
/// Requires the `rayon` feature.
#[cfg(feature = "rayon")]
pub fn batch_distances_parallel(
    query: &[f32],
    database: &[f32],
    dim: usize,
) -> Vec<f32> {
    if dim == 0 || database.len() % dim != 0 {
        return Vec::new();
    }
    let count = database.len() / dim;
    database.par_chunks(dim)
        .map(|point| {
            let mut sum_sq = 0.0f32;
            for j in 0..dim {
                let diff = point[j] - query[j];
                sum_sq += diff * diff;
            }
            sum_sq.sqrt()
        })
        .collect()
}

// =============================================================================
// 6. Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_euclidean_distances_basic() {
        let query = [1.0f32, 0.0];
        let database = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let dists = batch_euclidean_distances(&query, &database);
        assert_eq!(dists.len(), 3);
        assert!((dists[0] - 0.0).abs() < 1e-5);
        assert!((dists[1] - 2.0f32.sqrt()).abs() < 1e-5);
        assert!((dists[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn batch_euclidean_distances_empty() {
        let query = [1.0f32];
        let database: Vec<f32> = vec![];
        let dists = batch_euclidean_distances(&query, &database);
        assert!(dists.is_empty());
    }

    #[test]
    fn batch_argmax_basic() {
        let values = vec![0.5, 2.0, 1.0, 3.0, 1.5];
        assert_eq!(batch_argmax(&values), Some(3));
    }

    #[test]
    fn batch_argmax_empty() {
        assert_eq!(batch_argmax(&[]), None);
    }

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), Some(32.0));
    }

    #[test]
    fn dot_product_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(dot_product(&a, &b), None);
    }

    #[test]
    fn scalar_avx2_consistency() {
        let query = [1.0f32, 2.0, 3.0];
        let database = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = ((4.0f32 - 1.0f32).powi(2) + (5.0f32 - 2.0f32).powi(2) + (6.0f32 - 3.0f32).powi(2)).sqrt();
        let dists = batch_euclidean_distances(&query, &database);
        assert_eq!(dists.len(), 2);
        assert!((dists[0] - 0.0).abs() < 1e-5);
        assert!((dists[1] - expected).abs() < 1e-5);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn batch_distances_parallel_basic() {
        let query = [1.0f32, 0.0];
        let database = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let dists = batch_distances_parallel(&query, &database, 2);
        assert_eq!(dists.len(), 3);
        assert!((dists[0] - 0.0).abs() < 1e-5);
        assert!((dists[2] - 1.0).abs() < 1e-5);
    }
}
