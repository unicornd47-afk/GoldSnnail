use goldsnnail::substrate::WeightMatrix;
use proptest::prelude::*;

proptest! {
    #[test]
    fn weight_matrix_index_in_bounds(
        rows in 1usize..500,
        cols in 1usize..500,
        row in 0usize..500,
        col in 0usize..500,
    ) {
        prop_assume!(row < rows && col < cols);
        let mat = WeightMatrix::new(rows, cols);
        let idx = mat.index(row, col);
        prop_assert!(idx < rows * cols, "index {} out of bounds for {}x{}", idx, rows, cols);
        prop_assert_eq!(mat.get(row, col), 0.0);
    }

    #[test]
    fn weight_matrix_row_slice_len(rows in 1usize..100, cols in 1usize..100) {
        let mat = WeightMatrix::new(rows, cols);
        prop_assert_eq!(mat.row(0).len(), cols);
        prop_assert_eq!(mat.row(rows - 1).len(), cols);
    }

    #[test]
    fn state_arena_extend_preserves_lengths(capacity in 1usize..1000, additional in 1usize..100) {
        let mut arena = goldsnnail::substrate::StateArena::new(capacity);
        arena.extend(additional);
        prop_assert_eq!(arena.membrane.len(), capacity + additional);
        prop_assert_eq!(arena.recovery.len(), capacity + additional);
        prop_assert_eq!(arena.threshold.len(), capacity + additional);
        prop_assert_eq!(arena.refractory.len(), capacity + additional);
    }
}
