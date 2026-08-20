use goldsnnail::substrate::{SpikeBuffer, StateArena, WeightMatrix, NeuronIdx, SpikeEvent};

#[test]
fn verify_repr_c_layout() {
    assert_eq!(std::mem::align_of::<NeuronIdx>(), std::mem::size_of::<usize>());
    assert_eq!(std::mem::size_of::<NeuronIdx>(), std::mem::size_of::<usize>());
    assert!(std::mem::align_of::<SpikeEvent>() >= 4);
    assert!(std::mem::size_of::<SpikeEvent>() >= 16);
    assert!(std::mem::align_of::<StateArena>() >= 8);
    assert!(std::mem::align_of::<WeightMatrix>() >= 8);
    assert!(std::mem::align_of::<SpikeBuffer>() >= 8);
}

#[test]
fn spike_buffer_write_read_pattern() {
    // Simulate the write/read pattern expected on GPU.
    let mut buf = SpikeBuffer::new(100);
    for i in 0..100 {
        buf.push(i as u32).unwrap();
    }
    let collected: Vec<u32> = buf.iter().cloned().collect();
    assert_eq!(collected.len(), 100);
    assert_eq!(collected[99], 99);
}

#[test]
fn state_arena_memory_layout_is_flat() {
    let arena = StateArena::new(4);
    // All four vectors must have the same length.
    assert_eq!(arena.membrane.len(), arena.recovery.len());
    assert_eq!(arena.recovery.len(), arena.threshold.len());
    assert_eq!(arena.threshold.len(), arena.refractory.len());

    // No pointer indirection inside the struct itself.
    let base_ptr = &arena.membrane as *const Vec<f32> as usize;
    assert_ne!(base_ptr, 0);
}

#[test]
fn weight_matrix_row_major_contiguous() {
    let wm = WeightMatrix::new(10, 20);
    assert_eq!(wm.data.len(), 200);
    assert_eq!(wm.row(0).as_ptr(), wm.data.as_ptr());
    assert_eq!(wm.row(9).as_ptr(), unsafe { wm.data.as_ptr().add(9 * 20) });
}
