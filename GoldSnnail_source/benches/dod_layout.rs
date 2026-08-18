use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use goldworm::substrate::{StateArena, WeightMatrix};

fn bench_state_arena_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("dod_layout");

    for &size in &[1_000, 10_000, 100_000] {
        let mut arena = StateArena::new(size);

        group.bench_with_input(BenchmarkId::new("membrane_update", size), &size, |b, _| {
            b.iter(|| {
                for i in 0..size {
                    let val = black_box(arena.membrane[i]);
                    arena.membrane[i] = val * 0.9 + 0.1;
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("recovery_update", size), &size, |b, _| {
            b.iter(|| {
                for i in 0..size {
                    let val = black_box(arena.recovery[i]);
                    arena.recovery[i] = val * 0.99;
                }
            });
        });
    }

    group.finish();
}

fn bench_weight_matrix_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("dod_weight_access");

    for &size in &[100, 1_000, 10_000] {
        let wm = WeightMatrix::new(size, size);

        group.bench_with_input(BenchmarkId::new("random_read", size), &size, |b, _| {
            b.iter(|| {
                for i in 0..size {
                    let _ = black_box(wm.get(i, i));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_state_arena_update, bench_weight_matrix_access);
criterion_main!(benches);
