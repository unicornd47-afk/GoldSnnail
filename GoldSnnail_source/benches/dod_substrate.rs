use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use goldworm::StateArena;

fn bench_membrane_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("dod_membrane_update");
    
    for &size in &[1_000, 10_000, 100_000, 1_000_000] {
        let mut arena = StateArena::new(size);
        for i in 0..size {
            arena.membrane[i] = 0.5;
            arena.refractory[i] = 0;
        }
        
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |b, _| {
            b.iter(|| {
                for i in 0..size {
                    arena.membrane[i] = black_box(arena.membrane[i]) * 0.9 + 0.1;
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_membrane_update);
criterion_main!(benches);
