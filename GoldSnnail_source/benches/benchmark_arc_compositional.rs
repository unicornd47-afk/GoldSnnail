//! Criterion benchmark for the ARC compositional solver.
//!
//! Measures solve rate and latency on ARC-AGI-1 training tasks.
//! Outputs are saved to `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use goldworm::arc_search::{search_program, SearchConfig};
use goldworm::vision::ArcDataset;

fn benchmark_arc_compositional(c: &mut Criterion) {
    let dataset = ArcDataset::load_from_directory("data/arc-agi-repo/data/training")
        .expect("Failed to load ARC dataset");

    let tasks: Vec<_> = dataset.tasks.into_iter().take(100).collect();

    c.bench_function("arc_compositional_search_depth3", |b| {
        b.iter(|| {
            let mut solved = 0;
            for task in black_box(&tasks) {
                let result = search_program(task, SearchConfig::default());
                if result.program.is_some() {
                    solved += 1;
                }
            }
            solved
        });
    });
}

criterion_group!(benches, benchmark_arc_compositional);
criterion_main!(benches);
