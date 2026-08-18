use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use goldworm::RSTDP;

fn bench_stdp_compute(c: &mut Criterion) {
    let stdp = RSTDP::new(0.01, 20.0, -1.0);

    c.bench_function("stdp_compute_potentiation", |b| {
        b.iter(|| {
            let dw = stdp.compute(
                black_box(1.0),
                black_box(0.0),
                black_box(5.0),
                black_box(0.1),
                black_box(0.11),
            );
            black_box(dw);
        });
    });

    c.bench_function("stdp_compute_depression", |b| {
        b.iter(|| {
            let dw = stdp.compute(
                black_box(-1.0),
                black_box(0.0),
                black_box(5.0),
                black_box(0.1),
                black_box(0.11),
            );
            black_box(dw);
        });
    });
}

fn bench_weight_update(c: &mut Criterion) {
    let stdp = RSTDP::new(0.01, 20.0, -1.0);
    let mut group = c.benchmark_group("stdp_weight_update");

    for &n_pre in &[100, 1_000, 10_000] {
        let mut weights = vec![0.0f64; n_pre];
        let pre_spikes: Vec<usize> = (0..n_pre).collect();
        let pre_times: Vec<f64> = (0..n_pre).map(|i| (i as f64) * 0.01).collect();
        let pre_embeds: Vec<f32> = (0..n_pre).map(|i| (i as f32) * 0.001).collect();
        let post_embed: f32 = 0.1;

        group.bench_with_input(
            BenchmarkId::new("batch_update", n_pre),
            &n_pre,
            |b, _| {
                b.iter(|| {
                    stdp.update_weights(
                        black_box(&mut weights),
                        black_box(&pre_spikes),
                        black_box(&pre_times),
                        black_box(5.0),
                        black_box(&pre_embeds),
                        black_box(post_embed),
                        black_box(1.0),
                    );
                    black_box(&weights);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_stdp_compute, bench_weight_update);
criterion_main!(benches);
