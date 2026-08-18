use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use goldworm::{HyperbolicPoint, PoincareBall};
use ndarray::array;

fn bench_hyperbolic_distance(c: &mut Criterion) {
    let ball = PoincareBall::new(1.0);
    let p = HyperbolicPoint::new(array![0.1, 0.2, 0.0, 0.0]).unwrap();
    let q = HyperbolicPoint::new(array![0.3, -0.1, 0.05, 0.0]).unwrap();
    
    c.bench_function("hyperbolic_distance_4d", |b| {
        b.iter(|| {
            let d = ball.distance(black_box(&p), black_box(&q)).unwrap();
            black_box(d);
        });
    });
}

fn bench_exp_map(c: &mut Criterion) {
    let ball = PoincareBall::new(1.0);
    let base = HyperbolicPoint::new(array![0.1, 0.0]).unwrap();
    let tangent = array![0.05, 0.02];
    
    c.bench_function("exp_map_2d", |b| {
        b.iter(|| {
            let result = ball.exp_map(black_box(&base), black_box(&tangent)).unwrap();
            black_box(result);
        });
    });
}

fn bench_distance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperbolic_distance_scaling");
    let ball = PoincareBall::new(1.0);
    
    for &dim in &[2, 4, 8, 16, 32] {
        let coords_p = ndarray::Array1::from_vec((0..dim).map(|i| (i as f64 * 0.01).sin() * 0.1).collect());
        let coords_q = ndarray::Array1::from_vec((0..dim).map(|i| (i as f64 * 0.02).cos() * 0.1).collect());
        let p = HyperbolicPoint::new(coords_p).unwrap();
        let q = HyperbolicPoint::new(coords_q).unwrap();
        
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                let d = ball.distance(black_box(&p), black_box(&q)).unwrap();
                black_box(d);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hyperbolic_distance, bench_exp_map, bench_distance_scaling);
criterion_main!(benches);
