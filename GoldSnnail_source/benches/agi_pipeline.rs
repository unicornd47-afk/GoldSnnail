use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::array;
use goldworm::{
    GeometricBottleneck, HyperbolicPoint, Quaternion, QuaternionAttention,
    RLAgent, RSTDP, SpikeBuffer, StateVector, WorldModel, WorkingMemory,
};

fn bench_pipeline_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("agi_pipeline");
    
    for &neurons in &[16, 64, 256] {
        group.bench_with_input(BenchmarkId::new("full_step", neurons), &neurons, |b, _| {
            let attn = QuaternionAttention::new();
            let mut mem = WorkingMemory::new(neurons, 0.9, 1.0);
            let mut bn = GeometricBottleneck::new(neurons, 4, 0.05, 1.0);
            let mut wm = WorldModel::new(4, 8, 1.0);
            let stdp = RSTDP::new(0.01, 20.0, 1.0);
            
            let sensor = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0)];
            let phases = vec![Quaternion::new(1.0, 0.0, 0.0, 0.0); neurons];
            
            b.iter(|| {
                let attended = attn.forward(&sensor, &sensor, &sensor);
                let spikes = mem.step(black_box(&attended), 1.0, 0.0);
                
                let latent = bn.compress(
                    black_box(&SpikeBuffer::new(neurons)),
                    black_box(&phases),
                ).unwrap();
                
                let predicted = if let Some(ref l) = latent {
                    wm.predict(black_box(l)).unwrap()
                } else {
                    HyperbolicPoint::new(array![0.0, 0.0, 0.0, 0.0]).unwrap()
                };
                
                let state = StateVector::new(
                    latent.unwrap_or_else(|| HyperbolicPoint::new(ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0])).unwrap()),
                    &spikes,
                );
                let mut agent = RLAgent::new(state.dim(), 0.9);
                let action = agent.act(black_box(&state));
                
                black_box(action);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pipeline_throughput);
criterion_main!(benches);
