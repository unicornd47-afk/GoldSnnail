use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use goldworm::{QLIFNeuron, Quaternion, WorkingMemory};

fn bench_qlif_single_step(c: &mut Criterion) {
    let mut neuron = QLIFNeuron::new(0.9, 1.0);
    let input = Quaternion::new(0.5, 0.0, 0.0, 0.0);
    
    c.bench_function("qlif_step_no_spike", |b| {
        b.iter(|| {
            let _ = neuron.step(black_box(&input), 1.0, 0.0);
            black_box(&neuron);
        });
    });
    
    let strong_input = Quaternion::new(5.0, 0.0, 0.0, 0.0);
    c.bench_function("qlif_step_with_spike", |b| {
        b.iter(|| {
            neuron.reset();
            let spike = neuron.step(black_box(&strong_input), 1.0, 0.0);
            black_box(spike);
        });
    });
}

fn bench_working_memory_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("working_memory");
    
    for &size in &[8, 64, 256, 1024] {
        let mut mem = WorkingMemory::new(size, 0.9, 1.0);
        let inputs = vec![Quaternion::new(0.3, 0.1, 0.0, 0.0); size];
        
        group.bench_with_input(BenchmarkId::new("step", size), &size, |b, _| {
            b.iter(|| {
                let spikes = mem.step(black_box(&inputs), 1.0, 0.0);
                black_box(spikes);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qlif_single_step, bench_working_memory_step);
criterion_main!(benches);
