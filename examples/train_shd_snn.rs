//! SHD surrogate-gradient SNN trainer — Block 3.
//!
//! Trains a recurrent LIF network end-to-end on Spiking Heidelberg Digits and
//! reports test accuracy against the 46.5% MLP baseline.
//!
//! Tunable via env vars (defaults are a fast smoke run):
//!   SHD_DATA  data/shd/shd.json | N_IN 70 | N_HID 128 | T 25
//!   EPOCHS 15 | LR 0.01 | N_TRAIN 500 | N_TEST 300 | SEED 42

use goldworm::audio::shd_loader::ShdDataset;
use goldworm::trainer::dataset::sample_to_tensor;
use goldworm::trainer::model::{softmax_cross_entropy, RnnLif};
use std::path::Path;
use std::time::Instant;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn predict(model: &RnnLif, x: &[Vec<f32>]) -> usize {
    let f = model.forward(x, false);
    let mut best = 0usize;
    for i in 1..f.logits.len() {
        if f.logits[i] > f.logits[best] {
            best = i;
        }
    }
    best
}

fn main() {
    let data_path = std::env::var("SHD_DATA").unwrap_or_else(|_| "data/shd/shd.json".to_string());
    let n_in: usize = env("N_IN", 70);
    let n_hid: usize = env("N_HID", 128);
    let t_steps: usize = env("T", 25);
    let epochs: usize = env("EPOCHS", 15);
    let lr: f32 = env("LR", 0.01);
    let n_train: usize = env("N_TRAIN", 500);
    let n_test: usize = env("N_TEST", 300);
    let seed: u64 = env("SEED", 42);

    let t0 = Instant::now();
    println!("Loading SHD from {data_path} ...");
    let ds = ShdDataset::from_json(Path::new(&data_path)).expect("failed to load SHD");
    println!(
        "loaded: {} train / {} test, {} neurons, {} ms, {} classes ({:.1}s)",
        ds.train.len(),
        ds.test.len(),
        ds.num_neurons,
        ds.duration_ms,
        ds.num_classes,
        t0.elapsed().as_secs_f32()
    );

    let n_train = n_train.min(ds.train.len());
    let n_test = n_test.min(ds.test.len());

    // Event → [T, n_in] → [T][n_in] steps.
    let to_steps = |sample: &goldworm::audio::shd_loader::ShdSample| -> Vec<Vec<f32>> {
        let flat = sample_to_tensor(sample, ds.num_neurons, ds.duration_ms, t_steps, n_in);
        (0..t_steps)
            .map(|t| flat[t * n_in..(t + 1) * n_in].to_vec())
            .collect()
    };

    println!("featurizing {} train + {} test samples ...", n_train, n_test);
    let train_x: Vec<Vec<Vec<f32>>> = ds.train[..n_train].iter().map(to_steps).collect();
    let train_y: Vec<usize> = ds.train[..n_train].iter().map(|s| s.label as usize).collect();
    let test_x: Vec<Vec<Vec<f32>>> = ds.test[..n_test].iter().map(to_steps).collect();
    let test_y: Vec<usize> = ds.test[..n_test].iter().map(|s| s.label as usize).collect();
    let feat_time = t0.elapsed().as_secs_f32();

    let mut model = RnnLif::new(n_in, n_hid, ds.num_classes, 0.8, 0.1, 10.0, seed);

    // Diagnostic: is the input actually driving spiking?
    {
        let f0 = model.forward(&train_x[0], false);
        let active_in: usize = train_x[0].iter().flatten().filter(|&&v| v > 0.0).count();
        let spiked: usize = f0.spikes.iter().flatten().filter(|&&s| s > 0.5).count();
        let vmax: f32 = f0.v_pre.iter().flatten().cloned().fold(0.0f32, f32::max);
        println!(
            "diag: input active {active_in}/{} | hidden spiked {spiked}/{} | max v_pre {vmax:.3}",
            t_steps * n_in,
            t_steps * n_hid,
        );
    }

    println!(
        "training: n_in={n_in} n_hid={n_hid} T={t_steps} epochs={epochs} lr={lr} (featurize {feat_time:.1}s)"
    );

    let mut best_acc = 0.0f32;
    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        for (x, &y) in train_x.iter().zip(&train_y) {
            total_loss += model.sgd_step(x, y, lr);
        }
        let mut correct = 0usize;
        for (x, &y) in test_x.iter().zip(&test_y) {
            if predict(&model, x) == y {
                correct += 1;
            }
        }
        let acc = correct as f32 / n_test as f32;
        if acc > best_acc {
            best_acc = acc;
        }
        println!(
            "epoch {epoch:>2}: loss {:.4} | test acc {:.2}% ({correct}/{n_test})",
            total_loss / n_train as f32,
            acc * 100.0
        );
    }

    let total = t0.elapsed().as_secs_f32();
    println!("---");
    println!("best test acc: {:.2}%", best_acc * 100.0);
    println!("baseline (MLP rate): 46.50%");
    println!("chance: {:.2}%", 100.0 / ds.num_classes as f32);
    println!("wall time: {total:.1}s");
}
