//! GoldSnnail PatchEncoder Pre-Training
//!
//! Pre-trains the PatchEncoder weights using contrastive learning directly
//! in hyperbolic space. Uses Poincaré distance instead of quaternion cosine.
//!
//! Usage:
//!   cargo run --example pretrain_encoder

use goldsnnail::{
    PatchEncoder, PoincareBall, HyperbolicPoint,
    EncoderTrainer, SeparationMetrics,
};
use ndarray::Array1;
use std::time::Instant;

fn main() {
    println!("=== GoldSnnail PatchEncoder Pre-Training (Hyperbolic) ===\n");

    let encoder = PatchEncoder::new(8, 8, 1.0);
    let mut trainer = EncoderTrainer::new(encoder, 0.03, 0.3);

    // Generate synthetic training images
    println!("Generating synthetic CIFAR-10 (1000 images, 10 classes)...");
    let images = generate_synthetic_cifar10_batch(1_000, None);
    println!("Training on {} images...\n", images.len());

    // Pre-Training Loop
    let epochs = 100;
    for epoch in 0..epochs {
        let start = Instant::now();
        let loss = trainer.train_epoch(&images);
        let duration = start.elapsed();

        if epoch % 10 == 0 {
            let sep = trainer.measure_separation(&images);
            println!(
                "[Epoch {:>3}] loss={:.4} | intra={:.4} | inter={:.4} | ratio={:.4}x | {:?}",
                epoch, loss, sep.avg_intra, sep.avg_inter, sep.ratio, duration
            );
        }
    }

    // Final Evaluation
    let final_sep = trainer.measure_separation(&images);
    println!(
        "\n=== Final Separation ===\n  Intra:  {:.6}\n  Inter:  {:.6}\n  Ratio:  {:.4}x",
        final_sep.avg_intra, final_sep.avg_inter, final_sep.ratio
    );

    if final_sep.ratio > 2.0 {
        println!("✅ Encoder discriminates classes effectively");
    } else {
        println!("⚠️  Weak discrimination — try more epochs or higher lr");
    }

    // Export for baby_agent
    export_encoder(&trainer, "encoder_pretrained.json");
    println!("\nExported to encoder_pretrained.json");
    println!("Run: cargo run --example baby_agent");
}

fn export_encoder(trainer: &EncoderTrainer, path: &str) {
    let proj_json: Vec<String> = trainer.encoder.weights.iter()
        .chain(&trainer.encoder.latent_proj)
        .map(|w| format!("{:.6}", w))
        .collect();
    
    let export_json = format!(
        "{{\"patch_size\":{},\"latent_dim\":{},\"weights\":[{}],\"latent_proj\":[{}]}}",
        trainer.encoder.patch_size,
        trainer.encoder.latent_dim,
        proj_json[..trainer.encoder.weights.len()].join(","),
        proj_json[trainer.encoder.weights.len()..].join(",")
    );
    
    std::fs::write(path, export_json).expect("Failed to write encoder_pretrained.json");
}

/// Generate highly distinct synthetic images
fn generate_distinct_image(pattern: &str, width: usize, height: usize) -> Vec<f64> {
    let mut image = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            image[idx] = match pattern {
                "all_white" => 0.9,
                "all_dark" => 0.1,
                "left_bright" => if x < width / 2 { 0.9 } else { 0.1 },
                "top_bright" => if y < height / 2 { 0.9 } else { 0.1 },
                _ => 0.5,
            };
        }
    }
    image
}

// Re-export from vision module for convenience
use goldsnnail::vision::generate_synthetic_cifar10_batch;
