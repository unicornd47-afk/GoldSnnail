//! GoldWorm Vision-Semantic Training Loop
//!
//! Trains a PatchEncoder on synthetic image-label pairs, verifying that
//! visual tokens learn to cluster in the hyperbolic semantic space.
//!
//! Usage:
//!   cargo run --example train_vision

use goldworm::{
    PatchEncoder, ImagePatch, VisualToken, SemanticEncoder, PoincareBall,
    generate_test_image,
};
use std::time::Instant;

fn main() {
    println!("=== GoldWorm Vision-Semantic Training ===\n");

    // Labels: 3 semantic classes
    let vocab = vec!["bright".into(), "dark".into(), "noise".into()];
    let semantic = SemanticEncoder::new(vocab.clone(), 2);
    let mut enc = PatchEncoder::new(4, 2, 1.0).with_semantic(semantic.clone());

    // Generate synthetic training pairs: (image_pattern, label)
    let patterns = vec![
        ("bright_patch", "bright"),
        ("dark_patch", "dark"),
        ("noise", "noise"),
        ("gradient", "bright"),
        ("horizontal_stripes", "dark"),
        ("vertical_stripes", "dark"),
        ("checkerboard", "noise"),
    ];

    // Convert patterns to ImagePatches
    let mut training_pairs: Vec<(ImagePatch, String)> = Vec::new();
    for (pattern, label) in &patterns {
        let img = match *pattern {
            "bright_patch" => vec![0.9; 16],
            "dark_patch" => vec![0.1; 16],
            "noise" => (0..16).map(|i| ((i * 37) % 100) as f64 / 100.0).collect(),
            "gradient" => generate_test_image("gradient", 4, 4),
            "horizontal_stripes" => generate_test_image("horizontal_stripes", 4, 4),
            "vertical_stripes" => generate_test_image("vertical_stripes", 4, 4),
            "checkerboard" => generate_test_image("checkerboard", 4, 4),
            _ => vec![0.5; 16],
        };
        training_pairs.push((ImagePatch::new(4, 4, img), label.to_string()));
    }

    let epochs = 20;
    let mut history = Vec::with_capacity(epochs);

    println!("Training on {} image-label pairs for {} epochs...\n", training_pairs.len(), epochs);

    for epoch in 0..epochs {
        let start = Instant::now();
        let mut total_shift = 0.0;
        let mut total_dist = 0.0;

        for (patch, label) in &training_pairs {
            let mut token = VisualToken {
                patch: patch.clone(),
                embedding: enc.encode_patch(patch),
                hyperbolic: enc.to_hyperbolic(&enc.encode_patch(patch)).unwrap(),
                label: String::new(),
                salience: 1.0,
            };

            let before = token.hyperbolic.coords.clone();
            enc.bind_visual_semantic(&mut token, label).unwrap();
            let after = token.hyperbolic.coords.clone();

            let shift: f64 = before.iter().zip(&after).map(|(b, a)| (a - b).abs()).sum();
            total_shift += shift;

            // Distance to target label
            let label_emb = semantic.encode_token(label).unwrap();
            let label_h = enc.to_hyperbolic(&label_emb).unwrap();
            let ball = PoincareBall::new(1.0);
            total_dist += ball.distance(&token.hyperbolic, &label_h).unwrap();
        }

        let duration = start.elapsed();
        let avg_shift = total_shift / training_pairs.len() as f64;
        let avg_dist = total_dist / training_pairs.len() as f64;

        history.push((avg_shift, avg_dist));

        println!(
            "[Epoch {:>3}] avg_shift={:.6} | avg_dist_to_label={:.4} | {:>6.2?}",
            epoch + 1,
            avg_shift,
            avg_dist,
            duration
        );
    }

    // === GENERALIZATION TEST ===
    println!("\n=== Generalization Test ===");

    // Create a "bright" patch (should map to "bright")
    let bright_patch = ImagePatch::new(4, 4, vec![0.85; 16]);
    let q_bright = enc.encode_patch(&bright_patch);
    let h_bright = enc.to_hyperbolic(&q_bright).unwrap();

    // Create a "dark" patch (should map to "dark")
    let dark_patch = ImagePatch::new(4, 4, vec![0.15; 16]);
    let q_dark = enc.encode_patch(&dark_patch);
    let h_dark = enc.to_hyperbolic(&q_dark).unwrap();

    let bright_emb = semantic.encode_token("bright").unwrap();
    let bright_h = enc.to_hyperbolic(&bright_emb).unwrap();
    let dark_emb = semantic.encode_token("dark").unwrap();
    let dark_h = enc.to_hyperbolic(&dark_emb).unwrap();

    let ball = PoincareBall::new(1.0);
    let bright_to_bright = ball.distance(&h_bright, &bright_h).unwrap();
    let bright_to_dark = ball.distance(&h_bright, &dark_h).unwrap();
    let dark_to_bright = ball.distance(&h_dark, &bright_h).unwrap();
    let dark_to_dark = ball.distance(&h_dark, &dark_h).unwrap();

    println!(
        "Bright patch → 'bright' dist={:.4}, → 'dark' dist={:.4}",
        bright_to_bright, bright_to_dark
    );
    println!(
        "Dark patch → 'bright' dist={:.4}, → 'dark' dist={:.4}",
        dark_to_bright, dark_to_dark
    );

    if bright_to_bright < bright_to_dark && dark_to_dark < dark_to_bright {
        println!("✅ Generalization: patches map to correct labels");
    } else {
        println!("⚠️  Weak clustering — may need more epochs or richer features");
    }

    // Verify all tokens stay inside the Poincaré ball
    for (i, (patch, label)) in training_pairs.iter().enumerate() {
        let q = enc.encode_patch(patch);
        let h = enc.to_hyperbolic(&q).unwrap();
        assert!(
            h.euclidean_norm() < 1.0,
            "Training pair {i} ({label}) escaped ball: norm={}",
            h.euclidean_norm()
        );
    }
    println!("\n✅ All training tokens remain inside the Poincaré ball.");
}