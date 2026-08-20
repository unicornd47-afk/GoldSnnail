//! Integration tests for CIFAR-10 vision pipeline.
//!
//! Tests the CIFAR-10 loader, synthetic generator, and PatchEncoder
//! on 32×32 RGB images.

use goldsnnail::{
    generate_synthetic_cifar10_batch, PatchEncoder, map_cifar_label_to_lexicon,
};

#[test]
fn synthetic_cifar10_through_encoder() {
    let batch = generate_synthetic_cifar10_batch(50, None);
    let encoder = PatchEncoder::new(8, 8, 1.0);
    
    let mut encoded = 0;
    for img in &batch {
        let pixels_f64: Vec<f64> = img.pixels.iter().map(|&p| p as f64).collect();
        let tokens = encoder.encode_image(&pixels_f64, 32, 32);
        assert_eq!(tokens.len(), 16); // 4×4 Grid bei 32×32, patch=8, stride=8
        assert!(tokens.iter().all(|t| t.hyperbolic.euclidean_norm() < 1.0));
        encoded += 1;
    }
    assert_eq!(encoded, 50);
}

#[test]
fn cifar_label_mapping_exists_for_all() {
    for i in 0..10u8 {
        let word = map_cifar_label_to_lexicon(i);
        assert_ne!(word, "???", "Label {} should map to a real word", i);
    }
}

#[test]
fn different_cifar_classes_different_embeddings() {
    let encoder = PatchEncoder::new(8, 8, 1.0);
    let batch = generate_synthetic_cifar10_batch(20, Some(&[2, 2, 0, 0, 0, 0, 0, 0, 0, 0]));
    // 2× airplane (label 0), 2× automobile (label 1)
    
    let pixels_plane: Vec<f64> = batch[0].pixels.iter().map(|&p| p as f64).collect();
    let plane = encoder.encode_image(&pixels_plane, 32, 32);
    
    let pixels_car: Vec<f64> = batch[2].pixels.iter().map(|&p| p as f64).collect();
    let car = encoder.encode_image(&pixels_car, 32, 32);
    
    // Zumindest ein Paar sollte sich unterscheiden
    let mut diff = false;
    for (a, b) in plane.iter().zip(car.iter()) {
        if (a.embedding.w - b.embedding.w).abs() > 1e-6 {
            diff = true;
            break;
        }
    }
    assert!(diff, "Different CIFAR classes should produce different embeddings");
}

#[test]
fn synthetic_batch_label_coverage() {
    let batch = generate_synthetic_cifar10_batch(100, None);
    let mut counts = [0usize; 10];
    for img in &batch {
        counts[img.label as usize] += 1;
    }
    // Each class should have at least some samples
    for (i, &c) in counts.iter().enumerate() {
        assert!(c > 0, "Class {} has no samples", i);
    }
}