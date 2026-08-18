//! Integration tests for the vision-semantic multimodal pipeline.
//!
//! Tests bridge visual patches to the existing semantic hyperbolic space,
//! verifying that visual tokens learn meaningful bindings to labels.

use goldworm::{
    PatchEncoder, ImagePatch, VisualToken, SemanticEncoder, PoincareBall,
    generate_test_image, HyperbolicPoint, Quaternion,
};
use ndarray::Array1;

#[test]
fn patch_encoder_extracts_correct_count() {
    let enc = PatchEncoder::new(4, 2, 1.0);
    let img = generate_test_image("gradient", 16, 16);
    let patches = enc.extract_patches(&img, 16, 16);
    assert_eq!(patches.len(), 16); // 16x16 image with 4x4 patches = 4x4 grid
}

#[test]
fn visual_token_hyperbolic_norm_inside_ball() {
    let enc = PatchEncoder::new(4, 2, 1.0);
    let img = generate_test_image("checkerboard", 8, 8);
    let tokens = enc.encode_image(&img, 8, 8);
    for token in &tokens {
        assert!(
            token.hyperbolic.euclidean_norm() < 1.0,
            "Visual token must lie inside Poincaré ball, got norm={}",
            token.hyperbolic.euclidean_norm()
        );
    }
}

#[test]
fn semantic_binding_moves_toward_label() {
    let vocab = vec![
        "cat".into(),
        "dog".into(),
        "tree".into(),
        "car".into(),
        "house".into(),
    ];
    let semantic = SemanticEncoder::new(vocab.clone(), 2);
    let mut enc = PatchEncoder::new(4, 2, 1.0).with_semantic(semantic.clone());

    // Two distinct patches
    let patch_a = ImagePatch::new(4, 4, vec![0.8; 16]); // bright
    let patch_b = ImagePatch::new(4, 4, vec![0.2; 16]); // dark

    let mut token_a = VisualToken {
        patch: patch_a.clone(),
        embedding: enc.encode_patch(&patch_a),
        hyperbolic: enc.to_hyperbolic(&enc.encode_patch(&patch_a)).unwrap(),
        label: String::new(),
        salience: 1.0,
    };
    let mut token_b = VisualToken {
        patch: patch_b.clone(),
        embedding: enc.encode_patch(&patch_b),
        hyperbolic: enc.to_hyperbolic(&enc.encode_patch(&patch_b)).unwrap(),
        label: String::new(),
        salience: 1.0,
    };

    let before_a = token_a.hyperbolic.coords.clone();
    let before_b = token_b.hyperbolic.coords.clone();

    enc.bind_visual_semantic(&mut token_a, "cat").unwrap();
    enc.bind_visual_semantic(&mut token_b, "tree").unwrap();

    let after_a = token_a.hyperbolic.coords.clone();
    let after_b = token_b.hyperbolic.coords.clone();

    // Both must have shifted
    let shift_a: f64 = before_a.iter().zip(&after_a).map(|(b, a)| (a - b).abs()).sum();
    let shift_b: f64 = before_b.iter().zip(&after_b).map(|(b, a)| (a - b).abs()).sum();
    assert!(shift_a > 1e-6, "Token A should shift toward label 'cat'");
    assert!(shift_b > 1e-6, "Token B should shift toward label 'tree'");
}

#[test]
fn repeated_binding_improves_clustering() {
    let vocab = vec!["cat".into(), "dog".into()];
    let semantic = SemanticEncoder::new(vocab.clone(), 2);
    let mut enc = PatchEncoder::new(4, 2, 1.0).with_semantic(semantic.clone());

    let patch = ImagePatch::new(4, 4, vec![0.6; 16]);
    let mut token = VisualToken {
        patch: patch.clone(),
        embedding: enc.encode_patch(&patch),
        hyperbolic: enc.to_hyperbolic(&enc.encode_patch(&patch)).unwrap(),
        label: String::new(),
        salience: 1.0,
    };

    // Repeated binding should converge
    for _ in 0..10 {
        enc.bind_visual_semantic(&mut token, "cat").unwrap();
    }

    let h = &token.hyperbolic;
    assert!(h.euclidean_norm() < 1.0, "Must stay inside ball after repeated binding");

    let cat_emb = semantic.encode_token("cat").unwrap();
    let cat_h = enc.to_hyperbolic(&cat_emb).unwrap();
    let ball = PoincareBall::new(1.0);
    let dist = ball.distance(h, &cat_h).unwrap();
    assert!(dist < 0.5, "After repeated binding, token should be close to 'cat' label");
}

#[test]
fn multimodal_generalization_unseen_patch() {
    // Create a training set of bright patches → "cat", dark patches → "dog"
    let vocab = vec!["cat".into(), "dog".into()];
    let semantic = SemanticEncoder::new(vocab.clone(), 2);
    let mut enc = PatchEncoder::new(4, 2, 1.0).with_semantic(semantic.clone());

    // Train on a few examples
    for _ in 0..5 {
        let bright = ImagePatch::new(4, 4, vec![0.9; 16]);
        let mut t_bright = VisualToken {
            patch: bright,
            embedding: Quaternion::new(0.0, 0.0, 0.0, 0.0),
            hyperbolic: HyperbolicPoint::new(Array1::zeros(2)).unwrap(),
            label: String::new(),
            salience: 1.0,
        };
        t_bright.embedding = enc.encode_patch(&t_bright.patch);
        t_bright.hyperbolic = enc.to_hyperbolic(&t_bright.embedding).unwrap();
        enc.bind_visual_semantic(&mut t_bright, "cat").unwrap();

        let dark = ImagePatch::new(4, 4, vec![0.1; 16]);
        let mut t_dark = VisualToken {
            patch: dark,
            embedding: Quaternion::new(0.0, 0.0, 0.0, 0.0),
            hyperbolic: HyperbolicPoint::new(Array1::zeros(2)).unwrap(),
            label: String::new(),
            salience: 1.0,
        };
        t_dark.embedding = enc.encode_patch(&t_dark.patch);
        t_dark.hyperbolic = enc.to_hyperbolic(&t_dark.embedding).unwrap();
        enc.bind_visual_semantic(&mut t_dark, "dog").unwrap();
    }

    // Test a new bright patch (unseen but similar to training)
    let new_bright = ImagePatch::new(4, 4, vec![0.85; 16]);
    let q_new = enc.encode_patch(&new_bright);
    let h_new = enc.to_hyperbolic(&q_new).unwrap();

    let cat_emb = semantic.encode_token("cat").unwrap();
    let cat_h = enc.to_hyperbolic(&cat_emb).unwrap();
    let dog_emb = semantic.encode_token("dog").unwrap();
    let dog_h = enc.to_hyperbolic(&dog_emb).unwrap();

    let ball = PoincareBall::new(1.0);
    let dist_cat = ball.distance(&h_new, &cat_h).unwrap();
    let dist_dog = ball.distance(&h_new, &dog_h).unwrap();

    assert!(
        dist_cat < dist_dog,
        "Bright unseen patch should be closer to 'cat' than 'dog': cat_dist={:.4}, dog_dist={:.4}",
        dist_cat, dist_dog
    );
}

#[test]
fn full_pipeline_image_to_semantic_spike() {
    // Simulate a full pipeline: image → patches → visual tokens → semantic labels → spikes
    let vocab = vec!["red".into(), "blue".into(), "green".into()];
    let semantic = SemanticEncoder::new(vocab.clone(), 3);
    let mut enc = PatchEncoder::new(8, 3, 1.0).with_semantic(semantic);

    let img = generate_test_image("gradient", 16, 16);
    let mut tokens = enc.encode_image(&img, 16, 16);

    let last_idx = tokens.len() - 1;
    enc.bind_visual_semantic(&mut tokens[0], "red").unwrap();
    enc.bind_visual_semantic(&mut tokens[last_idx], "blue").unwrap();

    assert_eq!(tokens[0].label, "red");
    assert_eq!(tokens[tokens.len() - 1].label, "blue");

    // All tokens must remain inside the hyperbolic ball
    for (i, t) in tokens.iter().enumerate() {
        assert!(
            t.hyperbolic.euclidean_norm() < 1.0,
            "Token {i} escaped Poincaré ball: norm={}",
            t.hyperbolic.euclidean_norm()
        );
    }
}
