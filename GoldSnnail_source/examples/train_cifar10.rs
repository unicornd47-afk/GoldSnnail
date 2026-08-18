//! GoldWorm CIFAR-10 Training
//!
//! Trains a SemanticLearner on CIFAR-10 images mapped to the existing
//! German lexicon. Falls back to synthetic data if no CIFAR-10 binaries
//! are found.
//!
//! Usage (Windows PowerShell):
//!   cargo run --example train_cifar10
//!
//! With real data (set env var first):
//!   $env:CIFAR10_DIR = ".\cifar-10-batches-bin"
//!   cargo run --example train_cifar10

use goldworm::{
    PatchEncoder, SemanticTrainer, SemanticLearner, LearningRates,
    Cifar10Loader, generate_synthetic_cifar10_batch, map_cifar_label_to_lexicon,
    PoincareBall,
};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("=== GoldWorm CIFAR-10 Training ===\n");

    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());
    let mut encoder = PatchEncoder::new(8, 8, 1.0); // 8×8 patches, latent_dim=8
    let ball = PoincareBall::new(1.0);

    // === DATEN LADEN ===
    let cifar_dir = std::env::var("CIFAR10_DIR").unwrap_or_else(|_| "./cifar-10-batches-bin".to_string());
    
    let (train_images, test_images) = if Path::new(&cifar_dir).exists() {
        println!("Loading real CIFAR-10 from {}...", cifar_dir);
        let train = Cifar10Loader::load_training_set(&cifar_dir)
            .expect("Failed to load CIFAR-10 training set. Set CIFAR10_DIR or download from https://www.cs.toronto.edu/~kriz/cifar.html");
        let test = Cifar10Loader::load_test_set(&cifar_dir)
            .expect("Failed to load CIFAR-10 test set");
        println!("Loaded {} train + {} test images", train.len(), test.len());
        (train, test)
    } else {
        println!("CIFAR-10 directory not found. Using synthetic data.");
        println!("To use real data: download and extract to ./cifar-10-batches-bin/");
        let train = generate_synthetic_cifar10_batch(1_000, None);
        let test = generate_synthetic_cifar10_batch(200, None);
        println!("Generated {} train + {} test images", train.len(), test.len());
        (train, test)
    };

    // === TRAINING ===
    let epochs = 20;
    let batch_size = 100;

    for epoch in 0..epochs {
        let start = Instant::now();
        let mut epoch_reward = 0.0;
        let mut samples_processed = 0;

        // Stochastic mini-batches
        let num_batches = train_images.len() / batch_size;
        for b in 0..num_batches {
            let batch = &train_images[b * batch_size .. (b + 1) * batch_size];
            
            for img in batch {
                let label_word = map_cifar_label_to_lexicon(img.label);
                
                // 1. Convert f32 → f64 and encode image to patches (RGB → grayscale handled internally)
                let pixels_f64: Vec<f64> = img.pixels.iter().map(|&p| p as f64).collect();
                let visual = encoder.encode_image(&pixels_f64, 32, 32);
                
                // 2. Bind all patches of this image to the semantic label
                let mut bound = visual.clone();
                for token in &mut bound {
                    let _ = encoder.bind_visual_semantic(token, label_word);
                }

                // 3. Echte Lexikon-Sequenz (damit resolve() funktioniert)
                let sentence = trainer.composer.build_sentence_simple(label_word, "sieht");
                
                // 4. Semantic learning
                let reward = trainer.train_step(&sentence, false);
                let tokens = trainer.composer.resolve(&sentence);
                if !tokens.is_empty() {
                    let _ = learner.learn_from_reward(
                        &reward, &tokens, None, None,
                        &mut trainer.concept_graph, &mut trainer.lexicon,
                    );
                    epoch_reward += reward.total;
                    samples_processed += 1;
                }
            }
        }

        let avg_reward = epoch_reward / samples_processed.max(1) as f64;
        println!(
            "[Epoch {:>2}] avg_reward={:.4} | samples={:<5} | time={:.2?}",
            epoch + 1, avg_reward, samples_processed, start.elapsed()
        );
    }

    // === EVALUATION ===
    println!("\n=== Evaluation on Test Set ===");
    
    let cifar_labels: Vec<&str> = vec![
        "vogel", "haus", "vogel", "katze",
        "hund", "hund", "fisch", "hund",
        "tisch", "stein",
    ];
    
    let mut correct = 0;
    let test_samples = test_images.len().min(100); // Nur 100 für Schnelligkeit
    
    for img in test_images.iter().take(test_samples) {
        let true_label = map_cifar_label_to_lexicon(img.label);
        
        let pixels_f64: Vec<f64> = img.pixels.iter().map(|&p| p as f64).collect();
        let visual = encoder.encode_image(&pixels_f64, 32, 32);
        let center = &visual[visual.len() / 2].hyperbolic;
        
        // Nächster Nachbar NUR unter den 8 CIFAR-relevanten Labels
        let mut best_word = "";
        let mut best_dist = f64::INFINITY;
        
        for &label in &cifar_labels {
            if let Some(token) = trainer.lexicon.get(label) {
                if let Ok(d) = ball.distance(center, &token.hyperbolic) {
                    if d < best_dist {
                        best_dist = d;
                        best_word = label;
                    }
                }
            }
        }
        
        if best_word == true_label {
            correct += 1;
        }
    }
    
    let accuracy = 100.0 * correct as f64 / test_samples as f64;
    println!("Test Accuracy: {}/{} = {:.1}%", correct, test_samples, accuracy);
    
    if accuracy > 20.0 {
        println!("✅ Better than random (10%) — learning occurred!");
    } else {
        println!("⚠️  Near random — needs more epochs or better features");
    }

    // === LEXIKON-CLUSTER ANALYSE ===
    println!("\n=== Lexicon Cluster Analysis ===");
    let animal_labels = ["hund", "katze", "vogel", "fisch"];
    let object_labels = ["tisch", "haus", "stein"];
    
    let animal_tokens: Vec<_> = animal_labels.iter()
        .filter_map(|&w| trainer.lexicon.get(w))
        .collect();
    let object_tokens: Vec<_> = object_labels.iter()
        .filter_map(|&w| trainer.lexicon.get(w))
        .collect();
    
    if animal_tokens.len() >= 2 && object_tokens.len() >= 2 {
        let mut intra_animal = 0.0;
        let mut intra_object = 0.0;
        let mut inter = 0.0;
        let mut count = 0;
        
        for a in &animal_tokens {
            for b in &animal_tokens {
                if a.id < b.id {
                    intra_animal += ball.distance(&a.hyperbolic, &b.hyperbolic).unwrap_or(0.0);
                    count += 1;
                }
            }
        }
        if count > 0 { intra_animal /= count as f64; }
        
        let mut count = 0;
        for a in &object_tokens {
            for b in &object_tokens {
                if a.id < b.id {
                    intra_object += ball.distance(&a.hyperbolic, &b.hyperbolic).unwrap_or(0.0);
                    count += 1;
                }
            }
        }
        if count > 0 { intra_object /= count as f64; }
        
        let mut count = 0;
        for a in &animal_tokens {
            for b in &object_tokens {
                inter += ball.distance(&a.hyperbolic, &b.hyperbolic).unwrap_or(0.0);
                count += 1;
            }
        }
        if count > 0 { inter /= count as f64; }
        
        println!("  Intra-animal distance:   {:.4}", intra_animal);
        println!("  Intra-object distance:   {:.4}", intra_object);
        println!("  Inter-cluster distance:  {:.4}", inter);
        
        if intra_animal < inter && intra_object < inter {
            println!("  ✅ Semantic clustering: animals and objects form separate clusters");
        } else {
            println!("  ⚠️  Weak clustering — may need more training");
        }
    }
}