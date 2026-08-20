//! GoldSnnail CIFAR-10 Baby Agent — Integrated Vision + Semantic Learning
//!
//! Complete pipeline demonstrating actual learning:
//! 1. Load/generate CIFAR-10 images (32x32 RGB)
//! 2. Pre-train PatchEncoder with contrastive loss in hyperbolic space
//! 3. Map visual embeddings to German lexicon words
//! 4. Train semantic associations
//! 5. Measure separation improvement
//!
//! Usage:
//!   cargo run --example cifar_baby

use goldsnnail::{
     PatchEncoder, SemanticTrainer, SemanticLearner, LearningRates, EpochMetrics,
     PoincareBall, HyperbolicPoint, LexiconToken, TokenClass, Quaternion, CifarImage,
     EncoderTrainer, map_cifar_label_to_lexicon, generate_synthetic_cifar10_batch, Cifar10Loader,
};
use goldsnnail::baby::{InfomaxReward, TransitionalLearner};
use ndarray::Array1;
use rand::prelude::IteratorRandom;
use rand::Rng;
use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

static RUNNING: AtomicBool = AtomicBool::new(true);

fn main() {
    println!("=== GoldSnnail CIFAR-10 Baby Agent ===\n");
    println!("1. Load CIFAR-10 images (synthetic for demo)");
    println!("2. Pre-train PatchEncoder with contrastive loss");
    println!("3. Map visuals to German lexicon");
    println!("4. Train semantic associations");
    println!("5. Measure separation improvement\n");
    println!("Press Ctrl+C to stop.\n");

    {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
        ctrlc::set_handler(move || {
            println!("\n\n🛑 Ctrl+C received — shutting down gracefully...");
            RUNNING.store(false, Ordering::SeqCst);
            r.store(false, Ordering::SeqCst);
        }).expect("Error setting Ctrl+C handler");

        let r2 = running.clone();
        std::thread::spawn(move || {
            while r2.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
            }
        });
    }

    let mut trainer = SemanticTrainer::new(1.0);
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());
    let ball = PoincareBall::new(1.0);
    let mut rng = rand::thread_rng();

    let infomax = InfomaxReward::new(10);
    let mut transitional = TransitionalLearner::new();

    // Try real CIFAR-10 first, fall back to synthetic
    let images = match Cifar10Loader::load_training_set("cifar-10-batches-bin") {
        Ok(real_images) => {
            println!("Loaded {} real CIFAR-10 images ({}x{} RGB)", real_images.len(), 32, 32);
            real_images
        }
        Err(_) => {
            println!("Real CIFAR-10 not found, using synthetic images...");
            generate_synthetic_cifar10_batch(200, None)
        }
    };

    println!("\nPre-training PatchEncoder with contrastive loss...");
    let mut encoder = PatchEncoder::new(8, 2, 1.0);
    let mut pretrainer = EncoderTrainer::new(encoder.clone(), 0.05, 0.2);

    let before_sep = pretrainer.measure_separation(&images);
    println!("Before pre-training: intra={:.4}, inter={:.4}, ratio={:.2}x",
        before_sep.avg_intra, before_sep.avg_inter, before_sep.ratio);

    for epoch in 1..=10 {
        let loss = pretrainer.train_epoch(&images);
        if epoch % 2 == 0 {
            println!("  Pretrain epoch {:>2}: loss={:.4}", epoch, loss);
        }
    }

    encoder = pretrainer.encoder.clone();

    let after_sep = pretrainer.measure_separation(&images);
    println!("After pre-training:  intra={:.4}, inter={:.4}, ratio={:.2}x",
        after_sep.avg_intra, after_sep.avg_inter, after_sep.ratio);
    println!("Separation improvement: {:.2}x\n", after_sep.ratio / before_sep.ratio);

    println!("Building lexicon from CIFAR-10 labels...");
    trainer.lexicon.tokens.clear();
    trainer.lexicon.word_index.clear();
    trainer.lexicon.class_index.clear();

    let mut class_words: HashMap<u8, String> = HashMap::new();
    for label in 0..10u8 {
        let word = map_cifar_label_to_lexicon(label);
        class_words.insert(label, word.to_string());
    }

    let mut added_words: HashMap<String, usize> = HashMap::new();
    let mut word_idx = 0;
    for (_label, word) in &class_words {
        if !added_words.contains_key(word) {
            let angle = (word_idx as f64) * 2.0 * std::f64::consts::PI / 6.0;
            let r = 0.5;
            let coords = Array1::from_vec(vec![r * angle.cos(), r * angle.sin()]);
            let q = Quaternion::new(
                coords[0] as f32, coords[1] as f32, 0.0, 0.0
            ).normalize();

            let id = trainer.lexicon.tokens.len();
            trainer.lexicon.tokens.push(LexiconToken {
                id,
                surface: word.to_string(),
                class: TokenClass::NounConcrete,
                embedding: q,
                hyperbolic: HyperbolicPoint::new(coords).unwrap(),
                salience: 0.5,
            });
            trainer.lexicon.word_index.insert(word.to_string(), id);
            added_words.insert(word.to_string(), id);
            word_idx += 1;
        }
    }
    println!("Lexicon: {} unique words for 10 CIFAR-10 classes", added_words.len());

    println!("\nTraining semantic associations...");
    let mut epoch: u64 = 0;
    let mut history: Vec<EpochMetrics> = Vec::with_capacity(1000);

    // Initial export + open browser (single-command startup)
    let _ = export_cifar_state(&trainer, &encoder, &images, &ball, &history, "docs/src/development/cifar_baby_state.json");
    println!("📊 Initial state exported.");
    open_browser();
    println!("🌐 Browser opened. Live visualization running at docs/src/development/cifar_baby.html\n");
    let mut last_export = Instant::now();
    let mut last_used: HashMap<String, u64> = HashMap::new();

    println!("CIFAR-10 Baby Agent started.\n");

    while RUNNING.load(Ordering::SeqCst) {
        epoch += 1;
        let mut total_reward = 0.0;
        let mut correct_guesses = 0;
        let mut total_guesses = 0;
        let mut curiosity_rewards = Vec::new();

        let batch_size = 20;
        let batch: Vec<_> = images.iter()
            .choose_multiple(&mut rng, batch_size.min(images.len()));

        for img in &batch {
            let label = img.label;
            let expected_word = map_cifar_label_to_lexicon(label);

            let pixels_f64: Vec<f64> = img.pixels.iter().map(|&x| x as f64).collect();
            let visual = encoder.encode_image(&pixels_f64, 32, 32);
            if visual.is_empty() { continue; }

            let mut cx = 0.0;
            let mut cy = 0.0;
            for t in &visual {
                cx += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                cy += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
            }
            let n = visual.len() as f64;
            cx /= n;
            cy /= n;

            let candidates: Vec<String> = trainer.lexicon.tokens.iter()
                .filter(|t| t.class != TokenClass::Noise)
                .map(|t| t.surface.clone())
                .collect();

            if candidates.is_empty() { continue; }

            // === SPEAKER: Image-aware epsilon-greedy ===
            let spoken_word = if rng.r#gen::<f64>() < 0.2 {
                // Explore: random word
                let idx = rng.r#gen::<usize>() % candidates.len();
                candidates[idx].clone()
            } else {
                // Exploit: word closest to current image centroid
                let mut best_word = &candidates[0];
                let mut best_dist = f64::INFINITY;
                for word in &candidates {
                    if let Some(token) = trainer.lexicon.get(word) {
                        let word_hp = HyperbolicPoint::new(
                            Array1::from_vec(vec![
                                token.hyperbolic.coords.get(0).copied().unwrap_or(0.0),
                                token.hyperbolic.coords.get(1).copied().unwrap_or(0.0),
                            ])
                        ).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] });
                        let img_hp = HyperbolicPoint::new(Array1::from_vec(vec![cx, cy]))
                            .unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] });
                        if let Ok(d) = ball.distance(&word_hp, &img_hp) {
                            if d < best_dist {
                                best_dist = d;
                                best_word = word;
                            }
                        }
                    }
                }
                best_word.clone()
            };
            last_used.insert(spoken_word.clone(), epoch);

            let spoken_word_hp = if let Some(token) = trainer.lexicon.get(&spoken_word) {
                HyperbolicPoint::new(Array1::from_vec(vec![
                    token.hyperbolic.coords.get(0).copied().unwrap_or(0.0),
                    token.hyperbolic.coords.get(1).copied().unwrap_or(0.0),
                ])).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] })
            } else {
                HyperbolicPoint { coords: vec![0.0, 0.0] }
            };
            let img_hp = HyperbolicPoint::new(Array1::from_vec(vec![cx, cy]))
                .unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] });
            let curiosity = if let Ok(d) = ball.distance(&spoken_word_hp, &img_hp) {
                (1.0 - d).max(0.0)
            } else {
                0.0
            };
            curiosity_rewards.push(curiosity);

            let guess_correct;
            let mut best_guess_label = 255u8;
            let mut best_guess_dist = f64::INFINITY;

            for other_img in batch.iter().choose_multiple(&mut rng, 5) {
                let other_pixels_f64: Vec<f64> = other_img.pixels.iter().map(|&x| x as f64).collect();
                let other_visual = encoder.encode_image(&other_pixels_f64, 32, 32);
                if other_visual.is_empty() { continue; }

                let mut ocx = 0.0;
                let mut ocy = 0.0;
                for t in &other_visual {
                    ocx += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                    ocy += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
                }
                let on = other_visual.len() as f64;
                ocx /= on;
                ocy /= on;

                if let Some(token) = trainer.lexicon.get(&spoken_word) {
                    let word_hp = HyperbolicPoint::new(
                        Array1::from_vec(vec![
                            token.hyperbolic.coords.get(0).copied().unwrap_or(0.0),
                            token.hyperbolic.coords.get(1).copied().unwrap_or(0.0),
                        ])
                    ).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] });

                    let other_hp = HyperbolicPoint::new(Array1::from_vec(vec![ocx, ocy]))
                        .unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] });

                    if let Ok(d) = ball.distance(&word_hp, &other_hp) {
                        if d < best_guess_dist {
                            best_guess_dist = d;
                            best_guess_label = other_img.label;
                        }
                    }
                }
            }

            guess_correct = best_guess_label == label;
            total_guesses += 1;
            if guess_correct { correct_guesses += 1; }

            let extrinsic = if guess_correct { 1.0 } else { -0.3 };
            let reward_val = extrinsic + curiosity * 0.5;
            total_reward += reward_val;

            // UCB no longer needed - using epsilon-greedy image-aware selection

            let sentence = vec!["der".into(), spoken_word.clone(), "ist".into()];
            transitional.observe(&sentence);

            let reward_signal = trainer.train_step(&sentence, false);
            let tokens = trainer.composer.resolve(&sentence);
            if !tokens.is_empty() {
                let mut custom_reward = reward_signal;
                custom_reward.semantic = reward_val.max(0.0);
                custom_reward.total = reward_val;

                let _ = learner.learn_from_reward(
                    &custom_reward,
                    &tokens,
                    None,
                    None,
                    &mut trainer.concept_graph,
                    &mut trainer.lexicon,
                );
            }

            if let Some(token) = trainer.lexicon.tokens.iter_mut()
                .find(|t| t.surface == expected_word)
            {
                let tx = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                let ty = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);

                let new_x = tx + (cx - tx) * 0.1;
                let new_y = ty + (cy - ty) * 0.1;

                let norm = (new_x * new_x + new_y * new_y).sqrt();
                let (nx, ny) = if norm >= 1.0 {
                    let scale = 0.99 / norm;
                    (new_x * scale, new_y * scale)
                } else {
                    (new_x, new_y)
                };

                token.hyperbolic = HyperbolicPoint::new(Array1::from_vec(vec![nx, ny]))
                    .unwrap_or_else(|_| token.hyperbolic.clone());
            }

            if epoch <= 5 || epoch % 10 == 0 {
                let result = if guess_correct { "✅" } else { "❌" };
                println!("  [Epoch {:>4}] label={:>2} '{}' → says: {:>8} → {} [curiosity={:.3}]",
                    epoch, label, expected_word, spoken_word, result, curiosity);
            }
        }

        if epoch % 5 == 0 && transitional.size() > 0 {
            let start_token = class_words.values().choose(&mut rng).unwrap();
            let generated = transitional.generate(start_token, 3, &mut rng);
            println!("  [Epoch {:>4}] 🧠 Generated: {:?}", epoch, generated);
        }

        let accuracy = 100.0 * correct_guesses as f64 / total_guesses.max(1) as f64;
        let avg_curiosity = curiosity_rewards.iter().sum::<f64>() / curiosity_rewards.len().max(1) as f64;
        
        let metrics = EpochMetrics {
            avg_reward: total_reward / total_guesses.max(1) as f64,
            avg_embedding_shift: 0.0,
            new_edges: trainer.concept_graph.edges.len(),
        };
        history.push(metrics.clone());

        if epoch % 5 == 0 {
            println!("  [Epoch {:>4}] reward={:.4} | accuracy={:.1}% | curiosity={:.4} | vocab={}",
                epoch, metrics.avg_reward, accuracy, avg_curiosity, trainer.lexicon.tokens.len());
        }

        if last_export.elapsed() > Duration::from_secs(30) {
            let _ = export_cifar_state(&trainer, &encoder, &images, &ball, &history, "docs/src/development/cifar_baby_state.json");
            println!("  📊 Exported cifar_baby_state.json (epoch {})", epoch);
            last_export = Instant::now();
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    println!("\n=== Final Export ===");
    let _ = export_cifar_state(&trainer, &encoder, &images, &ball, &history, "docs/src/development/cifar_baby_state.json");
    println!("Final state exported to docs/src/development/cifar_baby_state.json");

    println!("\n=== CIFAR-10 Baby Agent Summary ===");
    println!("Total epochs:     {}", epoch);
    println!("Final reward:     {:.4}", history.last().map(|m| m.avg_reward).unwrap_or(0.0));
    println!("Vocabulary size:  {}", trainer.lexicon.tokens.len());
    println!("Transition rules: {}", transitional.size());
    println!("Concept edges:    {}", trainer.concept_graph.edges.len());
    println!("Final MI:         {:.4}", infomax.current_mi());
    println!("Encoder separation ratio: {:.2}x", after_sep.ratio);
    println!("Open docs/src/development/cifar_baby.html to view live visualization.");
}

fn open_browser() {
    let path = "docs/src/development/cifar_baby.html";
    let result = match std::env::consts::OS {
        "windows" => Command::new("cmd").args(["/c", "start", "", path]).spawn(),
        "macos" => Command::new("open").arg(path).spawn(),
        "linux" => Command::new("xdg-open").arg(path).spawn(),
        _ => Command::new("cmd").args(["/c", "start", "", path]).spawn(),
    };
    if result.is_err() {
        eprintln!("⚠ Could not auto-open browser. Open {} manually.", path);
    }
}


fn export_cifar_state(
    trainer: &SemanticTrainer,
    encoder: &PatchEncoder,
    test_images: &[CifarImage],
    ball: &PoincareBall,
    history: &[EpochMetrics],
    path: &str,
) {
    use std::fs::File;
    use std::io::Write;

    let mut json = String::from("{\n");

    // 1. Lexikon-Tokens
    json.push_str("  \"lexicon\": [\n");
    for (i, token) in trainer.lexicon.tokens.iter().enumerate() {
        let x = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
        let y = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        let cls = format!("{:?}", token.class).replace("TokenClass::", "");
        json.push_str(&format!(
            "    {{\"word\": \"{}\", \"class\": \"{}\", \"x\": {:.6}, \"y\": {:.6}, \"salience\": {:.4}}}",
            token.surface, cls, x, y, token.salience
        ));
        if i < trainer.lexicon.tokens.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    // 2. Training history (for curves)
    json.push_str("  \"history\": [\n");
    let start = history.len().saturating_sub(500);
    for (i, m) in history.iter().enumerate().skip(start) {
        json.push_str(&format!(
            "    {{\"epoch\": {}, \"reward\": {:.6}, \"edges\": {}}}",
            i, m.avg_reward, m.new_edges
        ));
        if i < history.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    // 3. Test-Bild-Centroids
    json.push_str("  \"centroids\": [\n");
    let mut centroid_entries = Vec::new();
    for (idx, img) in test_images.iter().take(50).enumerate() {
        let pixels_f64: Vec<f64> = img.pixels.iter().map(|&x| x as f64).collect();
        let visual = encoder.encode_image(&pixels_f64, 32, 32);
        if visual.is_empty() { continue; }

        let mut sum: Array1<f64> = Array1::zeros(2);
        for v in &visual {
            sum[0] += v.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
            sum[1] += v.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        }
        sum = &sum / visual.len() as f64;
        let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 { sum = &sum * (0.99 / norm); }

        let word = map_cifar_label_to_lexicon(img.label);
        centroid_entries.push(format!(
            "    {{\"id\": {}, \"label_word\": \"{}\", \"true_label\": {}, \"x\": {:.6}, \"y\": {:.6}}}",
            idx, word, img.label, sum[0], sum[1]
        ));
    }
    json.push_str(&centroid_entries.join(",\n"));
    json.push_str("\n  ],\n");

    // 4. Assoziationen
    json.push_str("  \"associations\": [\n");
    let mut assoc_entries = Vec::new();
    for (idx, img) in test_images.iter().take(50).enumerate() {
        let pixels_f64: Vec<f64> = img.pixels.iter().map(|&x| x as f64).collect();
        let visual = encoder.encode_image(&pixels_f64, 32, 32);
        if visual.is_empty() { continue; }

        let mut sum: Array1<f64> = Array1::zeros(2);
        for v in &visual {
            sum[0] += v.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
            sum[1] += v.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        }
        sum = &sum / visual.len() as f64;
        let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 { sum = &sum * (0.99 / norm); }
        let hp = HyperbolicPoint::new(sum).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0, 0.0] });

        let mut best_word = "";
        let mut best_dist = f64::INFINITY;
        for (word, &id) in &trainer.lexicon.word_index {
            let d = ball.distance(&hp, &trainer.lexicon.tokens[id].hyperbolic).unwrap_or(f64::INFINITY);
            if d < best_dist {
                best_dist = d;
                best_word = word.as_str();
            }
        }

        let true_word = map_cifar_label_to_lexicon(img.label);
        let correct = best_word == true_word;
        assoc_entries.push(format!(
            "    {{\"centroid_id\": {}, \"predicted\": \"{}\", \"true\": \"{}\", \"distance\": {:.4}, \"correct\": {}}}",
            idx, best_word, true_word, best_dist, correct
        ));
    }
    json.push_str(&assoc_entries.join(",\n"));
    json.push_str("\n  ],\n");

    // 5. Concept-Graph Kanten
    json.push_str("  \"edges\": [\n");
    let mut edge_entries = Vec::new();
    for edge in &trainer.concept_graph.edges {
        let from = trainer.concept_graph.nodes.get(edge.source)
            .map(|n| n.label.clone()).unwrap_or_default();
        let to = trainer.concept_graph.nodes.get(edge.target)
            .map(|n| n.label.clone()).unwrap_or_default();
        edge_entries.push(format!(
            "    {{\"from\": \"{}\", \"to\": \"{}\", \"weight\": {:.4}}}",
            from, to, edge.weight
        ));
    }
    json.push_str(&edge_entries.join(",\n"));
    json.push_str("\n  ]\n}");

    let mut file = File::create(path).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}
