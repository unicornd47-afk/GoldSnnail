//! GoldWorm Emergent Naming Game
//!
//! Two agents play a referential communication game with a TRAINABLE PatchEncoder.
//! - Agent A (Speaker) sees an image and describes it with a single word
//! - Agent B (Listener) hears the word and guesses which image it refers to
//! - If B guesses correctly, embeddings shift toward correct image
//! - If B fails, the PatchEncoder LEARNS via backprop and words may split
//!
//! Usage:
//!   cargo run --example emergent_naming_game
//!
//! Press Ctrl+C to stop. Training state is exported every 30s for the HTML demo.

use goldworm::{
    PatchEncoder, SemanticTrainer, SemanticLearner, LearningRates, EpochMetrics,
    PoincareBall, HyperbolicPoint, LexiconToken, TokenClass, Quaternion,
};
use rand::Rng;
use ndarray::Array1;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

static RUNNING: AtomicBool = AtomicBool::new(true);

fn main() {
    println!("=== GoldWorm Emergent Naming Game (Trainable Encoder) ===\n");
    println!("Agent A (Speaker)  → sees image → says ONE word");
    println!("Agent B (Listener) → hears word  → guesses image");
    println!("PatchEncoder LEARNS from reward via BACKPROP.");
    println!("Press Ctrl+C to stop.\n");

    // Ctrl+C handler
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

    // --- Setup ---
    let mut trainer = SemanticTrainer::new(1.0);
    let _learner = SemanticLearner::new(1.0, LearningRates::default());
    let mut encoder = PatchEncoder::new(4, 2, 1.0); // TRAINABLE
    let ball = PoincareBall::new(1.0);

    // --- Fresh lexicon ---
    trainer.lexicon.tokens.clear();
    trainer.lexicon.word_index.clear();
    trainer.lexicon.class_index.clear();
    
    let category_words = ["white", "dark", "left", "top"];
    let positions = [
        [0.6, 0.0],
        [-0.6, 0.0],
        [0.0, 0.6],
        [0.0, -0.6],
    ];
    
    for (i, word) in category_words.iter().enumerate() {
        let coords = Array1::from_vec(positions[i].to_vec());
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
    }
    println!("Fresh lexicon initialized with: {:?}\n", category_words);

    // --- Distinct visual patterns ---
    let image_patterns = [
        ("all_white", "white"),
        ("all_dark", "dark"),
        ("left_bright", "left"),
        ("top_bright", "top"),
    ];

    // --- Game state ---
    let mut epoch: u64 = 0;
    let mut history: Vec<EpochMetrics> = Vec::with_capacity(1000);
    let mut last_export = Instant::now();
    let mut rng = rand::thread_rng();

    // Track per (word, pattern): success/failure counts
    let mut pair_stats: HashMap<(String, String), (usize, usize)> = HashMap::new();
    let mut last_used: HashMap<String, u64> = HashMap::new();
    const MAX_VOCAB: usize = 20;

    println!("Game started. Agent A describes images, Agent B guesses.\n");

    while RUNNING.load(Ordering::SeqCst) {
        epoch += 1;
        let mut total_reward = 0.0;
        let mut correct_guesses = 0;
        let mut total_guesses = 0;

        // Zero gradients at start of epoch
        encoder.zero_grad();

        // Play one round: each image is shown once
        for (pattern, expected_word) in &image_patterns {
            // Generate image
            let img = generate_distinct_image(pattern, 16, 16);
            
            // Use TRAINABLE encoding to cache intermediates for backward
            let patches = encoder.extract_patches(&img, 16, 16);
            let mut visual = Vec::new();
            let mut all_cached: Vec<(Vec<f64>, Vec<f64>)> = Vec::new(); // (patch_data, q_comps)
            
            for patch in &patches {
                let (patch_data, q_comps) = encoder.encode_patch_trainable(patch);
                let (_, latent) = encoder.to_hyperbolic_trainable(&q_comps);
                all_cached.push((patch_data, q_comps.clone()));
                
                let q = Quaternion::new(
                    q_comps[0] as f32, q_comps[1] as f32, 
                    q_comps[2] as f32, q_comps[3] as f32
                ).normalize();
                let h = HyperbolicPoint::new(Array1::from_vec(latent)).unwrap_or_else(|_| HyperbolicPoint { coords: vec![0.0; 2] });
                visual.push(goldworm::VisualToken {
                    patch: patch.clone(),
                    embedding: q,
                    hyperbolic: h,
                    label: String::new(),
                    salience: 1.0,
                });
            }
            
            // Find centroid of image patches
            let mut cx = 0.0;
            let mut cy = 0.0;
            for t in &visual {
                cx += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                cy += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
            }
            let n = visual.len().max(1) as f64;
            cx /= n;
            cy /= n;

            // === AGENT A (Speaker): 80% exploit, 20% explore ===
            let spoken_word = if rng.r#gen::<f64>() < 0.2 {
                // EXPLORE: pick random existing word
                let idx = (rng.r#gen::<f64>() * trainer.lexicon.tokens.len() as f64) as usize;
                trainer.lexicon.tokens[idx.min(trainer.lexicon.tokens.len() - 1)].surface.clone()
            } else {
                // EXPLOIT: pick word with best success rate for this pattern
                let mut best_word = None;
                let mut best_score = f64::NEG_INFINITY;
                
                for token in &trainer.lexicon.tokens {
                    if token.class == TokenClass::Noise { continue; }
                    let key = (token.surface.clone(), pattern.to_string());
                    let (succ, fail) = pair_stats.get(&key).copied().unwrap_or((0, 0));
                    let score = if succ + fail > 0 {
                        (succ as f64) / ((succ + fail) as f64)
                    } else {
                        0.5
                    };
                    
                    if score > best_score {
                        best_score = score;
                        best_word = Some(token.surface.clone());
                    }
                }
                
                best_word.unwrap_or_else(|| expected_word.to_string())
            };

            last_used.insert(spoken_word.clone(), epoch);

            // === AGENT B (Listener): Guess which image based on word ===
            let guess_correct;
            let mut best_guess_pattern = "";
            let mut best_guess_dist = f64::INFINITY;

            for (other_pattern, _) in &image_patterns {
                let other_img = generate_distinct_image(other_pattern, 16, 16);
                let other_visual = encoder.encode_image(&other_img, 16, 16);
                
                let mut ocx = 0.0;
                let mut ocy = 0.0;
                for t in &other_visual {
                    ocx += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                    ocy += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
                }
                let on = other_visual.len().max(1) as f64;
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
                            best_guess_pattern = other_pattern;
                        }
                    }
                }
            }

            guess_correct = best_guess_pattern == *pattern;
            total_guesses += 1;
            if guess_correct { correct_guesses += 1; }

            // === REWARD ===
            let reward_val: f64 = if guess_correct { 1.0 } else { -0.3 };
            total_reward += reward_val;

            // Update pair stats
            let entry = pair_stats.entry((spoken_word.clone(), pattern.to_string())).or_default();
            if guess_correct {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }

            // === BACKPROP: Compute gradient from listener error ===
            if !guess_correct {
                // Gradient: push image centroid AWAY from wrong guess
                let gx = best_guess_pattern_coords(best_guess_pattern).0;
                let gy = best_guess_pattern_coords(best_guess_pattern).1;
                let dx = cx - gx;
                let dy = cy - gy;
                let d_norm = (dx * dx + dy * dy).sqrt().max(1e-12);
                let d_latent = vec![dx / d_norm, dy / d_norm];
                
                for (patch_data, q_comps) in &all_cached {
                    encoder.backward(&d_latent, q_comps, patch_data);
                }
            } else {
                // Gradient: pull image centroid TOWARD correct word embedding
                if let Some(token) = trainer.lexicon.get(&spoken_word) {
                    let word_x = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                    let word_y = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
                    let dx = word_x - cx;
                    let dy = word_y - cy;
                    let d_norm = (dx * dx + dy * dy).sqrt().max(1e-12);
                    let d_latent = vec![dx / d_norm, dy / d_norm];
                    
                    for (patch_data, q_comps) in &all_cached {
                        encoder.backward(&d_latent, q_comps, patch_data);
                    }
                }
            }

            // === EMBEDDING UPDATE ===
            if let Some(token) = trainer.lexicon.tokens.iter_mut()
                .find(|t| t.surface == spoken_word) 
            {
                let tx = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                let ty = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
                
                if guess_correct {
                    // Strong shift toward correct image
                    let new_x = tx + (cx - tx) * 0.6;
                    let new_y = ty + (cy - ty) * 0.6;
                    let (nx, ny) = clamp_to_ball(new_x, new_y);
                    token.hyperbolic = HyperbolicPoint::new(Array1::from_vec(vec![nx, ny]))
                        .unwrap_or_else(|_| token.hyperbolic.clone());
                } else {
                    // Push away from guessed (wrong) image
                    let gx = best_guess_pattern_coords(best_guess_pattern).0;
                    let gy = best_guess_pattern_coords(best_guess_pattern).1;
                    let dx = tx - gx;
                    let dy = ty - gy;
                    let new_x = tx + dx * 0.4;
                    let new_y = ty + dy * 0.4;
                    let (nx, ny) = clamp_to_ball(new_x, new_y);
                    token.hyperbolic = HyperbolicPoint::new(Array1::from_vec(vec![nx, ny]))
                        .unwrap_or_else(|_| token.hyperbolic.clone());
                }
            }

            // === DIRECT VISUAL-SEMANTIC ASSOCIATION (no new words) ===
            if !guess_correct {
                if let Some(token) = trainer.lexicon.tokens.iter_mut()
                    .find(|t| t.surface == spoken_word)
                {
                    let tx = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                    let ty = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
                    
                    let (kx, ky) = best_guess_pattern_coords(pattern);
                    let new_x = tx + (kx - tx) * 0.4;
                    let new_y = ty + (ky - ty) * 0.4;
                    
                    let (gx, gy) = best_guess_pattern_coords(best_guess_pattern);
                    let new_x = new_x + (tx - gx) * 0.2;
                    let new_y = new_y + (ty - gy) * 0.2;
                    
                    let (nx, ny) = clamp_to_ball(new_x, new_y);
                    token.hyperbolic = HyperbolicPoint::new(Array1::from_vec(vec![nx, ny]))
                        .unwrap_or_else(|_| token.hyperbolic.clone());
                    
                    println!("  🔄 Adjusting '{}' → correct pattern '{}'", spoken_word, pattern);
                }
            }

            // Log
            if epoch <= 5 || epoch % 10 == 0 {
                let result = if guess_correct { "✅" } else { "❌" };
                println!("  [Epoch {:>4}] Image: {:>14} → Agent A says: {:>8} → Agent B guesses: {:>14} {}",
                    epoch, pattern, spoken_word, best_guess_pattern, result);
            }
        }

        // Train encoder weights with SGD
        encoder.step(0.005);

        // Cleanup: remove words unused for 100+ epochs (cap at MAX_VOCAB)
        if epoch % 20 == 0 && trainer.lexicon.tokens.len() > MAX_VOCAB {
            let mut removable: Vec<_> = trainer.lexicon.tokens.iter()
                .filter(|t| {
                    let last = last_used.get(&t.surface).copied().unwrap_or(0);
                    (epoch - last) >= 100
                })
                .collect();
            
            // Sort by last used (oldest first), keep only enough to get under MAX_VOCAB
            removable.sort_by_key(|t| last_used.get(&t.surface).copied().unwrap_or(0));
            let to_remove: HashSet<_> = removable.iter().take(trainer.lexicon.tokens.len() - MAX_VOCAB + 5)
                .map(|t| t.surface.clone())
                .collect();
            
            if !to_remove.is_empty() {
                trainer.lexicon.tokens.retain(|t| !to_remove.contains(&t.surface));
                trainer.lexicon.word_index.clear();
                for (i, token) in trainer.lexicon.tokens.iter().enumerate() {
                    trainer.lexicon.word_index.insert(token.surface.clone(), i);
                }
                println!("  🧹 Cleanup: removed {} words (vocab now {})", to_remove.len(), trainer.lexicon.tokens.len());
            }
        }

        let metrics = EpochMetrics {
            avg_reward: total_reward / total_guesses.max(1) as f64,
            avg_embedding_shift: 0.0,
            new_edges: trainer.concept_graph.edges.len(),
        };
        history.push(metrics.clone());

        if epoch % 5 == 0 {
            let accuracy = 100.0 * correct_guesses as f64 / total_guesses.max(1) as f64;
            println!("  [Epoch {:>4}] reward={:.4} | accuracy={:.1}% | vocab={}",
                epoch, metrics.avg_reward, accuracy, trainer.lexicon.tokens.len());
        }

        if last_export.elapsed() > Duration::from_secs(30) {
            let _ = export_state(&trainer, &history, "docs/src/development/naming_game_state.json");
            println!("  📊 Exported naming_game_state.json (epoch {})", epoch);
            last_export = Instant::now();
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    println!("\n=== Final Export ===");
    let _ = export_state(&trainer, &history, "docs/src/development/naming_game_state.json");
    println!("Final state exported to docs/src/development/naming_game_state.json");

    println!("\n=== Game Summary ===");
    println!("Total epochs:     {}", epoch);
    println!("Final reward:     {:.4}", history.last().map(|m| m.avg_reward).unwrap_or(0.0));
    println!("Vocabulary size:  {}", trainer.lexicon.tokens.len());
    println!("Concept edges:    {}", trainer.concept_graph.edges.len());
    println!("\nOpen docs/src/development/agi demo.html to view results.");
}

fn clamp_to_ball(x: f64, y: f64) -> (f64, f64) {
    let norm = (x * x + y * y).sqrt();
    if norm >= 1.0 {
        let scale = 0.99 / norm;
        (x * scale, y * scale)
    } else {
        (x, y)
    }
}

fn best_guess_pattern_coords(pattern: &str) -> (f64, f64) {
    match pattern {
        "all_white" => (0.6, 0.0),
        "all_dark" => (-0.6, 0.0),
        "left_bright" => (0.0, 0.6),
        "top_bright" => (0.0, -0.6),
        _ => (0.0, 0.0),
    }
}



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

fn export_state(
    trainer: &SemanticTrainer,
    history: &[EpochMetrics],
    path: &str,
) -> Result<(), String> {
    let mut json = String::from("{\n");

    json.push_str("  \"epoch_history\": [\n");
    let start = history.len().saturating_sub(1000);
    for (i, m) in history.iter().enumerate().skip(start) {
        json.push_str(&format!(
            "    {{\"epoch\": {}, \"reward\": {:.6}, \"edges\": {}}}",
            i, m.avg_reward, m.new_edges
        ));
        if i < history.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    json.push_str("  \"tokens\": [\n");
    for (i, token) in trainer.lexicon.tokens.iter().enumerate() {
        let x = token.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
        let y = token.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
        let class_str = format!("{:?}", token.class).replace("TokenClass::", "");
        json.push_str(&format!(
            "    {{\"word\": \"{}\", \"class\": \"{}\", \"x\": {:.6}, \"y\": {:.6}, \"salience\": {:.4}}}",
            token.surface, class_str, x, y, token.salience
        ));
        if i < trainer.lexicon.tokens.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ],\n");

    json.push_str("  \"edges\": [\n");
    for (i, edge) in trainer.concept_graph.edges.iter().enumerate() {
        let from = trainer.concept_graph.nodes.get(edge.source)
            .map(|n| n.label.clone()).unwrap_or_default();
        let to = trainer.concept_graph.nodes.get(edge.target)
            .map(|n| n.label.clone()).unwrap_or_default();
        json.push_str(&format!(
            "    {{\"from\": \"{}\", \"to\": \"{}\", \"weight\": {:.4}, \"rel\": \"{:?}\"}}",
            from, to, edge.weight, edge.rel
        ));
        if i < trainer.concept_graph.edges.len() - 1 { json.push_str(",\n"); } else { json.push('\n'); }
    }
    json.push_str("  ]\n}");

    fs::create_dir_all("docs/src/development").map_err(|e| e.to_string())?;
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}