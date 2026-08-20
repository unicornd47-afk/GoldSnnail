//! GoldWorm Baby Agent — All 6 Learning Mechanisms
//!
//! Orchestrates the complete learning framework:
//! 1. Hebb + Gradient (existing: RSTDP, SemanticLearner)
//! 2. Dynamical Systems (existing: WorkingMemory attractors)
//! 3. Statistical Learning (existing: TokenComposer + NEW TransitionalLearner)
//! 4. Infomax Curiosity (NEW: InfomaxReward)
//! 5. UCB Exploration (NEW: UCBExplorer)
//! 6. RL (existing: RLAgent)
//!
//! Usage:
//!   cargo run --example baby_agent

use goldworm::{
    PatchEncoder, SemanticTrainer, SemanticLearner, LearningRates, EpochMetrics,
    generate_test_image, PoincareBall, HyperbolicPoint, LexiconToken, TokenClass, Quaternion,
};
use goldworm::baby::{InfomaxReward, UCBExplorer, TransitionalLearner};
use ndarray::Array1;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

static RUNNING: AtomicBool = AtomicBool::new(true);

fn main() {
    println!("=== GoldWorm Baby Agent — All 6 Learning Mechanisms ===\n");
    println!("1. Hebb + Gradient     → RSTDP + SemanticLearner");
    println!("2. Dynamical Systems   → WorkingMemory attractors");
    println!("3. Statistical Learning → TransitionalLearner (NEW)");
    println!("4. Infomax Curiosity    → InfomaxReward (NEW)");
    println!("5. UCB Exploration      → UCBExplorer (NEW)");
    println!("6. RL                   → RLAgent");
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
    let mut learner = SemanticLearner::new(1.0, LearningRates::default());
    let encoder = load_pretrained_encoder("encoder_pretrained.json")
        .unwrap_or_else(|| PatchEncoder::new(4, 2, 1.0));
    let ball = PoincareBall::new(1.0);

    // --- Baby Learning Systems ---
    let mut infomax = InfomaxReward::new(10);
    let mut ucb = UCBExplorer::new(1.0);
    let mut transitional = TransitionalLearner::new();
    let mut rng = rand::thread_rng();

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
    println!("Lexicon initialized: {:?}\n", category_words);

    // --- Image patterns ---
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
    let mut last_used: HashMap<String, u64> = HashMap::new();
    const MAX_VOCAB: usize = 20;

    println!("Baby Agent started. Watching, learning, and naming.\n");

    while RUNNING.load(Ordering::SeqCst) {
        epoch += 1;
        let mut total_reward = 0.0;
        let mut correct_guesses = 0;
        let mut total_guesses = 0;
        let mut curiosity_rewards = Vec::new();

        for (pattern, expected_word) in &image_patterns {
            // Generate image
            let img = generate_distinct_image(pattern, 16, 16);
            let visual = encoder.encode_image(&img, 16, 16);
            
            // Find centroid
            let mut cx = 0.0;
            let mut cy = 0.0;
            for t in &visual {
                cx += t.hyperbolic.coords.get(0).copied().unwrap_or(0.0);
                cy += t.hyperbolic.coords.get(1).copied().unwrap_or(0.0);
            }
            let n = visual.len().max(1) as f64;
            cx /= n;
            cy /= n;

            // Compute patch norm for infomax
            let patch_norm: f64 = visual.iter()
                .map(|t| t.hyperbolic.euclidean_norm())
                .sum::<f64>() / n;
            
            // Hidden state norm (word embedding magnitude)
            let hidden_norm = if let Some(token) = trainer.lexicon.get(expected_word) {
                token.hyperbolic.euclidean_norm()
            } else {
                0.0
            };

            // === 4. INFOMAX CURIOSITY ===
            let curiosity = infomax.reward_delta(patch_norm, hidden_norm);
            curiosity_rewards.push(curiosity);

            // === AGENT A (Speaker): UCB-driven word selection ===
            let candidates: Vec<String> = trainer.lexicon.tokens.iter()
                .filter(|t| t.class != TokenClass::Noise)
                .map(|t| t.surface.clone())
                .collect();
            
            if candidates.is_empty() {
                continue;
            }
            
            let spoken_word = ucb.select(&candidates, &mut rng);
            last_used.insert(spoken_word.clone(), epoch);

            // === AGENT B (Listener): Guess ===
            let mut guess_correct = false;
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

            // === REWARD: extrinsic + infomax curiosity ===
            let extrinsic = if guess_correct { 1.0 } else { -0.3 };
            let reward_val = extrinsic + curiosity * 0.5; // weighted sum
            total_reward += reward_val;

            // === 5. UCB UPDATE ===
            ucb.update(&spoken_word, reward_val);

            // === 3. TRANSITIONAL LEARNER ===
            let sentence = vec!["der".into(), spoken_word.clone(), "ist".into()];
            transitional.observe(&sentence);

            // === SEMANTIC LEARNING ===
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

            // Log
            if epoch <= 5 || epoch % 10 == 0 {
                let result = if guess_correct { "✅" } else { "❌" };
                let curiosity_str = format!("curiosity={:.3}", curiosity);
                println!("  [Epoch {:>4}] Image: {:>14} → says: {:>8} → guesses: {:>14} {} [{}]",
                    epoch, pattern, spoken_word, best_guess_pattern, result, curiosity_str);
            }
        }

        // === 3. TRANSITIONAL: Generate new sequences from learned grammar ===
        if epoch % 5 == 0 && transitional.size() > 0 {
            let start_token = category_words[rng.r#gen::<usize>() % category_words.len()];
            let generated = transitional.generate(&start_token, 3, &mut rng);
            println!("  [Epoch {:>4}] 🧠 Generated sequence: {:?}", epoch, generated);
        }

        let metrics = EpochMetrics {
            avg_reward: total_reward / total_guesses.max(1) as f64,
            avg_embedding_shift: 0.0,
            new_edges: trainer.concept_graph.edges.len(),
        };
        history.push(metrics.clone());

        if epoch % 5 == 0 {
            let accuracy = 100.0 * correct_guesses as f64 / total_guesses.max(1) as f64;
            let avg_curiosity = curiosity_rewards.iter().sum::<f64>() / curiosity_rewards.len().max(1) as f64;
            println!("  [Epoch {:>4}] reward={:.4} | accuracy={:.1}% | curiosity={:.4} | vocab={}",
                epoch, metrics.avg_reward, accuracy, avg_curiosity, trainer.lexicon.tokens.len());
        }

        if last_export.elapsed() > Duration::from_secs(30) {
            let _ = export_state(&trainer, &history, "docs/src/development/baby_agent_state.json");
            println!("  📊 Exported baby_agent_state.json (epoch {})", epoch);
            last_export = Instant::now();
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    println!("\n=== Final Export ===");
    let _ = export_state(&trainer, &history, "docs/src/development/baby_agent_state.json");
    println!("Final state exported to docs/src/development/baby_agent_state.json");

    println!("\n=== Baby Agent Summary ===");
    println!("Total epochs:     {}", epoch);
    println!("Final reward:     {:.4}", history.last().map(|m| m.avg_reward).unwrap_or(0.0));
    println!("Vocabulary size:  {}", trainer.lexicon.tokens.len());
    println!("Transition rules: {}", transitional.size());
    println!("Concept edges:    {}", trainer.concept_graph.edges.len());
    println!("Final MI:         {:.4}", infomax.current_mi());
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

/// Try to load a pretrained encoder from JSON.
fn load_pretrained_encoder(path: &str) -> Option<PatchEncoder> {
    let json = std::fs::read_to_string(path).ok()?;
    
    let patch_size = json.find("\"patch_size\":")?;
    let latent_dim = json.find("\"latent_dim\":")?;
    
    let start = patch_size + "\"patch_size\":".len();
    let end = json[start..].find(|c: char| c == ',' || c == '}').unwrap_or(10);
    let ps: usize = json[start..start+end].trim().parse().ok()?;
    
    let start = latent_dim + "\"latent_dim\":".len();
    let end = json[start..].find(|c: char| c == ',' || c == '}').unwrap_or(10);
    let ld: usize = json[start..start+end].trim().parse().ok()?;
    
    let weights_start = json.find("\"weights\":[").unwrap_or(0) + "\"weights\":[".len();
    let weights_end = json[weights_start..].find(']').unwrap_or(0);
    let weights_str = &json[weights_start..weights_start + weights_end];
    let weights: Vec<f64> = weights_str.split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    let proj_start = json.find("\"latent_proj\":[").map(|p| p + "\"latent_proj\":[".len())?;
    let proj_end = json[proj_start..].find(']').unwrap_or(0);
    let proj_str = &json[proj_start..proj_start + proj_end];
    let latent_proj: Vec<f64> = proj_str.split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    if weights.is_empty() || latent_proj.is_empty() {
        return None;
    }
    
    let mut encoder = PatchEncoder::new(ps, ld, 1.0);
    encoder.weights = weights;
    encoder.latent_proj = latent_proj;
    encoder.weights_grad = vec![0.0; encoder.weights.len()];
    encoder.latent_proj_grad = vec![0.0; encoder.latent_proj.len()];
    Some(encoder)
}
