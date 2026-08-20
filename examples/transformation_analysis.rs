//! Transformation-Vektor-Analyse im Hyperbolic Space
//!
//! Lädt ARC-Tasks, klassifiziert Transformationen und prüft,
//! ob ähnliche Transformationen konsistente Vektoren erzeugen.

use goldsnnail::ArcDataset;
use goldsnnail::vision::grid_encoder::GridEncoder;
use goldsnnail::vision::transformation_analyzer::{TransformationAnalyzer, TransformationVector};

fn main() {
    println!("=== GoldSnnail Transformation Analysis ===\n");

    // 1. Load dataset
    let dataset = match ArcDataset::load_from_directory("data/arc") {
        Ok(ds) => ds,
        Err(e) => {
            println!("Failed to load ARC data: {}", e);
            println!("Download with: git clone https://github.com/fchollet/ARC.git");
            return;
        }
    };

    println!("Loaded {} tasks", dataset.tasks.len());

    if dataset.tasks.len() < 2 {
        println!("Need at least 2 tasks for transformation analysis.");
        return;
    }

    // 2. Initialize encoder
    let mut encoder = GridEncoder::new(100, 32, 16, 0.75);

    // 3. Collect all transformation vectors (pre-training)
    println!("Computing transformation vectors...\n");
    let mut vectors: Vec<TransformationVector> = Vec::new();
    let mut skipped = 0;

    {
        let analyzer = TransformationAnalyzer::new(&encoder);

        for task in dataset.tasks.iter() {
            if task.train_pairs.is_empty() {
                skipped += 1;
                continue;
            }

            for (input, output) in &task.train_pairs {
                let t_type = GridEncoder::classify_transformation(input, output);
                match analyzer.compute_vector(&task.id, input, output, t_type) {
                    Ok(vec) => vectors.push(vec),
                    Err(e) => {
                        eprintln!("  Encode failed for task {}: {}", task.id, e);
                        skipped += 1;
                    }
                }
            }
        }
    } // analyzer dropped here — encoder is free for mutable borrow

    println!("Computed {} transformation vectors ({} skipped)\n", vectors.len(), skipped);

    if vectors.is_empty() {
        println!("No valid transformation vectors computed.");
        return;
    }

    // 4. Analyze each transformation type (pre-training)
    let types = [
        "rotation_90",
        "flip_horizontal",
        "flip_vertical",
        "color_mapping",
        "count_objects",
        "unknown",
    ];

    println!("=== Pre-Training Transformation Consistency Analysis ===");
    println!(
        "{:<20} {:>6} {:>12} {:>12} {:>10}",
        "Type", "Count", "Mean Sim", "Std Sim", "Consistent"
    );
    println!("{}", "-".repeat(65));

    let mut consistent_count = 0;
    {
        let analyzer = TransformationAnalyzer::new(&encoder);
        for t_type in &types {
            let analysis = analyzer.analyze_type(&vectors, t_type);
            println!(
                "{:<20} {:>6} {:>12.3} {:>12.3} {:>10}",
                t_type,
                analysis.task_count,
                analysis.mean_similarity,
                analysis.std_similarity,
                if analysis.is_consistent { "✅" } else { "❌" }
            );
            if analysis.is_consistent {
                consistent_count += 1;
            }
        }
    }

    println!("\n=== Pre-Training Summary ===");
    println!(
        "Consistent transformation types: {}/{}",
        consistent_count,
        types.len()
    );

    if consistent_count >= 3 {
        println!(
            "✅ The hyperbolic space encodes transformation semantics! (>=3 types consistent)"
        );
    } else if consistent_count >= 2 {
        println!("⚠️ Partial success. Simple transformations work, complex ones need work.");
    } else if consistent_count >= 1 {
        println!("⚠️ Only one transformation type is consistent. Feature engineering needed.");
    } else {
        println!("❌ No consistent transformations found. The space may not encode transformation semantics.");
    }

    // 5. Quick self-supervised training
    println!("\n=== Quick Training (50 epochs) ===");
    let tasks: Vec<_> = dataset.tasks.iter().take(10).cloned().collect();
    if !tasks.is_empty() {
        for epoch in (0..50).step_by(10) {
            let mut total_loss = 0.0;
            let mut count = 0;
            for task in &tasks {
                for (input, output) in &task.train_pairs {
                    if let Ok(loss) = encoder.train_step(input, output, 0.01) {
                        total_loss += loss;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                println!("Epoch {:>2}: Avg Distance = {:.4}", epoch, total_loss / count as f64);
            }
        }
    }

    // 6. Re-analyze after training
    println!("\n=== Post-Training Analysis ===");
    let mut post_vectors: Vec<TransformationVector> = Vec::new();

    {
        let analyzer = TransformationAnalyzer::new(&encoder);
        for task in dataset.tasks.iter() {
            if task.train_pairs.is_empty() {
                continue;
            }
            for (input, output) in &task.train_pairs {
                let t_type = GridEncoder::classify_transformation(input, output);
                if let Ok(vec) = analyzer.compute_vector(&task.id, input, output, t_type) {
                    post_vectors.push(vec);
                }
            }
        }
    }

    let mut post_consistent = 0;
    {
        let analyzer = TransformationAnalyzer::new(&encoder);
        for t_type in &types {
            let analysis = analyzer.analyze_type(&post_vectors, t_type);
            if analysis.is_consistent {
                post_consistent += 1;
            }
        }
    }

    println!(
        "Post-training consistent types: {}/{}",
        post_consistent,
        types.len()
    );
}
