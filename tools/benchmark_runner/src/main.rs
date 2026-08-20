use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::PathBuf;

mod evaluator;
mod registry;
mod submitter;
mod tracker;

use registry::BenchmarkRegistry;

#[derive(Parser)]
#[command(name = "goldsnnail-bench")]
#[command(about = "Lokaler Benchmark-Runner & Leaderboard-Tracker für GoldSnnail")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Pfad zum GoldSnnail-Repo-Root
    #[arg(short, long, default_value = "../..")]
    repo: PathBuf,

    /// Output-Verzeichnis für Reports
    #[arg(short, long, default_value = "./benchmark_artifacts")]
    out: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Repo scannen und Benchmark-Kandidaten auflisten
    Scan,
    /// Einen Benchmark lokal evaluieren
    Eval {
        /// Benchmark-Name (z.B. arc-prize, n-mnist)
        name: String,
    },
    /// Alle bekannten Benchmarks durchlaufen
    RunAll,
    /// Submission-Paket erstellen (ohne Upload)
    Package {
        /// Benchmark-Name
        name: String,
    },
    /// Leaderboard-Status anzeigen
    Status,
    /// Tracker initialisieren/aktualisieren
    Init,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let registry = BenchmarkRegistry::default();

    match cli.command {
        Commands::Scan => {
            println!("{}", "🔍 Scanne GoldSnnail-Repo...".bold().cyan());
            let candidates = registry.discover(&cli.repo)?;
            println!("{} Kandidaten gefunden:\n", candidates.len());
            for c in &candidates {
                println!(
                    "  • {} — {} (Prio: {})",
                    c.name.bold(),
                    c.description,
                    c.priority
                );
                println!("    Metric: {} | Prize: {}", c.metric, c.prize_pool);
            }
        }

        Commands::Eval { name } => {
            println!("{} Benchmark: {}", "▶️ Running".bold().green(), name);
            let bench = registry
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("Unbekannter Benchmark: {}", name))?;
            let result = evaluator::run(bench, &cli.repo, &cli.out)?;
            println!("{}\n{:#?}", "✅ Ergebnis:".bold().green(), result);
        }

        Commands::RunAll => {
            println!("{}", "🚀 Starte alle Benchmarks...".bold().magenta());
            for bench in registry.all() {
                println!("\n{}", "─".repeat(60).dimmed());
                match evaluator::run(bench, &cli.repo, &cli.out) {
                    Ok(r) => println!("{} {}: {}", "✓".green(), bench.name, r.score),
                    Err(e) => println!("{} {}: {}", "✗".red(), bench.name, e),
                }
            }
        }

        Commands::Package { name } => {
            let bench = registry
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("Unbekannter Benchmark: {}", name))?;
            let path = submitter::package(bench, &cli.repo, &cli.out)?;
            println!("{} Paket erstellt: {}", "📦".bold().yellow(), path.display());
        }

        Commands::Status => {
            tracker::print_status(&cli.out)?;
        }

        Commands::Init => {
            tracker::init(&cli.out)?;
            println!("{}", "📝 Tracker initialisiert.".bold().green());
        }
    }

    Ok(())
}
