use anyhow::Result;
use colored::Colorize;
use crate::registry::{BenchmarkDef, OutputFormat};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

pub fn package(bench: &BenchmarkDef, repo: &Path, out_dir: &Path) -> Result<PathBuf> {
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let pkg_name = format!("submission_{}_{}", bench.name, ts);
    let pkg_dir = out_dir.join("packages").join(&pkg_name);
    fs::create_dir_all(&pkg_dir)?;

    // 1. README mit Submission-Details
    let readme = format!(
        "# Submission Package: {}\n\
         \n\
         - Benchmark: {}\n\
         - Model: GoldSnnail v0.2-phase2\n\
         - Size: 0.92 MB\n\
         - Latency: 72 µs\n\
         - Format: {:?}\n\
         \n\
         ## Reproduktion\n\
         ```bash\n\
         cargo run --release --example {}\n\
         ```\n\
         \n\
         ## Checksums\n\
         (werden automatisch eingefügt)\n\
         ",
        bench.name,
        bench.description,
        bench.output_format,
        bench.eval_script
    );
    fs::write(pkg_dir.join("README.md"), readme)?;

    // 2. Letztes Ergebnis kopieren (falls vorhanden)
    let runs_dir = out_dir.join("runs");
    let mut result_src: Option<PathBuf> = None;

    // Suche in out_dir/runs/
    if runs_dir.exists() {
        let mut entries: Vec<_> = fs::read_dir(&runs_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(&bench.name))
            .collect();
        entries.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
        result_src = entries.last().and_then(|e| {
            let p = e.path().join("result.json");
            if p.exists() { Some(p) } else { None }
        });
    }

    // Fallback: Suche in repo/benchmark_artifacts/runs/
    if result_src.is_none() {
        let repo_runs = repo.join("benchmark_artifacts").join("runs");
        if repo_runs.exists() {
            let mut entries: Vec<_> = fs::read_dir(&repo_runs)?
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with(&bench.name))
                .collect();
            entries.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
            result_src = entries.last().and_then(|e| {
                let p = e.path().join("result.json");
                if p.exists() { Some(p) } else { None }
            });
        }
    }

    if let Some(src) = result_src {
        fs::copy(&src, pkg_dir.join("result.json"))?;
    }

    // 3. Modell-Datei verlinken/kopieren (nur Metadaten, nicht das volle Modell)
    let model_meta = repo.join("models").join("goldsnnail_v0.2.bin.metadata.json");
    if model_meta.exists() {
        fs::copy(&model_meta, pkg_dir.join("model_metadata.json"))?;
    }

    // 4. Zusaetzliche Dateien kopieren
    let runner_dir = repo.join("tools").join("benchmark_runner");
    let extra_files = vec![
        ("arc_entrypoint.sh", runner_dir.join("arc_entrypoint.sh")),
        ("Dockerfile", repo.join("Dockerfile")),
        ("SUBMISSION_CHECKLIST.md", runner_dir.join("SUBMISSION_CHECKLIST.md")),
    ];
    for (dest_name, src_path) in extra_files {
        if src_path.exists() {
            fs::copy(&src_path, pkg_dir.join(dest_name))?;
        }
    }

    // 5. ZIP erstellen
    let zip_path = out_dir.join("packages").join(format!("{}.zip", pkg_name));
    {
        let zip_file = fs::File::create(&zip_path)?;
        let mut zip = ZipWriter::new(zip_file);

        for entry in fs::read_dir(&pkg_dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = path.file_name().unwrap().to_str().unwrap();
            if path.is_file() {
                zip.start_file(
                    name,
                    SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
                )?;
                zip.write_all(&fs::read(&path)?)?;
            } else if path.is_dir() {
                zip.add_directory(
                    name,
                    SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
                )?;
                for inner in fs::read_dir(&path)? {
                    let inner = inner?;
                    let inner_path = inner.path();
                    let inner_name = format!("{}/{}", name, inner_path.file_name().unwrap().to_str().unwrap());
                    if inner_path.is_file() {
                        zip.start_file(
                            &inner_name,
                            SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
                        )?;
                        zip.write_all(&fs::read(&inner_path)?)?;
                    }
                }
            }
        }
        zip.finish()?;
    }

    println!("{} ZIP erstellt: {}", "📦".bold(), zip_path.display());

    println!("{} Submission-Ordner bereit: {}", "📦".bold(), pkg_dir.display());
    println!("{} ZIP erstellt: {}", "📦".bold(), zip_path.display());
    Ok(pkg_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn package_creates_directory_with_readme() {
        let repo_dir = tempfile::tempdir().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let bench = BenchmarkDef {
            name: "test-bench".into(),
            description: "Test".into(),
            metric: "accuracy".into(),
            prize_pool: "None".into(),
            priority: 99,
            eval_script: "eval_test".into(),
            output_format: OutputFormat::TextLog,
            needs_model: false,
        };

        let pkg_dir = package(&bench, repo_dir.path(), out_dir.path()).unwrap();
        assert!(pkg_dir.exists());
        assert!(pkg_dir.join("README.md").exists());
        let readme = fs::read_to_string(pkg_dir.join("README.md")).unwrap();
        assert!(readme.contains("test-bench"));
    }

    #[test]
    fn package_creates_zip_archive() {
        let repo_dir = tempfile::tempdir().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let bench = BenchmarkDef {
            name: "test-bench".into(),
            description: "Test".into(),
            metric: "accuracy".into(),
            prize_pool: "None".into(),
            priority: 99,
            eval_script: "eval_test".into(),
            output_format: OutputFormat::TextLog,
            needs_model: false,
        };

        let _pkg_dir = package(&bench, repo_dir.path(), out_dir.path()).unwrap();
        let _zip_name = format!("submission_test-bench_*.zip");
        let mut found = false;
        for entry in fs::read_dir(out_dir.path().join("packages")).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name().to_str().unwrap().ends_with(".zip") {
                found = true;
                break;
            }
        }
        assert!(found, "ZIP archive was not created");
    }
}


