use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDef {
    pub name: String,
    pub description: String,
    pub metric: String,
    pub prize_pool: String,
    pub priority: u8,
    pub eval_script: String, // Name des GoldSnnail-Examples
    pub output_format: OutputFormat,
    pub needs_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    JsonGrid,    // ARC-AGI Format
    CsvLabels,   // N-MNIST Format
    TextLog,     // Generic
}

#[derive(Default)]
pub struct BenchmarkRegistry {
    defs: HashMap<String, BenchmarkDef>,
}

impl BenchmarkRegistry {
    pub fn default() -> Self {
        let mut defs = HashMap::new();

        defs.insert(
            "arc-prize".into(),
            BenchmarkDef {
                name: "arc-prize".into(),
                description: "ARC-AGI Efficiency Leaderboard".into(),
                metric: "accuracy_per_dollar".into(),
                prize_pool: "$1,000,000+".into(),
                priority: 1,
                eval_script: "eval_arc_prize".into(),
                output_format: OutputFormat::JsonGrid,
                needs_model: true,
            },
        );

        defs.insert(
            "n-mnist".into(),
            BenchmarkDef {
                name: "n-mnist".into(),
                description: "N-MNIST 10-Digit Classification".into(),
                metric: "accuracy".into(),
                prize_pool: "Prestige".into(),
                priority: 2,
                eval_script: "eval_nmnist".into(),
                output_format: OutputFormat::CsvLabels,
                needs_model: true,
            },
        );

        defs.insert(
            "efficiency-baseline".into(),
            BenchmarkDef {
                name: "efficiency-baseline".into(),
                description: "GoldSnnail Eigen-Metriken (Size, Latenz)".into(),
                metric: "size_mb / latency_us".into(),
                prize_pool: "None".into(),
                priority: 3,
                eval_script: "".into(),
                output_format: OutputFormat::TextLog,
                needs_model: false,
            },
        );

        defs.insert(
            "shd".into(),
            BenchmarkDef {
                name: "shd".into(),
                description: "Spiking Heidelberg Digits (Audio)".into(),
                metric: "accuracy".into(),
                prize_pool: "Prestige".into(),
                priority: 2,
                eval_script: "eval_shd".into(),
                output_format: OutputFormat::CsvLabels,
                needs_model: true,
            },
        );

        defs.insert(
            "shd-trained".into(),
            BenchmarkDef {
                name: "shd-trained".into(),
                description: "SHD mit trainiertem Hyperbolic Encoder".into(),
                metric: "accuracy".into(),
                prize_pool: "Prestige".into(),
                priority: 2,
                eval_script: "eval_shd_trained".into(),
                output_format: OutputFormat::CsvLabels,
                needs_model: true,
            },
        );

        Self { defs }
    }

    pub fn discover(&self, repo: &Path) -> Result<Vec<&BenchmarkDef>> {
        let mut found = Vec::new();

        // Prüfe, welche Examples im Repo existieren
        let examples_dir = repo.join("examples");
        if examples_dir.exists() {
            let available: Vec<String> = WalkDir::new(&examples_dir)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|e| e == "rs").unwrap_or(false))
                .filter_map(|e| {
                    e.path()
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                })
                .collect();

            for def in self.defs.values() {
                if def.eval_script.is_empty() || available.contains(&def.eval_script) {
                    found.push(def);
                }
            }
        }

        found.sort_by_key(|d| d.priority);
        Ok(found)
    }

    pub fn get(&self, name: &str) -> Option<&BenchmarkDef> {
        self.defs.get(name)
    }

    pub fn all(&self) -> Vec<&BenchmarkDef> {
        let mut v: Vec<_> = self.defs.values().collect();
        v.sort_by_key(|d| d.priority);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn default_registry_has_arc_prize() {
        let reg = BenchmarkRegistry::default();
        assert!(reg.get("arc-prize").is_some());
        assert_eq!(reg.get("arc-prize").unwrap().priority, 1);
    }

    #[test]
    fn default_registry_has_five_benchmarks() {
        let reg = BenchmarkRegistry::default();
        assert_eq!(reg.all().len(), 5);
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let reg = BenchmarkRegistry::default();
        assert!(reg.get("non-existent").is_none());
    }

    #[test]
    fn discover_filters_by_available_examples() {
        let reg = BenchmarkRegistry::default();
        // Use a path that definitely does not have examples
        let result = reg.discover(&PathBuf::from("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
