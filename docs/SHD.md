# SHD (Spiking Heidelberg Digits)

## Überblick

SHD ist ein neuromorpher Audio-Datensatz: 10.000+ Samples von gesprochenen Ziffern
(0–9 auf Englisch und Deutsch) als Spike-Trains aus einem Cochlea-Modell.

| Eigenschaft | Wert |
|-------------|------|
| Trainings-Samples | ~8.300 |
| Test-Samples | ~2.100 |
| Input-Neuronen | 700 (Cochlea-Frequenzkanäle) |
| Klassen | 20 (0–9 EN + 0–9 DE) |
| Dauer | ~0.4–1.4s pro Sample |
| Format | HDF5 (Original) → JSON (GoldSnnail) |

## Datenvorbereitung

### Schritt 1: Abhängigkeiten installieren

```bash
pip install h5py numpy requests
```

### Schritt 2: Konvertierung ausführen

```bash
python scripts/convert_shd.py
```

Das Skript:
1. Lädt `shd_train.h5` und `shd_test.h5` von [zenkelab.org](https://zenkelab.org/datasets)
2. Konvertiert Spike-Trains zu `(time_ms, neuron_id)` Paaren
3. Speichert `data/shd/shd.json` (~50–100 MB)

### Schritt 3: Evaluierung

```bash
# Direkt
cargo run --example eval_shd --release

# Via Benchmark-Runner
cargo run -p goldsnnail-bench -- --repo . eval shd
```

## Architektur

```
SHD-Spikes (700 Neuronen, variabel lang)
    ↓
Rate-Coding: Binned firing rates → 100D
    ↓
L2-Norm: target_radius = 0.75
    ↓
HyperbolicPoint (Poincaré-Ball)
    ↓
Hyperbolic k-NN (k=5) → Label
```

## Erwartete Ergebnisse

| Setup | Erwartete Accuracy | Status |
|-------|-------------------|--------|
| Ungetuned k-NN (Rate-Coding) | 5–15% | Baseline |
| Fine-tuned GridEncoder | 40–60% | Ziel |
| SOTA (snnTorch) | ~90% | Referenz |

## Dateien

| Datei | Beschreibung |
|-------|-------------|
| `src/audio/shd_loader.rs` | JSON-Loader + 100D Feature-Extraktion |
| `src/audio/hyperbolic_knn.rs` | k-NN im Poincaré-Ball |
| `examples/eval_shd.rs` | Evaluations-Pipeline |
| `scripts/convert_shd.py` | HDF5 → JSON Konverter |

## Troubleshooting

**`h5py` lässt sich nicht installieren:**
```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev
pip install h5py
```

**Download ist langsam:**
Die HDF5-Dateien sind ~50 MB. Alternativ manuell von
[zenkelab.org/datasets](https://zenkelab.org/datasets) herunterladen und nach
`data/shd/` verschieben.

**JSON zu groß:**
Falls `shd.json` >100 MB wird, kannst du in `shd_loader.rs` direkt HDF5 laden
(mit `hdf5-rs` Crate). Für den Moment reicht JSON.
