#!/usr/bin/env python3
"""
SHD Converter — echte Daten von ZenkeLab.
Voraussetzung: scripts/fetch_shd.sh wurde ausgeführt (Download + gunzip).
"""

import json
import os
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError:
    print("[ERROR] pip install h5py numpy")
    raise SystemExit(1)

DATA_DIR = Path("data/shd")
OUTPUT_FILE = DATA_DIR / "shd.json"

def convert_split(h5_path: Path, split: str) -> list:
    print(f"[CONVERT] {h5_path.name}")
    samples = []
    
    with h5py.File(h5_path, 'r') as f:
        labels = f['labels']
        spikes_group = f['spikes']
        times = spikes_group['times']
        units = spikes_group['units']
        
        for i in range(len(labels)):
            spike_times = times[i].tolist()
            spike_units = [int(u) for u in units[i]]
            samples.append({
                'spikes': list(zip(spike_times, spike_units)),
                'label': int(labels[i])
            })
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(labels)}")
    
    print(f"[DONE] {split}: {len(samples)} Samples")
    return samples

def main():
    train_samples = convert_split(DATA_DIR / "shd_train.h5", "train")
    test_samples = convert_split(DATA_DIR / "shd_test.h5", "test")
    
    dataset = {
        'train': train_samples,
        'test': test_samples,
        'num_neurons': 700,
        'duration_ms': 1000.0,
        'num_classes': 20,
        'source': 'https://zenkelab.org/datasets/shd'
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f)
    
    print(f"\n[SUCCESS] {OUTPUT_FILE}")
    print(f"  Train: {len(train_samples)}")
    print(f"  Test:  {len(test_samples)}")
    print(f"\nJetzt ausführen:")
    print(f"  cargo run --example eval_shd --release")

if __name__ == "__main__":
    main()
