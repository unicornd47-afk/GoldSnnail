#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/shd"
mkdir -p "$DATA_DIR"

echo "[FETCH] SHD von ZenkeLab..."

# Download .h5.gz
for split in train test; do
    url="https://zenkelab.org/datasets/shd_${split}.h5.gz"
    gz_file="$DATA_DIR/shd_${split}.h5.gz"
    h5_file="$DATA_DIR/shd_${split}.h5"
    
    if [[ -f "$h5_file" ]]; then
        echo "[SKIP] shd_${split}.h5 existiert."
        continue
    fi
    
    if [[ ! -f "$gz_file" ]]; then
        echo "[DOWNLOAD] $url"
        curl -L -o "$gz_file" "$url"
    fi
    
    echo "[EXTRACT] $gz_file"
    gunzip -c "$gz_file" > "$h5_file"
    rm "$gz_file"
done

echo "[CONVERT] HDF5 -> JSON"
python scripts/convert_shd.py

echo "[DONE] Echte SHD-Daten in $DATA_DIR/shd.json"
