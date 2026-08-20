#!/bin/bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
TASKS_FILE="${INPUT_DIR}/tasks.json"
PREDICTIONS_FILE="${OUTPUT_DIR}/submission.json"
META_FILE="${OUTPUT_DIR}/meta.json"

mkdir -p "$OUTPUT_DIR"

# Meta-Info für ARC-Prize Evaluator
cat > "$META_FILE" <<EOF
{
  "model": "GoldWorm-v0.2-phase2",
  "size_mb": 0.92,
  "latency_us": 72,
  "language": "rust",
  "status": "running"
}
EOF

echo "[GoldWorm] Starte ARC-AGI Evaluierung..."
echo "[GoldWorm] Input:  $TASKS_FILE"
echo "[GoldWorm] Output: $PREDICTIONS_FILE"

# Prüfe Input
if [[ ! -f "$TASKS_FILE" ]]; then
    echo "{\"error\": \"tasks.json nicht gefunden in $INPUT_DIR\"}" > "$PREDICTIONS_FILE"
    exit 1
fi

# Führe Eval aus (passt Pfade via Env-Var an)
export GOLDWORM_ARC_INPUT="$TASKS_FILE"
export GOLDWORM_ARC_OUTPUT="$PREDICTIONS_FILE"

eval_arc_prize

# Abschluss-Meta
cat > "$META_FILE" <<EOF
{
  "model": "GoldWorm-v0.2-phase2",
  "size_mb": 0.92,
  "latency_us": 72,
  "status": "completed"
}
EOF

echo "[GoldWorm] Fertig. Ergebnis in $PREDICTIONS_FILE"

