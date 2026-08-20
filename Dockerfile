# syntax=docker/dockerfile:1
FROM rust:1.85-slim AS builder

WORKDIR /goldworm
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Workspace bauen (inkl. benchmark_runner)
COPY . .
RUN cargo build --release --example eval_arc_prize

# Runtime-Image (minimal)
FROM debian:bookworm-slim
WORKDIR /goldworm

# Binary + Entrypoint
COPY --from=builder /goldworm/target/release/examples/eval_arc_prize /usr/local/bin/eval_arc_prize
COPY tools/benchmark_runner/arc_entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint

# ARC-Evaluator erwartet Input/Output-Mounts
ENV INPUT_DIR=/data
ENV OUTPUT_DIR=/output

ENTRYPOINT ["/usr/local/bin/entrypoint"]
