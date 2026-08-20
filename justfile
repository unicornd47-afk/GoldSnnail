set shell := ["bash", "-cu"]

default:
    @just --list

check:
    cargo check --tests

test:
    cargo test --all-features

miri:
    cargo +nightly miri test

bench:
    cargo bench

lab:
    cargo fmt -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo check --tests
    cargo test --all-features

coverage:
    cargo llvm-cov --all-features --workspace --html --open

snapshot:
    cargo insta review

deny:
    cargo deny check
