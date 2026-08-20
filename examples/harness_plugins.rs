//! Harness Plugin Demo — TypeMap Context, Plugin Registration, and Waterfall Middleware
//!
//! Demonstrates the new extensibility layer:
//! - HarnessContext: type-safe service registry (TypeMap pattern)
//! - Plugin trait: mount capabilities at boot
//! - StepMiddleware: intercept pipeline stages via waterfall chain
//!
//! Usage:
//!   cargo run --example harness_plugins

use goldworm::harness::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Custom Plugin: Telemetry Collector
// ---------------------------------------------------------------------------

struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn name(&self) -> &'static str {
        "telemetry-collector"
    }

    fn apply(&self, ctx: &mut HarnessContext) {
        // Register a custom counter service
        ctx.provide(AtomicUsize::new(0));
        println!("  [plugin] registered telemetry-collector");
    }
}

// ---------------------------------------------------------------------------
// Custom Plugin: Reward Logger
// ---------------------------------------------------------------------------

struct RewardLoggerPlugin;

impl Plugin for RewardLoggerPlugin {
    fn name(&self) -> &'static str {
        "reward-logger"
    }

    fn apply(&self, ctx: &mut HarnessContext) {
        ctx.provide(RewardEngine::new());
        println!("  [plugin] registered reward-logger");
    }
}

// ---------------------------------------------------------------------------
// Custom Middleware: Spike Budget Guard
// ---------------------------------------------------------------------------

struct SpikeBudgetMiddleware {
    max_spikes: usize,
    call_count: AtomicUsize,
}

impl SpikeBudgetMiddleware {
    fn new(max_spikes: usize) -> Self {
        Self {
            max_spikes,
            call_count: AtomicUsize::new(0),
        }
    }
}

impl StepMiddleware for SpikeBudgetMiddleware {
    fn before_step(&self, _ctx: &HarnessContext, task_id: &str) -> bool {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        if count >= self.max_spikes {
            println!("  [middleware] budget exhausted at step {} for {}", count, task_id);
            return false; // short-circuit the waterfall
        }
        true
    }

    fn after_step(&self, _ctx: &HarnessContext, task_id: &str, reward: f64) {
        println!("  [middleware] step completed for {} reward={:.3}", task_id, reward);
    }
}

// ---------------------------------------------------------------------------
// Custom Middleware: Mode Transition Auditor
// ---------------------------------------------------------------------------

struct AuditMiddleware;

impl StepMiddleware for AuditMiddleware {
    fn before_step(&self, ctx: &HarnessContext, task_id: &str) -> bool {
        if let Some(meta) = ctx.get::<MetaController>() {
            println!("  [middleware] audit: mode={} epoch={} task={}", meta.mode.as_str(), meta.epoch, task_id);
        }
        true
    }

    fn after_step(&self, _ctx: &HarnessContext, _task_id: &str, _reward: f64) {}
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== GoldWorm Harness — Plugin & Middleware Demo ===\n");

    // --- Build context with plugins ---
    let mut ctx = HarnessContext::new();
    TelemetryPlugin.apply(&mut ctx);
    RewardLoggerPlugin.apply(&mut ctx);

    // Retrieve services from the TypeMap
    let reward_engine = ctx.get::<RewardEngine>().unwrap();
    let counter = ctx.get::<AtomicUsize>().unwrap();
    println!("Reward engine retrieved: {:?}", reward_engine.weights);
    println!("Telemetry counter initial: {}", counter.load(Ordering::SeqCst));

    // --- Build waterfall middleware chain ---
    let mut chain = MiddlewareChain::new();
    chain.push(std::sync::Arc::new(AuditMiddleware));
    chain.push(std::sync::Arc::new(SpikeBudgetMiddleware::new(3)));

    // --- Simulate a training loop with middleware interception ---
    let task_ids = vec!["arc_task_001", "arc_task_002", "arc_task_003", "arc_task_004"];
    let rewards = vec![0.8, 0.6, 0.9, 0.4];

    for (task_id, &reward) in task_ids.iter().zip(rewards.iter()) {
        println!("\n--- Processing {} ---", task_id);

        // Waterfall: before_step can short-circuit
        if !chain.before_step(&ctx, task_id) {
            println!("  [waterfall] step skipped by middleware");
            continue;
        }

        // Simulate work
        counter.fetch_add(1, Ordering::SeqCst);

        // Waterfall: after_step observers
        chain.after_step(&ctx, task_id, reward);
    }

    println!("\n=== Demo Complete ===");
    println!("Final counter: {}", counter.load(Ordering::SeqCst));
    println!("Services in context: {}", ctx.len());
}
