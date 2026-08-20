//! Event Sourcing — Append-only training log with JSONL persistence
//!
//! Inspired by DeepSeek Harness's durable SessionEvent log.
//! Every significant action during training is emitted as an immutable event.
//! A background subscriber writes events to a JSONL file for crash recovery
//! and post-hoc analysis.

use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

// ---------------------------------------------------------------------------
// Event Types
// ---------------------------------------------------------------------------

/// An immutable event emitted during harness execution.
/// Modeled after DeepSeek Harness's SessionEvent types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum HarnessEvent {
    /// A training run started.
    RunStarted {
        profile: String,
        timestamp_ns: u64,
    },
    /// An epoch boundary was crossed.
    EpochBoundary {
        epoch: u64,
        mode: String,
    },
    /// A single training step completed.
    StepCompleted {
        task_id: String,
        reward: f64,
        accuracy: f64,
        plasticity_applied: bool,
    },
    /// A mode transition occurred.
    ModeTransition {
        from: String,
        to: String,
        epoch: u64,
    },
    /// A checkpoint was written.
    CheckpointWritten {
        epoch: u64,
        path: String,
    },
    /// Forgetting was detected.
    ForgettingDetected {
        epoch: u64,
        drop: f64,
        peak_accuracy: f64,
    },
    /// Plateau was detected.
    PlateauDetected {
        epoch: u64,
        patience_exceeded: u64,
    },
    /// Curriculum advanced to a new stage.
    CurriculumAdvanced {
        from_stage: usize,
        to_stage: usize,
    },
    /// The run completed (normally or via crash recovery).
    RunEnded {
        final_epoch: u64,
        best_accuracy: f64,
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Event Bus
// ---------------------------------------------------------------------------

/// Fire-and-forget event bus.
///
/// Uses `std::sync::mpsc` under the hood. Multiple producers can emit events;
/// a single consumer persists them to disk.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: mpsc::Sender<HarnessEvent>,
}

impl EventBus {
    /// Creates a new event bus with a JSONL sink at `path`.
    pub fn new(path: impl Into<String>) -> Self {
        let (sender, receiver) = mpsc::channel::<HarnessEvent>();
        let sink = EventSink::new(path.into(), receiver);
        thread::spawn(move || sink.run());
        Self { sender }
    }

    /// Emits an event. Never blocks the hot loop; drops if the channel is full.
    pub fn emit(&self, event: HarnessEvent) {
        let _ = self.sender.send(event);
    }
}

// ---------------------------------------------------------------------------
// JSONL Sink
// ---------------------------------------------------------------------------

/// Writes events to a JSONL file on a background thread.
struct EventSink {
    path: String,
    receiver: mpsc::Receiver<HarnessEvent>,
}

impl EventSink {
    fn new(path: String, receiver: mpsc::Receiver<HarnessEvent>) -> Self {
        Self { path, receiver }
    }

    fn run(self) {
        use std::io::Write;
        let mut file = std::fs::File::create(&self.path).ok();
        loop {
            match self.receiver.recv() {
                Ok(event) => {
                    if let Some(ref mut f) = file {
                        let line = serde_json::to_string(&event).unwrap_or_default();
                        writeln!(f, "{}", line).ok();
                        let _ = f.flush();
                    }
                }
                Err(_) => break, // channel closed → graceful shutdown
            }
        }
    }
}

// ---------------------------------------------------------------------------
// In-Memory Replay Buffer (for EvalTracker replacement)
// ---------------------------------------------------------------------------

/// A simple append-only event log kept in memory.
/// Can be queried to reconstruct EvalMetrics on demand.
#[derive(Debug, Clone, Default)]
pub struct EventLog {
    events: Vec<HarnessEvent>,
    max_len: usize,
}

impl EventLog {
    pub fn new(max_len: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_len.min(1024)),
            max_len,
        }
    }

    pub fn push(&mut self, event: HarnessEvent) {
        self.events.push(event);
        if self.events.len() > self.max_len {
            self.events.remove(0);
        }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &HarnessEvent> {
        self.events.iter()
    }

    pub fn last(&self) -> Option<&HarnessEvent> {
        self.events.last()
    }

    /// Reconstructs the best accuracy seen so far from the event log.
    pub fn best_accuracy(&self) -> f64 {
        self.events
            .iter()
            .filter_map(|e| match e {
                HarnessEvent::StepCompleted { accuracy, .. } => Some(*accuracy),
                _ => None,
            })
            .fold(0.0, f64::max)
    }

    /// Returns the latest accuracy, if any.
    pub fn latest_accuracy(&self) -> Option<f64> {
        self.events
            .iter()
            .rev()
            .find_map(|e| match e {
                HarnessEvent::StepCompleted { accuracy, .. } => Some(*accuracy),
                _ => None,
            })
    }

    /// Exports the event log to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.events).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// HarnessContext (TypeMap)
// ---------------------------------------------------------------------------

/// A type-erased container for harness services.
/// Plugins insert concrete types; consumers retrieve them by type.
///
/// This is the Rust equivalent of DeepSeek's `ctx` object, but
/// type-safe through the TypeMap pattern.
#[derive(Debug, Default)]
pub struct HarnessContext {
    services: HashMap<std::any::TypeId, Box<dyn Any + Send + Sync>>,
}

impl HarnessContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a service into the context.
    pub fn provide<T: 'static + Send + Sync>(&mut self, service: T) {
        self.services.insert(std::any::TypeId::of::<T>(), Box::new(service));
    }

    /// Retrieves a service by type. Returns `None` if not registered.
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.services
            .get(&std::any::TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_ref::<T>())
    }

    /// Retrieves a mutable service by type.
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.services
            .get_mut(&std::any::TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_mut::<T>())
    }

    /// Returns the number of registered services.
    pub fn len(&self) -> usize {
        self.services.len()
    }

    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Plugin Trait
// ---------------------------------------------------------------------------

/// A plugin that can register services and hooks on a HarnessContext.
///
/// This is the Rust equivalent of DeepSeek's Cordis plugin system.
/// A plugin's `apply` method is called once at boot; it can register
/// any number of services or event listeners.
pub trait Plugin: Send + Sync {
    fn name(&self) -> &'static str;

    /// Called once at harness initialization. Mutates the context.
    fn apply(&self, ctx: &mut HarnessContext);
}

// ---------------------------------------------------------------------------
// Waterfall / Middleware
// ---------------------------------------------------------------------------

/// A middleware that can intercept pipeline stages.
///
/// DeepSeek's `tools/pre-execute` waterfall: listeners must call `next()`
/// to delegate, or short-circuit. In Rust, we use a simple chain.
pub trait StepMiddleware: Send + Sync {
    /// Called before a training step. Return `true` to continue, `false` to skip.
    fn before_step(&self, ctx: &HarnessContext, task_id: &str) -> bool;

    /// Called after a training step.
    fn after_step(&self, ctx: &HarnessContext, task_id: &str, reward: f64);
}

/// A chain of middleware executed in registration order.
#[derive(Default)]
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn StepMiddleware>>,
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, middleware: Arc<dyn StepMiddleware>) {
        self.middlewares.push(middleware);
    }

    pub fn before_step(&self, ctx: &HarnessContext, task_id: &str) -> bool {
        for mw in &self.middlewares {
            if !mw.before_step(ctx, task_id) {
                return false; // short-circuit
            }
        }
        true
    }

    pub fn after_step(&self, ctx: &HarnessContext, task_id: &str, reward: f64) {
        for mw in &self.middlewares {
            mw.after_step(ctx, task_id, reward);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_log_push_and_iter() {
        let mut log = EventLog::new(10);
        log.push(HarnessEvent::RunStarted {
            profile: "test".into(),
            timestamp_ns: 0,
        });
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn event_log_best_accuracy() {
        let mut log = EventLog::new(10);
        log.push(HarnessEvent::StepCompleted {
            task_id: "t1".into(),
            reward: 0.5,
            accuracy: 0.8,
            plasticity_applied: true,
        });
        log.push(HarnessEvent::StepCompleted {
            task_id: "t2".into(),
            reward: 0.9,
            accuracy: 0.95,
            plasticity_applied: true,
        });
        assert!((log.best_accuracy() - 0.95).abs() < 1e-9);
    }

    #[test]
    fn event_log_max_len() {
        let mut log = EventLog::new(3);
        for i in 0..5 {
            log.push(HarnessEvent::EpochBoundary {
                epoch: i,
                mode: "TRAIN".into(),
            });
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn harness_context_type_map() {
        let mut ctx = HarnessContext::new();
        ctx.provide(42u32);
        ctx.provide("hello".to_string());

        assert_eq!(ctx.get::<u32>(), Some(&42));
        assert_eq!(ctx.get::<String>(), Some(&"hello".to_string()));
        assert!(ctx.get::<f64>().is_none());
    }

    #[test]
    fn middleware_chain() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let flag = Arc::new(AtomicBool::new(true));
        let flag2 = flag.clone();

        struct TestMw(Arc<AtomicBool>);
        impl StepMiddleware for TestMw {
            fn before_step(&self, _: &HarnessContext, _: &str) -> bool {
                self.0.load(Ordering::SeqCst)
            }
            fn after_step(&self, _: &HarnessContext, _: &str, _: f64) {}
        }

        let mut chain = MiddlewareChain::new();
        chain.push(Arc::new(TestMw(flag2)));
        assert!(chain.before_step(&HarnessContext::new(), "task"));
    }
}
