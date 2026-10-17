//! Performance budgets and monitoring for xf.
//!
//! This module defines explicit performance expectations for all xf operations.
//! These budgets serve multiple purposes:
//!
//! 1. **Documentation**: Clear expectations for operation latencies
//! 2. **CI Enforcement**: Fail builds that exceed panic thresholds
//! 3. **Runtime Monitoring**: Log warnings when operations are slow
//! 4. **Optimization Guidance**: Know when performance is acceptable
//!
//! # Performance Tiers
//!
//! | Tier | Target | Warning | Panic | Use Case |
//! |------|--------|---------|-------|----------|
//! | Instant | <1ms | 5ms | 50ms | Search queries |
//! | Fast | <10ms | 50ms | 500ms | Index lookups |
//! | Normal | <100ms | 500ms | 5s | File parsing |
//! | Slow | <1s | 5s | 30s | Full indexing |

use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Performance budget for an operation.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Name of the operation.
    pub name: &'static str,
    /// Target latency (expected p99).
    pub target: Duration,
    /// Warning threshold (log warning if exceeded).
    pub warning: Duration,
    /// Panic threshold (CI failure if exceeded).
    pub panic: Duration,
}

impl Budget {
    /// Create a new budget with the given thresholds.
    #[must_use]
    pub const fn new(name: &'static str, target_ms: u64, warning_ms: u64, panic_ms: u64) -> Self {
        Self {
            name,
            target: Duration::from_millis(target_ms),
            warning: Duration::from_millis(warning_ms),
            panic: Duration::from_millis(panic_ms),
        }
    }

    /// Create a budget for instant operations (<1ms target).
    #[must_use]
    pub const fn instant(name: &'static str) -> Self {
        Self::new(name, 1, 5, 50)
    }

    /// Create a budget for fast operations (<10ms target).
    #[must_use]
    pub const fn fast(name: &'static str) -> Self {
        Self::new(name, 10, 50, 500)
    }

    /// Create a budget for normal operations (<100ms target).
    #[must_use]
    pub const fn normal(name: &'static str) -> Self {
        Self::new(name, 100, 500, 5000)
    }

    /// Create a budget for slow operations (<1s target).
    #[must_use]
    pub const fn slow(name: &'static str) -> Self {
        Self::new(name, 1000, 5000, 30000)
    }

    /// Check if a duration is within the target.
    #[must_use]
    pub fn is_within_target(&self, duration: Duration) -> bool {
        duration <= self.target
    }

    /// Check if a duration exceeds the warning threshold.
    #[must_use]
    pub fn exceeds_warning(&self, duration: Duration) -> bool {
        duration > self.warning
    }

    /// Check if a duration exceeds the panic threshold.
    #[must_use]
    pub fn exceeds_panic(&self, duration: Duration) -> bool {
        duration > self.panic
    }

    /// Get the status of a duration relative to this budget.
    #[must_use]
    pub fn status(&self, duration: Duration) -> BudgetStatus {
        if duration <= self.target {
            BudgetStatus::OnTarget
        } else if duration <= self.warning {
            BudgetStatus::Acceptable
        } else if duration <= self.panic {
            BudgetStatus::Warning
        } else {
            BudgetStatus::Exceeded
        }
    }
}

/// Status of an operation relative to its budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStatus {
    /// Duration is within target (excellent).
    OnTarget,
    /// Duration exceeds target but is acceptable.
    Acceptable,
    /// Duration exceeds warning threshold (needs attention).
    Warning,
    /// Duration exceeds panic threshold (critical).
    Exceeded,
}

impl BudgetStatus {
    /// Check if this status is acceptable for production.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::OnTarget | Self::Acceptable)
    }
}

// =============================================================================
// Search Operation Budgets
// =============================================================================

/// Budget for simple single-term searches.
pub const SEARCH_SIMPLE: Budget = Budget::instant("search_simple");

/// Budget for phrase searches.
pub const SEARCH_PHRASE: Budget = Budget::new("search_phrase", 2, 10, 100);

/// Budget for complex boolean queries.
pub const SEARCH_COMPLEX: Budget = Budget::new("search_complex", 5, 20, 200);

/// Budget for search with type filters.
pub const SEARCH_FILTERED: Budget = Budget::new("search_filtered", 2, 10, 100);

/// Budget for wildcard queries.
pub const SEARCH_WILDCARD: Budget = Budget::new("search_wildcard", 10, 50, 500);

// =============================================================================
// Indexing Operation Budgets
// =============================================================================

/// Budget for parsing a single data file (e.g., tweets.js).
pub const PARSE_FILE: Budget = Budget::normal("parse_file");

/// Budget for indexing a batch of tweets (per 1000 documents).
pub const INDEX_BATCH: Budget = Budget::new("index_batch_1k", 50, 200, 2000);

/// Budget for committing the search index.
pub const INDEX_COMMIT: Budget = Budget::new("index_commit", 100, 500, 5000);

/// Budget for full archive indexing (varies by size).
pub const INDEX_FULL: Budget = Budget::slow("index_full");

// =============================================================================
// Storage Operation Budgets
// =============================================================================

/// Budget for database open/init.
pub const STORAGE_OPEN: Budget = Budget::fast("storage_open");

/// Budget for single record lookup by ID.
pub const STORAGE_LOOKUP: Budget = Budget::instant("storage_lookup");

/// Budget for batch insert (per 1000 records).
pub const STORAGE_BATCH_INSERT: Budget = Budget::new("storage_batch_insert_1k", 50, 200, 2000);

/// Budget for FTS search.
pub const STORAGE_FTS: Budget = Budget::new("storage_fts", 5, 20, 200);

/// Budget for statistics query.
pub const STORAGE_STATS: Budget = Budget::fast("storage_stats");

// =============================================================================
// Timer Utility
// =============================================================================

/// A timer that tracks operation duration and checks against a budget.
#[derive(Debug)]
pub struct Timer {
    budget: Budget,
    start: Instant,
}

impl Timer {
    /// Start a new timer for the given budget.
    #[must_use]
    pub fn start(budget: Budget) -> Self {
        Self {
            budget,
            start: Instant::now(),
        }
    }

    /// Get the elapsed duration.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return the duration.
    /// Logs a warning if the budget was exceeded.
    pub fn stop(self) -> Duration {
        let duration = self.elapsed();
        let status = self.budget.status(duration);

        match status {
            BudgetStatus::OnTarget => {
                debug!(
                    operation = self.budget.name,
                    duration_ms = duration.as_millis(),
                    "Operation completed within target"
                );
            }
            BudgetStatus::Acceptable => {
                debug!(
                    operation = self.budget.name,
                    duration_ms = duration.as_millis(),
                    target_ms = self.budget.target.as_millis(),
                    "Operation completed above target but acceptable"
                );
            }
            BudgetStatus::Warning => {
                info!(
                    operation = self.budget.name,
                    duration_ms = duration.as_millis(),
                    warning_ms = self.budget.warning.as_millis(),
                    "Operation exceeded warning threshold"
                );
            }
            BudgetStatus::Exceeded => {
                info!(
                    operation = self.budget.name,
                    duration_ms = duration.as_millis(),
                    panic_ms = self.budget.panic.as_millis(),
                    "Operation exceeded panic threshold - CRITICAL"
                );
            }
        }

        duration
    }

    /// Stop the timer and check if it exceeded the panic threshold.
    #[must_use]
    pub fn stop_and_check(self) -> (Duration, bool) {
        let budget = self.budget;
        let duration = self.start.elapsed();
        let ok = !budget.exceeds_panic(duration);
        (duration, ok)
    }
}

/// Convenience macro for timing an operation.
#[macro_export]
macro_rules! timed {
    ($budget:expr, $expr:expr) => {{
        let timer = $crate::perf::Timer::start($budget);
        let result = $expr;
        timer.stop();
        result
    }};
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_thresholds() {
        let budget = Budget::instant("test");

        assert!(budget.is_within_target(Duration::from_micros(500)));
        assert!(!budget.is_within_target(Duration::from_millis(2)));

        assert!(!budget.exceeds_warning(Duration::from_millis(3)));
        assert!(budget.exceeds_warning(Duration::from_millis(10)));

        assert!(!budget.exceeds_panic(Duration::from_millis(40)));
        assert!(budget.exceeds_panic(Duration::from_millis(60)));
    }

    #[test]
    fn test_budget_status() {
        let budget = Budget::new("test", 10, 50, 100);

        assert_eq!(
            budget.status(Duration::from_millis(5)),
            BudgetStatus::OnTarget
        );
        assert_eq!(
            budget.status(Duration::from_millis(30)),
            BudgetStatus::Acceptable
        );
        assert_eq!(
            budget.status(Duration::from_millis(75)),
            BudgetStatus::Warning
        );
        assert_eq!(
            budget.status(Duration::from_millis(150)),
            BudgetStatus::Exceeded
        );
    }

    #[test]
    fn test_budget_status_is_ok() {
        assert!(BudgetStatus::OnTarget.is_ok());
        assert!(BudgetStatus::Acceptable.is_ok());
        assert!(!BudgetStatus::Warning.is_ok());
        assert!(!BudgetStatus::Exceeded.is_ok());
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start(Budget::slow("test"));
        std::thread::sleep(Duration::from_millis(10));
        let duration = timer.stop();
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_predefined_budgets() {
        // Verify all budgets have sensible thresholds
        let budgets = [
            SEARCH_SIMPLE,
            SEARCH_PHRASE,
            SEARCH_COMPLEX,
            INDEX_BATCH,
            STORAGE_LOOKUP,
        ];

        for budget in budgets {
            assert!(budget.target < budget.warning);
            assert!(budget.warning < budget.panic);
        }
    }
}
