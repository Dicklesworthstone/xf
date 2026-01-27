//! Progress bars and spinners for long-running operations.
//!
//! Provides styled progress indicators that automatically disable for
//! non-TTY environments and non-text output formats.
//!
//! # Example
//!
//! ```rust,ignore
//! use xf::progress::ProgressIndicator;
//! use xf::output::Output;
//!
//! let output = Output::new_tty();
//! let mut progress = ProgressIndicator::new_bar(100, "Processing", &output);
//!
//! for i in 0..100 {
//!     // Do work...
//!     progress.inc(1);
//! }
//!
//! progress.finish("Complete!");
//! ```

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

use crate::output::Output;

/// Type of progress indicator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProgressType {
    /// Known total - shows progress bar
    Determinate,
    /// Unknown total - shows spinner
    Indeterminate,
}

/// A progress indicator that auto-disables for non-TTY environments.
///
/// Provides both determinate (progress bar) and indeterminate (spinner)
/// modes, with throughput and ETA calculations.
pub struct ProgressIndicator {
    inner: Option<ProgressBar>,
    progress_type: ProgressType,
    total: u64,
    current: u64,
    message: String,
    start_time: Instant,
    visible: bool,
}

impl ProgressIndicator {
    /// Create a determinate progress bar with known total.
    ///
    /// The progress bar is only visible if output is not quiet, is styled, AND stdout is TTY.
    #[must_use]
    #[allow(clippy::missing_panics_doc)] // Template is static and valid
    pub fn new_bar(total: u64, message: &str, output: &Output) -> Self {
        let visible = !output.is_quiet() && output.is_styled() && output.stdout_is_tty();

        let inner = if visible {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.cyan} [{bar:40.cyan/dim}] {pos}/{len} {msg} ({eta})")
                    .expect("Invalid progress template")
                    .progress_chars("█▓░"),
            );
            pb.set_message(message.to_string());
            Some(pb)
        } else {
            None
        };

        Self {
            inner,
            progress_type: ProgressType::Determinate,
            total,
            current: 0,
            message: message.to_string(),
            start_time: Instant::now(),
            visible,
        }
    }

    /// Create an indeterminate spinner.
    ///
    /// The spinner is only visible if output is not quiet, is styled, AND stdout is TTY.
    #[must_use]
    #[allow(clippy::missing_panics_doc)] // Template is static and valid
    pub fn new_spinner(message: &str, output: &Output) -> Self {
        let visible = !output.is_quiet() && output.is_styled() && output.stdout_is_tty();

        let inner = if visible {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.cyan} {msg}")
                    .expect("Invalid spinner template")
                    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
            );
            pb.set_message(message.to_string());
            pb.enable_steady_tick(Duration::from_millis(80));
            Some(pb)
        } else {
            None
        };

        Self {
            inner,
            progress_type: ProgressType::Indeterminate,
            total: 0,
            current: 0,
            message: message.to_string(),
            start_time: Instant::now(),
            visible,
        }
    }

    /// Check if progress indicator is visible.
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Get current progress value.
    #[must_use]
    pub const fn current(&self) -> u64 {
        self.current
    }

    /// Get total (for determinate progress).
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.total
    }

    /// Get progress type.
    #[must_use]
    pub const fn progress_type(&self) -> ProgressType {
        self.progress_type
    }

    /// Get elapsed time since creation.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Calculate ETA based on current progress.
    ///
    /// Returns `None` for indeterminate progress or if no progress has been made.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn eta(&self) -> Option<Duration> {
        if self.current == 0 || self.progress_type == ProgressType::Indeterminate {
            return None;
        }

        let elapsed = self.start_time.elapsed();
        let rate = self.current as f64 / elapsed.as_secs_f64();
        if rate <= 0.0 {
            return None;
        }

        let remaining = self.total.saturating_sub(self.current);
        let eta_secs = remaining as f64 / rate;
        Some(Duration::from_secs_f64(eta_secs))
    }

    /// Calculate throughput (items per second).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        self.current as f64 / elapsed
    }

    /// Format throughput as human-readable string.
    #[must_use]
    pub fn throughput_string(&self) -> String {
        let rate = self.throughput();
        if rate >= 1000.0 {
            format!("{:.1}K/s", rate / 1000.0)
        } else {
            format!("{rate:.0}/s")
        }
    }

    /// Update progress to a specific value.
    pub fn update(&mut self, position: u64) {
        self.current = position;
        if let Some(ref pb) = self.inner {
            pb.set_position(position);
        }
    }

    /// Increment progress by delta.
    pub fn inc(&mut self, delta: u64) {
        self.current = self.current.saturating_add(delta);
        if let Some(ref pb) = self.inner {
            pb.inc(delta);
        }
    }

    /// Update message while progress is running.
    pub fn set_message(&mut self, message: &str) {
        self.message = message.to_string();
        if let Some(ref pb) = self.inner {
            pb.set_message(message.to_string());
        }
    }

    /// Get current message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Tick spinner (for indeterminate progress).
    pub fn tick(&self) {
        if let Some(ref pb) = self.inner {
            pb.tick();
        }
    }

    /// Finish progress with a final message.
    pub fn finish(&self, message: &str) {
        if let Some(ref pb) = self.inner {
            pb.finish_with_message(message.to_string());
        }
    }

    /// Finish and clear from display.
    pub fn finish_and_clear(&self) {
        if let Some(ref pb) = self.inner {
            pb.finish_and_clear();
        }
    }

    /// Abandon progress (error case).
    pub fn abandon(&self, message: &str) {
        if let Some(ref pb) = self.inner {
            pb.abandon_with_message(message.to_string());
        }
    }
}

/// Tracker for multiple parallel progress bars.
pub struct MultiProgressTracker {
    multi: Option<MultiProgress>,
    bars: Vec<ProgressIndicator>,
    visible: bool,
}

impl MultiProgressTracker {
    /// Create a new multi-progress tracker.
    #[must_use]
    pub fn new(output: &Output) -> Self {
        let visible = !output.is_quiet() && output.is_styled() && output.stdout_is_tty();

        Self {
            multi: if visible {
                Some(MultiProgress::new())
            } else {
                None
            },
            bars: vec![],
            visible,
        }
    }

    /// Check if tracker is visible.
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Add a new progress bar and return its index.
    #[allow(clippy::missing_panics_doc)] // Template is static and valid
    pub fn add_bar(&mut self, total: u64, message: &str) -> usize {
        let idx = self.bars.len();

        if let Some(ref multi) = self.multi {
            let pb = multi.add(ProgressBar::new(total));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.cyan} [{bar:30.cyan/dim}] {pos}/{len} {msg}")
                    .expect("Invalid template")
                    .progress_chars("█▓░"),
            );
            pb.set_message(message.to_string());

            self.bars.push(ProgressIndicator {
                inner: Some(pb),
                progress_type: ProgressType::Determinate,
                total,
                current: 0,
                message: message.to_string(),
                start_time: Instant::now(),
                visible: true,
            });
        } else {
            self.bars.push(ProgressIndicator {
                inner: None,
                progress_type: ProgressType::Determinate,
                total,
                current: 0,
                message: message.to_string(),
                start_time: Instant::now(),
                visible: false,
            });
        }

        idx
    }

    /// Get the number of bars.
    #[must_use]
    pub fn bar_count(&self) -> usize {
        self.bars.len()
    }

    /// Update a specific bar's position.
    pub fn update(&mut self, idx: usize, position: u64) {
        if let Some(bar) = self.bars.get_mut(idx) {
            bar.update(position);
        }
    }

    /// Increment a specific bar.
    pub fn inc(&mut self, idx: usize, delta: u64) {
        if let Some(bar) = self.bars.get_mut(idx) {
            bar.inc(delta);
        }
    }

    /// Get a bar's current position.
    #[must_use]
    pub fn current(&self, idx: usize) -> u64 {
        self.bars.get(idx).map_or(0, ProgressIndicator::current)
    }

    /// Finish all bars with a message.
    pub fn finish_all(&self, message: &str) {
        for bar in &self.bars {
            bar.finish(message);
        }
    }

    /// Finish and clear all bars.
    pub fn finish_and_clear_all(&self) {
        for bar in &self.bars {
            bar.finish_and_clear();
        }
    }
}

/// Specialized progress for indexing operations.
///
/// Tracks both document processing and embedding generation with
/// phase-aware messaging.
pub struct IndexingProgress {
    progress: ProgressIndicator,
    phase: String,
    documents_processed: u64,
    embeddings_generated: u64,
}

impl IndexingProgress {
    /// Create a new indexing progress tracker.
    #[must_use]
    pub fn new(total_docs: u64, output: &Output) -> Self {
        let progress = ProgressIndicator::new_bar(total_docs, "Indexing", output);

        Self {
            progress,
            phase: "Starting".to_string(),
            documents_processed: 0,
            embeddings_generated: 0,
        }
    }

    /// Check if progress is visible.
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.progress.is_visible()
    }

    /// Set the current indexing phase.
    pub fn set_phase(&mut self, phase: &str) {
        self.phase = phase.to_string();
        self.update_message();
    }

    /// Get the current phase.
    #[must_use]
    pub fn phase(&self) -> &str {
        &self.phase
    }

    /// Increment document count.
    pub fn increment_docs(&mut self, count: u64) {
        self.documents_processed += count;
        self.progress.inc(count);
        self.update_message();
    }

    /// Increment embedding count.
    pub fn increment_embeddings(&mut self, count: u64) {
        self.embeddings_generated += count;
        self.update_message();
    }

    /// Get documents processed count.
    #[must_use]
    pub const fn documents_processed(&self) -> u64 {
        self.documents_processed
    }

    /// Get embeddings generated count.
    #[must_use]
    pub const fn embeddings_generated(&self) -> u64 {
        self.embeddings_generated
    }

    /// Get elapsed time.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.progress.elapsed()
    }

    /// Get throughput.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        self.progress.throughput()
    }

    /// Update the progress message.
    fn update_message(&mut self) {
        let msg = format!(
            "{} ({} docs, {} embeddings)",
            self.phase, self.documents_processed, self.embeddings_generated
        );
        self.progress.set_message(&msg);
    }

    /// Finish indexing with summary.
    pub fn finish(&self) {
        let elapsed = self.progress.elapsed();
        let throughput = self.progress.throughput();

        self.progress.finish(&format!(
            "Done: {} docs, {} embeddings in {:.1}s ({:.0}/s)",
            self.documents_processed,
            self.embeddings_generated,
            elapsed.as_secs_f64(),
            throughput
        ));
    }

    /// Abandon indexing due to error.
    pub fn abandon(&self, error: &str) {
        self.progress.abandon(&format!(
            "Failed at {} docs: {}",
            self.documents_processed, error
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar_creation() {
        let output = Output::new_tty();
        let bar = ProgressIndicator::new_bar(100, "Test", &output);
        assert_eq!(bar.total(), 100);
        assert_eq!(bar.current(), 0);
        assert_eq!(bar.progress_type(), ProgressType::Determinate);
    }

    #[test]
    fn test_spinner_creation() {
        let output = Output::new_tty();
        let spinner = ProgressIndicator::new_spinner("Loading", &output);
        assert_eq!(spinner.progress_type(), ProgressType::Indeterminate);
        assert_eq!(spinner.message(), "Loading");
    }

    #[test]
    fn test_progress_update() {
        let output = Output::new_piped();
        let mut bar = ProgressIndicator::new_bar(100, "Test", &output);
        bar.update(50);
        assert_eq!(bar.current(), 50);
    }

    #[test]
    fn test_progress_increment() {
        let output = Output::new_piped();
        let mut bar = ProgressIndicator::new_bar(100, "Test", &output);
        bar.inc(10);
        bar.inc(20);
        assert_eq!(bar.current(), 30);
    }

    #[test]
    fn test_progress_increment_saturates() {
        let output = Output::new_piped();
        let mut bar = ProgressIndicator::new_bar(100, "Test", &output);
        bar.update(u64::MAX - 5);
        bar.inc(10);
        assert_eq!(bar.current(), u64::MAX);
    }

    #[test]
    fn test_progress_disabled_for_non_tty() {
        let output = Output::new_piped();
        let bar = ProgressIndicator::new_bar(100, "Test", &output);
        assert!(!bar.is_visible());
    }

    #[test]
    fn test_throughput_calculation() {
        let output = Output::new_piped();
        let mut bar = ProgressIndicator::new_bar(100, "Test", &output);

        bar.update(50);
        std::thread::sleep(Duration::from_millis(100));

        let throughput = bar.throughput();
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_throughput_string_small() {
        let output = Output::new_piped();
        let bar = ProgressIndicator::new_bar(100, "Test", &output);
        let s = bar.throughput_string();
        assert!(s.contains("/s"));
    }

    #[test]
    fn test_eta_no_progress() {
        let output = Output::new_piped();
        let bar = ProgressIndicator::new_bar(100, "Test", &output);
        assert!(bar.eta().is_none());
    }

    #[test]
    fn test_eta_with_progress() {
        let output = Output::new_piped();
        let mut bar = ProgressIndicator::new_bar(100, "Test", &output);

        bar.update(50);
        std::thread::sleep(Duration::from_millis(100));

        let eta = bar.eta();
        assert!(eta.is_some());
    }

    #[test]
    fn test_eta_indeterminate_returns_none() {
        let output = Output::new_piped();
        let mut spinner = ProgressIndicator::new_spinner("Test", &output);
        spinner.current = 50; // Even with progress, spinners have no ETA
        assert!(spinner.eta().is_none());
    }

    #[test]
    fn test_message_update() {
        let output = Output::new_piped();
        let mut bar = ProgressIndicator::new_bar(100, "Initial", &output);
        bar.set_message("Updated");
        assert_eq!(bar.message(), "Updated");
    }

    #[test]
    fn test_elapsed() {
        let output = Output::new_piped();
        let bar = ProgressIndicator::new_bar(100, "Test", &output);
        std::thread::sleep(Duration::from_millis(50));
        assert!(bar.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn test_finish_no_panic() {
        let output = Output::new_piped();
        let bar = ProgressIndicator::new_bar(100, "Test", &output);
        bar.finish("Done!");
        bar.finish_and_clear();
        bar.abandon("Error!");
    }

    #[test]
    fn test_tick_no_panic() {
        let output = Output::new_piped();
        let spinner = ProgressIndicator::new_spinner("Test", &output);
        spinner.tick();
    }

    #[test]
    fn test_multi_progress_creation() {
        let output = Output::new_piped();
        let mut multi = MultiProgressTracker::new(&output);

        let idx1 = multi.add_bar(100, "Bar 1");
        let idx2 = multi.add_bar(200, "Bar 2");

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(multi.bar_count(), 2);
    }

    #[test]
    fn test_multi_progress_update() {
        let output = Output::new_piped();
        let mut multi = MultiProgressTracker::new(&output);

        let idx = multi.add_bar(100, "Test");
        multi.update(idx, 50);

        assert_eq!(multi.current(idx), 50);
    }

    #[test]
    fn test_multi_progress_inc() {
        let output = Output::new_piped();
        let mut multi = MultiProgressTracker::new(&output);

        let idx = multi.add_bar(100, "Test");
        multi.inc(idx, 10);
        multi.inc(idx, 20);

        assert_eq!(multi.current(idx), 30);
    }

    #[test]
    fn test_multi_progress_finish() {
        let output = Output::new_piped();
        let mut multi = MultiProgressTracker::new(&output);

        multi.add_bar(100, "Bar 1");
        multi.add_bar(200, "Bar 2");

        // Should not panic
        multi.finish_all("Done!");
        multi.finish_and_clear_all();
    }

    #[test]
    fn test_multi_progress_invalid_index() {
        let output = Output::new_piped();
        let mut multi = MultiProgressTracker::new(&output);

        // Operations on invalid index should not panic
        multi.update(99, 50);
        multi.inc(99, 10);
        assert_eq!(multi.current(99), 0);
    }

    #[test]
    fn test_indexing_progress_creation() {
        let output = Output::new_piped();
        let progress = IndexingProgress::new(1000, &output);

        assert_eq!(progress.documents_processed(), 0);
        assert_eq!(progress.embeddings_generated(), 0);
        assert_eq!(progress.phase(), "Starting");
    }

    #[test]
    fn test_indexing_progress_phases() {
        let output = Output::new_piped();
        let mut progress = IndexingProgress::new(1000, &output);

        progress.set_phase("Loading");
        assert_eq!(progress.phase(), "Loading");

        progress.set_phase("Processing");
        assert_eq!(progress.phase(), "Processing");
    }

    #[test]
    fn test_indexing_progress_counts() {
        let output = Output::new_piped();
        let mut progress = IndexingProgress::new(1000, &output);

        progress.increment_docs(100);
        progress.increment_embeddings(50);

        assert_eq!(progress.documents_processed(), 100);
        assert_eq!(progress.embeddings_generated(), 50);
    }

    #[test]
    fn test_indexing_progress_finish() {
        let output = Output::new_piped();
        let mut progress = IndexingProgress::new(1000, &output);

        progress.increment_docs(100);
        progress.increment_embeddings(50);

        // Should not panic
        progress.finish();
    }

    #[test]
    fn test_indexing_progress_abandon() {
        let output = Output::new_piped();
        let mut progress = IndexingProgress::new(1000, &output);

        progress.increment_docs(50);

        // Should not panic
        progress.abandon("Test error");
    }

    #[test]
    fn test_indexing_progress_throughput() {
        let output = Output::new_piped();
        let mut progress = IndexingProgress::new(1000, &output);

        progress.increment_docs(100);
        std::thread::sleep(Duration::from_millis(50));

        assert!(progress.throughput() > 0.0);
        assert!(progress.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn test_progress_type_debug() {
        assert_eq!(format!("{:?}", ProgressType::Determinate), "Determinate");
        assert_eq!(
            format!("{:?}", ProgressType::Indeterminate),
            "Indeterminate"
        );
    }

    #[test]
    fn test_progress_type_eq() {
        assert_eq!(ProgressType::Determinate, ProgressType::Determinate);
        assert_ne!(ProgressType::Determinate, ProgressType::Indeterminate);
    }
}
