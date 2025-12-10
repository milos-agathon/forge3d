//! Performance statistics for the batching system.

/// Performance counters for batching and culling operations.
#[derive(Debug, Default, Clone)]
pub struct BatchingStats {
    /// Total primitives submitted for batching.
    pub total_primitives: usize,
    /// Primitives that passed visibility culling.
    pub visible_primitives: usize,
    /// Primitives culled by frustum test.
    pub culled_primitives: usize,
    /// Draw calls before batching optimization.
    pub draw_calls_before_batching: usize,
    /// Draw calls after batching optimization.
    pub draw_calls_after_batching: usize,
    /// Time spent in batching (milliseconds).
    pub batching_time_ms: f32,
    /// Time spent in culling (milliseconds).
    pub culling_time_ms: f32,
}

impl BatchingStats {
    /// Compute the ratio of culled to total primitives.
    pub fn culling_ratio(&self) -> f32 {
        if self.total_primitives == 0 {
            0.0
        } else {
            self.culled_primitives as f32 / self.total_primitives as f32
        }
    }

    /// Compute the batching efficiency (draw call reduction ratio).
    pub fn batching_efficiency(&self) -> f32 {
        if self.draw_calls_before_batching == 0 {
            0.0
        } else {
            1.0 - (self.draw_calls_after_batching as f32 / self.draw_calls_before_batching as f32)
        }
    }
}
