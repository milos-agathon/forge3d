//! Q3: GPU profiling markers & timestamp queries
//!
//! Provides GPU timing utilities for performance profiling and debugging.
//! Supports RenderDoc, Nsight Graphics, and RGP markers with configurable
//! timestamp collection for minimal overhead.

use crate::error::{RenderError, RenderResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::*;

/// Handle for a GPU timing scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimingScopeId(usize);

/// GPU timing configuration
#[derive(Debug, Clone)]
pub struct GpuTimingConfig {
    /// Enable timestamp queries (requires TIMESTAMP_QUERY feature)
    pub enable_timestamps: bool,
    /// Enable pipeline statistics (requires PIPELINE_STATISTICS_QUERY feature)
    pub enable_pipeline_stats: bool,
    /// Enable debug markers for external profilers
    pub enable_debug_markers: bool,
    /// Label prefix for timing scopes
    pub label_prefix: String,
    /// Maximum number of timing queries per frame
    pub max_queries_per_frame: u32,
}

impl Default for GpuTimingConfig {
    fn default() -> Self {
        Self {
            enable_timestamps: true,
            enable_pipeline_stats: false, // Often not supported
            enable_debug_markers: true,
            label_prefix: "forge3d".to_string(),
            max_queries_per_frame: 64,
        }
    }
}

/// GPU timing measurement result
#[derive(Debug, Clone, Default)]
pub struct TimingResult {
    /// Scope name/label
    pub name: String,
    /// GPU time in milliseconds
    pub gpu_time_ms: f32,
    /// Whether timestamp was successfully measured
    pub timestamp_valid: bool,
    /// Pipeline statistics (if available)
    pub pipeline_stats: Option<PipelineStatistics>,
}

/// Pipeline statistics from GPU
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    /// Number of vertex invocations
    pub vertex_invocations: u64,
    /// Number of clipping invocations
    pub clipper_invocations: u64,
    /// Number of fragment invocations  
    pub fragment_invocations: u64,
    /// Number of compute invocations
    pub compute_invocations: u64,
}

/// Active timing scope for measuring GPU work
pub struct TimingScope<'a> {
    timing_manager: &'a mut GpuTimingManager,
    scope_id: TimingScopeId,
    encoder: &'a mut CommandEncoder,
}

impl<'a> TimingScope<'a> {
    /// Begin timing scope with debug marker
    pub fn begin(&mut self, label: &str) {
        self.timing_manager
            .begin_scope_internal(self.encoder, self.scope_id, label);
    }

    /// End timing scope  
    pub fn end(self) {
        self.timing_manager
            .end_scope_internal(self.encoder, self.scope_id);
    }
}

/// GPU timing manager for performance profiling
pub struct GpuTimingManager {
    config: GpuTimingConfig,
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,

    // Timestamp queries
    timestamp_query_set: Option<QuerySet>,
    timestamp_buffer: Option<Buffer>,
    timestamp_readback_buffer: Option<Buffer>,

    // Pipeline statistics queries
    pipeline_stats_query_set: Option<QuerySet>,
    pipeline_stats_buffer: Option<Buffer>,
    pipeline_stats_readback_buffer: Option<Buffer>,

    // Timing state
    active_scopes: HashMap<TimingScopeId, String>,
    query_index: u32,
    #[allow(dead_code)]
    results: Arc<Mutex<Vec<TimingResult>>>,
    scope_labels: Vec<String>,

    // Feature support
    supports_timestamps: bool,
    supports_pipeline_stats: bool,
    timestamp_period: f64, // Nanoseconds per timestamp unit
}

impl GpuTimingManager {
    /// Create new GPU timing manager
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: GpuTimingConfig,
    ) -> RenderResult<Self> {
        let features = device.features();
        let _limits = device.limits();

        let supports_timestamps =
            config.enable_timestamps && features.contains(Features::TIMESTAMP_QUERY);
        let supports_pipeline_stats =
            config.enable_pipeline_stats && features.contains(Features::PIPELINE_STATISTICS_QUERY);

        // Get timestamp period from device limits (if supported)
        let timestamp_period = if supports_timestamps {
            // wgpu doesn't expose timestamp period directly, assume 1ns for now
            1.0
        } else {
            1.0
        };

        let mut manager = Self {
            config: config.clone(),
            device: device.clone(),
            queue,
            timestamp_query_set: None,
            timestamp_buffer: None,
            timestamp_readback_buffer: None,
            pipeline_stats_query_set: None,
            pipeline_stats_buffer: None,
            pipeline_stats_readback_buffer: None,
            active_scopes: HashMap::new(),
            query_index: 0,
            results: Arc::new(Mutex::new(Vec::new())),
            scope_labels: Vec::new(),
            supports_timestamps,
            supports_pipeline_stats,
            timestamp_period,
        };

        // Initialize query sets and buffers
        manager.initialize_queries()?;

        Ok(manager)
    }

    fn initialize_queries(&mut self) -> RenderResult<()> {
        let query_count = self.config.max_queries_per_frame * 2; // Begin + End for each scope

        // Initialize timestamp queries
        if self.supports_timestamps {
            let query_set = self.device.create_query_set(&QuerySetDescriptor {
                label: Some("gpu_timing_timestamps"),
                ty: QueryType::Timestamp,
                count: query_count,
            });

            let buffer_size = (query_count as u64) * std::mem::size_of::<u64>() as u64;

            let timestamp_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("gpu_timing_timestamp_buffer"),
                size: buffer_size,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let timestamp_readback = self.device.create_buffer(&BufferDescriptor {
                label: Some("gpu_timing_timestamp_readback"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            self.timestamp_query_set = Some(query_set);
            self.timestamp_buffer = Some(timestamp_buffer);
            self.timestamp_readback_buffer = Some(timestamp_readback);
        }

        // Initialize pipeline statistics queries
        if self.supports_pipeline_stats {
            let stats_query_set = self.device.create_query_set(&QuerySetDescriptor {
                label: Some("gpu_timing_pipeline_stats"),
                ty: QueryType::PipelineStatistics(
                    PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
                        | PipelineStatisticsTypes::CLIPPER_INVOCATIONS
                        | PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS
                        | PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS,
                ),
                count: query_count,
            });

            let stats_buffer_size = (query_count as u64) * std::mem::size_of::<u64>() as u64 * 4; // 4 stats

            let pipeline_stats_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("gpu_timing_pipeline_stats_buffer"),
                size: stats_buffer_size,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let pipeline_stats_readback = self.device.create_buffer(&BufferDescriptor {
                label: Some("gpu_timing_pipeline_stats_readback"),
                size: stats_buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            self.pipeline_stats_query_set = Some(stats_query_set);
            self.pipeline_stats_buffer = Some(pipeline_stats_buffer);
            self.pipeline_stats_readback_buffer = Some(pipeline_stats_readback);
        }

        Ok(())
    }

    /// Begin a new timing scope
    pub fn begin_scope<'a>(
        &'a mut self,
        encoder: &'a mut CommandEncoder,
        label: &str,
    ) -> TimingScopeId {
        let scope_id = TimingScopeId(self.scope_labels.len());
        self.scope_labels.push(label.to_string());
        self.active_scopes.insert(scope_id, label.to_string());

        self.begin_scope_internal(encoder, scope_id, label);
        scope_id
    }

    fn begin_scope_internal(
        &mut self,
        encoder: &mut CommandEncoder,
        _scope_id: TimingScopeId,
        label: &str,
    ) {
        let full_label = format!("{}.{}", self.config.label_prefix, label);

        // Insert debug marker for external profilers (RenderDoc, Nsight, RGP)
        if self.config.enable_debug_markers {
            encoder.push_debug_group(&full_label);
        }

        // Insert timestamp query
        if let Some(ref query_set) = self.timestamp_query_set {
            if self.query_index < self.config.max_queries_per_frame * 2 {
                encoder.write_timestamp(query_set, self.query_index);
                self.query_index += 1;
            }
        }

        // Insert pipeline statistics query (begin)
        // Note: pipeline statistics queries not available in current wgpu version
        /*
        if let Some(ref query_set) = self.pipeline_stats_query_set {
            if self.query_index < self.config.max_queries_per_frame * 2 {
                encoder.begin_pipeline_statistics_query(query_set, self.query_index);
            }
        }
        */
    }

    /// End timing scope
    pub fn end_scope(&mut self, encoder: &mut CommandEncoder, scope_id: TimingScopeId) {
        self.end_scope_internal(encoder, scope_id);
    }

    fn end_scope_internal(&mut self, encoder: &mut CommandEncoder, scope_id: TimingScopeId) {
        // End pipeline statistics query
        // Note: pipeline statistics queries not available in current wgpu version
        /*
        if let Some(ref query_set) = self.pipeline_stats_query_set {
            encoder.end_pipeline_statistics_query();
        }
        */

        // Insert end timestamp
        if let Some(ref query_set) = self.timestamp_query_set {
            if self.query_index < self.config.max_queries_per_frame * 2 {
                encoder.write_timestamp(query_set, self.query_index);
                self.query_index += 1;
            }
        }

        // Pop debug marker
        if self.config.enable_debug_markers {
            encoder.pop_debug_group();
        }

        self.active_scopes.remove(&scope_id);
    }

    /// Resolve timing queries and prepare for readback
    pub fn resolve_queries(&mut self, encoder: &mut CommandEncoder) {
        if self.query_index == 0 {
            return; // No queries to resolve
        }

        // Resolve timestamp queries
        if let (Some(ref query_set), Some(ref buffer)) =
            (&self.timestamp_query_set, &self.timestamp_buffer)
        {
            encoder.resolve_query_set(query_set, 0..self.query_index, buffer, 0);

            if let Some(ref readback_buffer) = &self.timestamp_readback_buffer {
                let size = (self.query_index as u64) * std::mem::size_of::<u64>() as u64;
                encoder.copy_buffer_to_buffer(buffer, 0, readback_buffer, 0, size);
            }
        }

        // Resolve pipeline statistics queries
        if let (Some(ref query_set), Some(ref buffer)) =
            (&self.pipeline_stats_query_set, &self.pipeline_stats_buffer)
        {
            encoder.resolve_query_set(query_set, 0..self.query_index, buffer, 0);

            if let Some(ref readback_buffer) = &self.pipeline_stats_readback_buffer {
                let size = (self.query_index as u64) * std::mem::size_of::<u64>() as u64 * 4;
                encoder.copy_buffer_to_buffer(buffer, 0, readback_buffer, 0, size);
            }
        }
    }

    /// Get timing results (async, may not be available immediately)
    pub async fn get_results(&mut self) -> RenderResult<Vec<TimingResult>> {
        let mut results = Vec::new();

        if self.query_index < 2 {
            self.scope_labels.clear();
            self.query_index = 0;
            return Ok(results); // Need at least begin/end pair
        }

        // Read timestamp results
        let timestamps = if let Some(ref readback_buffer) = &self.timestamp_readback_buffer {
            self.read_timestamp_buffer(readback_buffer).await?
        } else {
            Vec::new()
        };

        // Process timestamp pairs into timing results
        let scope_count = timestamps.len().saturating_sub(1) / 2;
        for scope_index in 0..scope_count {
            let begin_idx = scope_index * 2;
            let end_idx = begin_idx + 1;
            let begin_ns = timestamps[begin_idx] as f64 * self.timestamp_period;
            let end_ns = timestamps[end_idx] as f64 * self.timestamp_period;
            let duration_ms = (end_ns - begin_ns) / 1_000_000.0; // Convert ns to ms

            let name = self
                .scope_labels
                .get(scope_index)
                .cloned()
                .unwrap_or_else(|| format!("scope_{}", scope_index));

            results.push(TimingResult {
                name,
                gpu_time_ms: duration_ms as f32,
                timestamp_valid: true,
                pipeline_stats: None, // TODO: Read pipeline stats
            });
        }

        // Reset for next frame
        self.query_index = 0;
        self.scope_labels.clear();

        Ok(results)
    }

    async fn read_timestamp_buffer(&self, buffer: &Buffer) -> RenderResult<Vec<u64>> {
        let size = (self.query_index as u64) * std::mem::size_of::<u64>() as u64;
        let slice = buffer.slice(0..size);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = slice.get_mapped_range();
                let timestamps = bytemuck::cast_slice::<u8, u64>(&data).to_vec();
                drop(data);
                buffer.unmap();
                Ok(timestamps)
            }
            Some(Err(e)) => Err(RenderError::Readback(format!(
                "Failed to map timestamp buffer: {}",
                e
            ))),
            None => Err(RenderError::Readback(
                "Timestamp buffer mapping was cancelled".to_string(),
            )),
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &GpuTimingConfig {
        &self.config
    }

    /// Check if timing features are supported
    pub fn is_supported(&self) -> bool {
        self.supports_timestamps || self.supports_pipeline_stats
    }

    /// Get feature support information
    pub fn get_support_info(&self) -> HashMap<String, bool> {
        let mut info = HashMap::new();
        info.insert("timestamps".to_string(), self.supports_timestamps);
        info.insert("pipeline_stats".to_string(), self.supports_pipeline_stats);
        info.insert(
            "debug_markers".to_string(),
            self.config.enable_debug_markers,
        );
        info
    }
}

/// Convenience macro for timing GPU work
#[macro_export]
macro_rules! gpu_time {
    ($timing_manager:expr, $encoder:expr, $label:expr, $body:expr) => {{
        let scope_id = $timing_manager.begin_scope($encoder, $label);
        let result = $body;
        $timing_manager.end_scope($encoder, scope_id);
        result
    }};
}

/// Create default GPU timing configuration based on device features
pub fn create_default_config(device: &Device) -> GpuTimingConfig {
    let features = device.features();

    GpuTimingConfig {
        enable_timestamps: features.contains(Features::TIMESTAMP_QUERY),
        enable_pipeline_stats: features.contains(Features::PIPELINE_STATISTICS_QUERY),
        enable_debug_markers: true,
        label_prefix: "forge3d".to_string(),
        max_queries_per_frame: 32,
    }
}
