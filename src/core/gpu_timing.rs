//! Q3: GPU profiling markers & timestamp queries
//!
//! Provides GPU timing utilities for performance profiling and debugging.
//! Supports RenderDoc, Nsight Graphics, and RGP markers with configurable
//! timestamp collection for minimal overhead.
//!
//! A submitted batch owns its written query range, submission fence, scope
//! metadata and pending mapping. The two existing slots are reused only after
//! their batch is consumed or dropped. Nonblocking retrieval keeps a pending
//! mapping alive; when both slots are occupied, new work runs without timestamps.
//! `submit` resolves and submits together. `submit_pending` seals timing across
//! command encoders already submitted by a render graph. The legacy resolve /
//! finish methods remain available, but production callers need no slot reset.

use super::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use wgpu::*;

/// Number of frames in flight for the timestamp double buffer.
const SLOTS: usize = 2;

/// Handle for a GPU timing scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimingScopeId {
    generation: u64,
    index: usize,
}

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
    /// Number of draw calls recorded for the scope (0 when unknown)
    pub draw_calls: u32,
    /// Pipeline statistics (if available)
    pub pipeline_stats: Option<PipelineStatistics>,
}

impl TimingResult {
    /// Certificate-safe duration: invalid/absent timestamps are always zero.
    pub fn certificate_gpu_ms(&self) -> f64 {
        if self.timestamp_valid {
            self.gpu_time_ms as f64
        } else {
            0.0
        }
    }
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

#[derive(Default)]
struct RecordedScope {
    name: String,
    draw_calls: u32,
    query_start: Option<u32>,
    finished: bool,
}

#[derive(Default)]
struct RecordingTimingBatch {
    slot: Option<usize>,
    generation: u64,
    query_count: u32,
    scopes: Vec<RecordedScope>,
    active: Vec<usize>,
    resolved: bool,
}

/// Owns the readback and its mapping until the submitted slot is retired.
struct SubmittedTimingBatch {
    slot: usize,
    query_count: u32,
    scopes: Vec<RecordedScope>,
    readback: Arc<TrackedBuffer>,
    // Preserve the manager's native Send + Sync contract. Reading uses exclusive
    // mutable access through get_mut, so this mutex never needs to be locked.
    mapping: Option<Mutex<Receiver<Result<(), BufferAsyncError>>>>,
    mapping_active: bool,
}

impl SubmittedTimingBatch {
    fn start_mapping(&mut self) {
        if self.mapping.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        self.readback
            .slice(0..u64::from(self.query_count) * 8)
            .map_async(MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.mapping = Some(Mutex::new(receiver));
        self.mapping_active = true;
    }

    fn try_read(&mut self) -> RenderResult<Option<Vec<u64>>> {
        match self.mapping.as_mut().expect("mapping started")
            .get_mut().expect("private timing receiver mutex is never locked").try_recv() {
            Ok(Ok(())) => {
                let data = self
                    .readback
                    .slice(0..u64::from(self.query_count) * 8)
                    .get_mapped_range();
                let timestamps = bytemuck::cast_slice::<u8, u64>(&data).to_vec();
                drop(data);
                Ok(Some(timestamps))
            }
            Ok(Err(error)) => {
                // A failed/cancelled map has already returned to the idle state.
                self.mapping_active = false;
                Err(RenderError::Readback(format!(
                    "Failed to map timestamp buffer: {error}"
                )))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.mapping_active = false;
                Err(RenderError::Readback(
                    "Timestamp buffer mapping was cancelled".to_string(),
                ))
            }
        }
    }
}

impl Drop for SubmittedTimingBatch {
    fn drop(&mut self) {
        // Cancels a pending map or releases a completed map on every exit path.
        if self.mapping_active {
            self.readback.unmap();
        }
    }
}

/// GPU timing manager for performance profiling.
/// Recording and submitted metadata have distinct owners; a pending batch
/// prevents its query/readback slot from being selected for new recording.
pub struct GpuTimingManager {
    config: GpuTimingConfig,
    device: Arc<Device>,
    queue: Arc<Queue>,
    timestamp_query_set: [Option<QuerySet>; SLOTS],
    timestamp_buffer: [Option<TrackedBuffer>; SLOTS],
    timestamp_readback_buffer: [Option<Arc<TrackedBuffer>>; SLOTS],
    pipeline_stats_query_set: Option<QuerySet>,
    pipeline_stats_buffer: Option<TrackedBuffer>,
    pipeline_stats_readback_buffer: Option<TrackedBuffer>,
    pipeline_stats_query_index: u32,
    recording: RecordingTimingBatch,
    submitted: VecDeque<SubmittedTimingBatch>,
    supports_timestamps: bool,
    supports_pipeline_stats: bool,
    timestamp_period: f64,
}

impl GpuTimingManager {
    /// Create new GPU timing manager
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: GpuTimingConfig,
    ) -> RenderResult<Self> {
        let features = device.features();

        let supports_timestamps =
            config.enable_timestamps && features.contains(Features::TIMESTAMP_QUERY);
        let supports_pipeline_stats =
            config.enable_pipeline_stats && features.contains(Features::PIPELINE_STATISTICS_QUERY);

        // wgpu 0.19 exposes the timestamp period (ns per tick) on the Queue.
        let timestamp_period = if supports_timestamps {
            queue.get_timestamp_period() as f64
        } else {
            1.0
        };

        let mut manager = Self {
            config: config.clone(),
            device: device.clone(),
            queue,
            timestamp_query_set: [None, None],
            timestamp_buffer: [None, None],
            timestamp_readback_buffer: [None, None],
            pipeline_stats_query_set: None,
            pipeline_stats_buffer: None,
            pipeline_stats_readback_buffer: None,
            pipeline_stats_query_index: 0,
            recording: RecordingTimingBatch {
                slot: Some(0),
                ..Default::default()
            },
            submitted: VecDeque::new(),
            supports_timestamps,
            supports_pipeline_stats,
            timestamp_period,
        };

        // Initialize query sets and buffers
        manager.initialize_queries()?;

        Ok(manager)
    }

    fn initialize_queries(&mut self) -> RenderResult<()> {
        let query_count = self
            .config
            .max_queries_per_frame
            .checked_mul(2)
            .filter(|count| *count <= QUERY_SET_MAX_QUERIES)
            .ok_or_else(|| {
                RenderError::Render(
                    "GPU timing query capacity exceeds wgpu QUERY_SET_MAX_QUERIES".to_string(),
                )
            })?;
        if query_count == 0 {
            return Ok(());
        }

        // Initialize timestamp queries (one independent slot per frame in flight)
        if self.supports_timestamps {
            let buffer_size = (query_count as u64) * std::mem::size_of::<u64>() as u64;
            for slot in 0..SLOTS {
                let query_set = self.device.create_query_set(&QuerySetDescriptor {
                    label: Some("gpu_timing_timestamps"),
                    ty: QueryType::Timestamp,
                    count: query_count,
                });

                let timestamp_buffer = tracked_create_buffer(
                    &self.device,
                    &BufferDescriptor {
                        label: Some(&format!("gpu_timing.resolve_slot{slot}")),
                        size: buffer_size,
                        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                )?;

                let timestamp_readback = tracked_create_buffer(
                    &self.device,
                    &BufferDescriptor {
                        label: Some(&format!("gpu_timing.readback_slot{slot}")),
                        size: buffer_size,
                        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    },
                )?;

                self.timestamp_query_set[slot] = Some(query_set);
                self.timestamp_buffer[slot] = Some(timestamp_buffer);
                self.timestamp_readback_buffer[slot] = Some(Arc::new(timestamp_readback));
            }
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

            let pipeline_stats_buffer = tracked_create_buffer(
                &self.device,
                &BufferDescriptor {
                    label: Some("gpu_timing.pipeline_stats_resolve"),
                    size: stats_buffer_size,
                    usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            )?;

            let pipeline_stats_readback = tracked_create_buffer(
                &self.device,
                &BufferDescriptor {
                    label: Some("gpu_timing.pipeline_stats_readback"),
                    size: stats_buffer_size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )?;

            self.pipeline_stats_query_set = Some(stats_query_set);
            self.pipeline_stats_buffer = Some(pipeline_stats_buffer);
            self.pipeline_stats_readback_buffer = Some(pipeline_stats_readback);
        }

        Ok(())
    }

    /// Begin a new timing scope. Each scope reserves a begin/end pair so nested
    /// scopes, exhausted capacity and unmatched ends cannot mix query labels.
    pub fn begin_scope(&mut self, encoder: &mut CommandEncoder, label: &str) -> TimingScopeId {
        // A compatibility caller may start recording after resolving/submitting
        // externally. Fence that previous batch before selecting another slot.
        if self.recording.resolved {
            self.submit_pending();
        }
        if self.recording.scopes.is_empty() {
            self.recording.slot =
                (0..SLOTS).find(|slot| !self.submitted.iter().any(|batch| batch.slot == *slot));
        }
        let query_start = self.recording.slot.and_then(|slot| {
            if self.timestamp_query_set[slot].is_some()
                && self.recording.query_count / 2 < self.config.max_queries_per_frame
            {
                let index = self.recording.query_count;
                self.recording.query_count += 2;
                Some(index)
            } else {
                None
            }
        });
        let scope_id = TimingScopeId {
            generation: self.recording.generation,
            index: self.recording.scopes.len(),
        };
        self.recording.scopes.push(RecordedScope {
            name: label.to_string(),
            query_start,
            ..Default::default()
        });
        self.recording.active.push(scope_id.index);
        self.begin_scope_internal(encoder, scope_id, label);
        scope_id
    }

    fn begin_scope_internal(
        &mut self,
        encoder: &mut CommandEncoder,
        scope_id: TimingScopeId,
        label: &str,
    ) {
        if self.config.enable_debug_markers {
            encoder.push_debug_group(&format!("{}.{}", self.config.label_prefix, label));
        }
        if let (Some(slot), Some(index)) = (
            self.recording.slot,
            self.recording.scopes[scope_id.index].query_start,
        ) {
            encoder.write_timestamp(self.timestamp_query_set[slot].as_ref().unwrap(), index);
        }
    }

    /// End a timing scope.
    pub fn end_scope(&mut self, encoder: &mut CommandEncoder, scope_id: TimingScopeId) {
        self.end_scope_internal(encoder, scope_id);
    }

    /// End a scope and record its draw count.
    pub fn end_scope_with_draws(
        &mut self,
        encoder: &mut CommandEncoder,
        scope_id: TimingScopeId,
        draw_calls: u32,
    ) {
        if scope_id.generation == self.recording.generation {
            if let Some(scope) = self.recording.scopes.get_mut(scope_id.index) {
                scope.draw_calls = draw_calls;
            }
        }
        self.end_scope_internal(encoder, scope_id);
    }

    fn end_scope_internal(&mut self, encoder: &mut CommandEncoder, scope_id: TimingScopeId) {
        if scope_id.generation != self.recording.generation {
            return;
        }
        let Some(position) = self
            .recording
            .active
            .iter()
            .position(|id| *id == scope_id.index)
        else {
            return; // Duplicate or stale end: never write another scope's query.
        };
        let scope = &mut self.recording.scopes[scope_id.index];
        if let (Some(slot), Some(index)) = (self.recording.slot, scope.query_start) {
            encoder.write_timestamp(self.timestamp_query_set[slot].as_ref().unwrap(), index + 1);
        }
        scope.finished = true;
        self.recording.active.remove(position);
        if self.config.enable_debug_markers {
            encoder.pop_debug_group();
        }
    }

    /// Compatibility entry point for external encoder owners. The caller must
    /// submit this encoder before `finish_frame` or result retrieval. Internal
    /// callers use `submit`, which owns both operations. Only written ranges resolve.
    pub fn resolve_queries(&mut self, encoder: &mut CommandEncoder) {
        if self.recording.resolved {
            return;
        }
        // Complete the reserved ranges of abandoned scopes, but leave them
        // invalid in the result metadata. Never resolve an unwritten query.
        while let Some(index) = self.recording.active.pop() {
            let scope = &self.recording.scopes[index];
            if let (Some(slot), Some(start)) = (self.recording.slot, scope.query_start) {
                encoder
                    .write_timestamp(self.timestamp_query_set[slot].as_ref().unwrap(), start + 1);
            }
            if self.config.enable_debug_markers {
                encoder.pop_debug_group();
            }
        }
        if let Some(slot) = self.recording.slot {
            if self.recording.query_count > 0 {
                let query_set = self.timestamp_query_set[slot].as_ref().unwrap();
                let buffer = self.timestamp_buffer[slot].as_ref().unwrap();
                let readback = self.timestamp_readback_buffer[slot].as_ref().unwrap();
                encoder.resolve_query_set(query_set, 0..self.recording.query_count, buffer, 0);
                encoder.copy_buffer_to_buffer(
                    buffer,
                    0,
                    readback,
                    0,
                    u64::from(self.recording.query_count) * 8,
                );
            }
        }
        // Statistics are not begun by timing scopes. Preserve the guard against
        // resolving an unwritten statistics range (wgpu 0.19 Vulkan device loss).
        if self.pipeline_stats_query_index > 0 {
            if let (Some(query_set), Some(buffer), Some(readback)) = (
                self.pipeline_stats_query_set.as_ref(),
                self.pipeline_stats_buffer.as_ref(),
                self.pipeline_stats_readback_buffer.as_ref(),
            ) {
                encoder.resolve_query_set(query_set, 0..self.pipeline_stats_query_index, buffer, 0);
                encoder.copy_buffer_to_buffer(
                    buffer,
                    0,
                    readback,
                    0,
                    u64::from(self.pipeline_stats_query_index) * 8 * 4,
                );
            }
        }
        self.recording.resolved = true;
    }

    /// Resolve and submit work, transferring its slot and metadata to a batch.
    pub(crate) fn submit(&mut self, mut encoder: CommandEncoder) -> SubmissionIndex {
        self.resolve_queries(&mut encoder);
        let submission = self.queue.submit(Some(encoder.finish()));
        let generation = self.recording.generation.wrapping_add(1);
        let recording = std::mem::replace(
            &mut self.recording,
            RecordingTimingBatch {
                generation,
                ..Default::default()
            },
        );
        if let Some(slot) = recording.slot.filter(|_| recording.query_count > 0) {
            self.submitted.push_back(SubmittedTimingBatch {
                slot,
                query_count: recording.query_count,
                scopes: recording.scopes,
                readback: self.timestamp_readback_buffer[slot]
                    .as_ref()
                    .unwrap()
                    .clone(),
                mapping: None,
                mapping_active: false,
            });
        }
        self.pipeline_stats_query_index = 0;
        submission
    }

    /// Seal scopes written on already-submitted encoders (e.g. a render graph).
    /// Queue ordering ensures this resolution follows all of that GPU work.
    pub(crate) fn submit_pending(&mut self) {
        if !self.recording.scopes.is_empty() || self.recording.resolved {
            let encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("gpu_timing.submit_pending"),
                });
            self.submit(encoder);
        }
    }

    /// Compatibility fence for callers that submitted a resolved encoder.
    /// Slot retirement is owned by the submitted batch, never by this call.
    pub fn finish_frame(&mut self) {
        self.submit_pending();
    }

    /// Consume submitted results, waiting for their GPU work.
    pub async fn get_results(&mut self) -> RenderResult<Vec<TimingResult>> {
        self.get_results_blocking()
    }

    /// Consume ready batches without waiting. Pending mappings and labels stay
    /// alive for the next call, and their slots cannot be overwritten.
    pub fn try_get_results(&mut self) -> RenderResult<Vec<TimingResult>> {
        self.read_submitted(false)
    }

    /// Consume all submitted timing, fencing legacy externally submitted work.
    pub fn get_results_blocking(&mut self) -> RenderResult<Vec<TimingResult>> {
        self.submit_pending();
        self.read_submitted(true)
    }

    fn read_submitted(&mut self, blocking: bool) -> RenderResult<Vec<TimingResult>> {
        for batch in &mut self.submitted {
            batch.start_mapping();
        }
        let mut results = Vec::new();
        while let Some(batch) = self.submitted.front_mut() {
            // wgpu 0.19's blocking wait can retire a submission after its
            // internal HAL timeout, before Metal has finished executing it.
            // Polling observes the real fence; the mapping callback owns the
            // completion proof for this batch in both API modes.
            self.device.poll(Maintain::Poll);
            match batch.try_read() {
                Ok(Some(timestamps)) => {
                    let batch = self.submitted.pop_front().unwrap();
                    results.extend(timing_results_from_timestamps(
                        &batch.scopes,
                        &timestamps,
                        self.timestamp_period,
                    ));
                    // Dropping the batch unmaps before its slot becomes reusable.
                }
                Ok(None) if !blocking => break,
                Ok(None) => std::thread::yield_now(),
                Err(error) => {
                    self.submitted.pop_front(); // Drop/unmap even on readback failure.
                    return Err(error);
                }
            }
        }
        Ok(results)
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

fn timing_results_from_timestamps(
    scopes: &[RecordedScope],
    timestamps: &[u64],
    timestamp_period: f64,
) -> Vec<TimingResult> {
    scopes
        .iter()
        .map(|scope| {
            let pair = scope.query_start.and_then(|index| {
                Some((
                    *timestamps.get(index as usize)?,
                    *timestamps.get(index as usize + 1)?,
                ))
            });
            let timestamp_valid =
                scope.finished && pair.is_some_and(|(begin, end)| begin != 0 && end >= begin);
            let duration_ms = if timestamp_valid {
                let (begin, end) = pair.unwrap();
                (end - begin) as f64 * timestamp_period / 1_000_000.0
            } else {
                0.0
            };
            let name = scope.name.clone();
            let draw_calls = scope.draw_calls;
            TimingResult {
                name,
                gpu_time_ms: duration_ms as f32,
                timestamp_valid,
                draw_calls,
                pipeline_stats: None,
            }
        })
        .collect()
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

/// One-shot per-render GPU timing for certified offscreen render paths
/// (CENSOR audit F-04).
///
/// Wraps an optional [`GpuTimingManager`]: present when the initialized global
/// device granted `TIMESTAMP_QUERY`, absent otherwise. Callers wrap each GPU
/// pass in `begin`/`end` and use `submit` for an owned encoder. When a render
/// graph submits borrowed encoders, finalization owns a following resolve
/// submission. [`OneShotTiming::record_into_certificate`] then
/// blocking-reads the timestamps and records one certificate pass per scope
/// with its live `gpu_ms`. When timing is unavailable the caller must fall
/// back to recording the same pass labels (same order) with `gpu_ms == 0.0`
/// so certificate pass lists stay identical across adapters — 0.0 is the
/// honest "no timestamps" value, never a fabricated timing.
pub struct OneShotTiming {
    manager: Option<GpuTimingManager>,
    failure: Option<String>,
    labels: Vec<String>,
}

impl OneShotTiming {
    /// Build a timing manager from the already-initialized global GPU context.
    /// Returns an inert instance when no context exists, `TIMESTAMP_QUERY` was
    /// not granted, or manager construction fails (each pass then records 0.0
    /// and the `capability_absent` degradation already explains why).
    pub fn for_current_device() -> Self {
        let Some(ctx) = crate::core::gpu::ctx_if_initialized() else {
            return Self::inert();
        };
        Self::for_device(ctx.device.clone(), ctx.queue.clone())
    }

    /// Build a timing manager for an explicit device/queue pair — for render
    /// paths that own their device instead of using the global context (e.g.
    /// `TerrainSpike`). Returns an inert instance when `TIMESTAMP_QUERY` was
    /// not granted or manager construction fails (each pass then records 0.0).
    pub fn for_device(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        if !device.features().contains(Features::TIMESTAMP_QUERY) {
            return Self::inert();
        }
        let config = GpuTimingConfig {
            enable_timestamps: true,
            // Timestamps only: enabling statistics here would resolve a
            // never-written statistics range (see Scene::take_render_timing).
            enable_pipeline_stats: false,
            enable_debug_markers: false,
            label_prefix: "forge3d".to_string(),
            max_queries_per_frame: 32,
        };
        match GpuTimingManager::new(device, queue, config) {
            Ok(manager) => Self {
                manager: Some(manager),
                failure: None,
                labels: Vec::new(),
            },
            Err(e) => Self {
                manager: None,
                failure: Some(format!("timing manager construction failed: {e}")),
                labels: Vec::new(),
            },
        }
    }

    fn inert() -> Self {
        Self {
            manager: None,
            failure: None,
            labels: Vec::new(),
        }
    }

    /// True when live timestamps will be recorded (used by callers to decide
    /// whether the 0.0 fallback pass records are needed).
    pub fn is_live(&self) -> bool {
        self.manager.is_some()
    }

    /// Open a timing scope around a GPU pass. Must be called with the encoder
    /// outside any active render/compute pass.
    pub fn begin(&mut self, encoder: &mut CommandEncoder, label: &str) -> Option<TimingScopeId> {
        self.labels.push(label.to_string());
        self.manager.as_mut().map(|t| t.begin_scope(encoder, label))
    }

    /// Close a scope opened by [`OneShotTiming::begin`], recording draw count.
    pub fn end(
        &mut self,
        encoder: &mut CommandEncoder,
        scope: Option<TimingScopeId>,
        draw_calls: u32,
    ) {
        if let (Some(t), Some(id)) = (self.manager.as_mut(), scope) {
            t.end_scope_with_draws(encoder, id, draw_calls);
        }
    }

    /// Compatibility resolution for an externally owned encoder. New callers
    /// use `submit` or let certificate finalization seal graph-submitted work.
    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        if let Some(t) = self.manager.as_mut() {
            t.resolve_queries(encoder);
        }
    }

    /// Submit an owned encoder with its timing resolution. Inert timing still
    /// submits the caller's real GPU work on the supplied queue.
    pub(crate) fn submit(&mut self, queue: &Queue, encoder: CommandEncoder) -> SubmissionIndex {
        match self.manager.as_mut() {
            Some(manager) => manager.submit(encoder),
            None => queue.submit(Some(encoder.finish())),
        }
    }

    /// Blocking-read the resolved timestamps and record every VALID scope into
    /// the active certificate capture with its live timing. Returns `true`
    /// when live passes were recorded; on `false` the caller records its 0.0
    /// fallback passes instead.
    pub fn record_into_certificate(mut self) -> bool {
        if let Some(consequence) = self.failure.as_deref() {
            self.record_failure(consequence);
            return false;
        }
        let Some(manager) = self.manager.as_mut() else {
            return false;
        };
        match manager.get_results_blocking() {
            Ok(results) if !results.is_empty() && results.len() == self.labels.len() => {
                for result in &results {
                    if !result.timestamp_valid {
                        crate::core::degradation::record_degradation(
                            "timing_unavailable",
                            &result.name,
                            "timestamp query returned an unwritten or non-monotonic interval",
                        );
                    }
                    crate::core::certificate::record_pass(
                        &result.name,
                        result.certificate_gpu_ms(),
                        result.draw_calls,
                    );
                }
                true
            }
            Ok(_) => {
                self.record_failure("timestamp query results did not cover every recorded pass");
                false
            }
            Err(e) => {
                self.record_failure(&format!("timestamp query readback failed: {e}"));
                false
            }
        }
    }

    fn record_failure(&self, consequence: &str) {
        for label in &self.labels {
            crate::core::degradation::record_degradation("timing_unavailable", label, consequence);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CENSOR Task 4: end-to-end timestamp measurement over a real GPU submit.
    ///
    /// Times a buffer-to-buffer copy inside a begin/end scope, resolves, submits,
    /// and reads back with `get_results_blocking`. Skips cleanly when no adapter
    /// is available or TIMESTAMP_QUERY was not granted.
    #[test]
    fn gpu_timing_double_buffered_measures_copy() {
        let instance = Instance::new(InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            ..Default::default()
        })) else {
            eprintln!("[gpu_timing test] no GPU adapter available; skipping");
            return;
        };
        let info = adapter.get_info();
        let query_features = adapter.features()
            & (Features::TIMESTAMP_QUERY | Features::PIPELINE_STATISTICS_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("gpu_timing_query_features"),
                required_features: query_features,
                required_limits: Limits::default(),
            }, None,
        )).expect("device with advertised query features");
        eprintln!(
            "[gpu_timing test] adapter={info:?}, features={:?}",
            device.features()
        );
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let config = create_default_config(&device);
        if !config.enable_timestamps {
            eprintln!("[gpu_timing test] TIMESTAMP_QUERY not granted; skipping");
            return;
        }

        let mut manager = GpuTimingManager::new(device.clone(), queue.clone(), config)
            .expect("create timing manager");
        assert!(
            manager.is_supported(),
            "timing must be supported once TIMESTAMP_QUERY is granted"
        );

        // Two small buffers to copy between; the copy is what we time.
        let src = tracked_create_buffer(
            &device,
            &BufferDescriptor {
                label: Some("gpu_timing_test_src"),
                size: 4096,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .expect("alloc");
        let dst = tracked_create_buffer(
            &device,
            &BufferDescriptor {
                label: Some("gpu_timing_test_dst"),
                size: 4096,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .expect("alloc");

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_timing"),
        });

        let scope = manager.begin_scope(&mut encoder, "test_copy");
        // Loop the copy so the timed interval is not sub-tick on fast adapters.
        for _ in 0..100 {
            encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 4096);
        }
        manager.end_scope_with_draws(&mut encoder, scope, 100);

        manager.resolve_queries(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(Maintain::Wait);

        let results = manager.get_results_blocking().expect("get results");
        assert_eq!(results.len(), 1, "expected exactly one timed scope");
        let r = &results[0];
        assert_eq!(r.name, "test_copy");
        assert!(r.timestamp_valid, "timestamp must be valid");
        assert_eq!(r.draw_calls, 100, "draw_calls must round-trip");
        // Timestamps are monotonic; a real interval is >= 0. The 100x copy makes
        // it typically > 0, but we assert >= 0.0 to stay robust across adapters
        // whose period rounds a tiny copy to zero.
        assert!(
            r.gpu_time_ms >= 0.0,
            "gpu_time_ms must be non-negative, got {}",
            r.gpu_time_ms
        );

        // CENSOR Task 4 regression: a second offline single-shot round
        // (resolve -> submit -> get_results_blocking) WITHOUT an intervening
        // `finish_frame()` must see exactly the new scope, not the old one
        // re-reported on top of it. This is the documented single-shot
        // pattern an offline-render certificate path will use once per
        // render, multiple renders per process.
        let mut encoder2 = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_timing_2"),
        });
        let scope2 = manager.begin_scope(&mut encoder2, "test_copy");
        encoder2.copy_buffer_to_buffer(&src, 0, &dst, 0, 4096);
        manager.end_scope_with_draws(&mut encoder2, scope2, 1);
        manager.resolve_queries(&mut encoder2);
        queue.submit(std::iter::once(encoder2.finish()));
        device.poll(Maintain::Wait);

        let results2 = manager
            .get_results_blocking()
            .expect("get results (second single-shot read)");
        assert_eq!(
            results2.len(),
            1,
            "get_results_blocking must not accumulate scopes across calls without finish_frame"
        );
        assert_eq!(results2[0].name, "test_copy");

        // Consecutive owned submissions keep two independent pending batches.
        // Delay one mapping notification deterministically while retaining the
        // real map callback/result. The sample must survive a nonblocking read.
        for label in ["batch.first", "batch.second"] {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            let scope = manager.begin_scope(&mut encoder, label);
            encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 4096);
            manager.end_scope_with_draws(&mut encoder, scope, 1);
            manager.submit(encoder);
        }
        assert_eq!(manager.submitted.len(), SLOTS);
        let batch = manager.submitted.front_mut().unwrap();
        batch.start_mapping();
        let actual_mapping = batch.mapping.take();
        let (_delayed_sender, delayed_receiver) = mpsc::channel();
        batch.mapping = Some(Mutex::new(delayed_receiver));
        assert!(manager.try_get_results().unwrap().is_empty());
        assert_eq!(
            manager.submitted.len(),
            SLOTS,
            "pending reads cannot retire slots"
        );
        manager.submitted.front_mut().unwrap().mapping = actual_mapping;

        // A third frame must execute without overwriting either pending slot.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let skipped = manager.begin_scope(&mut encoder, "batch.no_free_slot");
        assert!(manager.recording.slot.is_none());
        manager.end_scope(&mut encoder, skipped);
        manager.submit(encoder);
        assert_eq!(manager.submitted.len(), SLOTS);
        let consecutive = manager.get_results_blocking().unwrap();
        assert_eq!(
            consecutive
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>(),
            ["batch.first", "batch.second"]
        );
        assert!(consecutive
            .iter()
            .all(|r| r.timestamp_valid && r.draw_calls == 1));
        assert!(
            manager.get_results_blocking().unwrap().is_empty(),
            "results are consumed once"
        );

        // Empty submissions and empty scopes retire without stale labels.
        manager.submit(device.create_command_encoder(&CommandEncoderDescriptor::default()));
        assert!(manager.get_results_blocking().unwrap().is_empty());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = manager.begin_scope(&mut encoder, "batch.empty_scope");
        manager.end_scope(&mut encoder, scope);
        manager.submit(encoder);
        device.poll(Maintain::Wait);
        let ready = manager.try_get_results().unwrap();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].name, "batch.empty_scope");
        assert!(ready[0].timestamp_valid);
        assert!(manager.try_get_results().unwrap().is_empty());

        // Nested scopes reserve pairs, overflow stays invalid, and an unfinished
        // scope resolves only written queries without claiming a valid duration.
        manager.config.max_queries_per_frame = 2;
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let outer = manager.begin_scope(&mut encoder, "batch.outer");
        let inner = manager.begin_scope(&mut encoder, "batch.inner");
        manager.end_scope(&mut encoder, inner);
        let overflow = manager.begin_scope(&mut encoder, "batch.overflow");
        manager.end_scope(&mut encoder, overflow);
        // Deliberately leave outer unfinished: submission closes it as invalid.
        manager.submit(encoder);
        let nested = manager.get_results_blocking().unwrap();
        assert_eq!(nested.len(), 3);
        assert!(!nested[0].timestamp_valid && nested[0].gpu_time_ms == 0.0);
        assert_eq!(nested[1].name, "batch.inner");
        assert!(nested[1].timestamp_valid);
        assert!(!nested[2].timestamp_valid && nested[2].gpu_time_ms == 0.0);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let fresh = manager.begin_scope(&mut encoder, "batch.fresh");
        manager.end_scope(&mut encoder, outer); // Stale generation cannot end fresh.
        manager.end_scope(&mut encoder, fresh);
        manager.submit(encoder);
        assert_eq!(
            manager.get_results_blocking().unwrap()[0].name,
            "batch.fresh"
        );

        // Render-graph callers submit scope-writing encoders themselves. The
        // timing owner seals their completed work with its own resolution fence.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = manager.begin_scope(&mut encoder, "batch.graph");
        manager.end_scope(&mut encoder, scope);
        queue.submit(Some(encoder.finish()));
        let graph = pollster::block_on(manager.get_results()).unwrap();
        assert_eq!(graph.len(), 1);
        assert_eq!(graph[0].name, "batch.graph");
        assert!(graph[0].timestamp_valid);

        // The public external-submission compatibility path owns retirement too.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = manager.begin_scope(&mut encoder, "batch.legacy_finish");
        manager.end_scope(&mut encoder, scope);
        manager.resolve_queries(&mut encoder);
        queue.submit(Some(encoder.finish()));
        manager.finish_frame();
        let legacy = manager.get_results_blocking().unwrap();
        assert_eq!(legacy.len(), 1);
        assert_eq!(legacy[0].name, "batch.legacy_finish");
        assert!(legacy[0].timestamp_valid);

        // Dropping a mapped/pending owner must cancel mapping before reuse.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = manager.begin_scope(&mut encoder, "batch.cancelled");
        manager.end_scope(&mut encoder, scope);
        manager.submit(encoder);
        manager.submitted.front_mut().unwrap().start_mapping();
        let cancelled = manager.submitted.pop_front().unwrap();
        drop(cancelled);
        device.poll(Maintain::Wait);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = manager.begin_scope(&mut encoder, "batch.after_cancel");
        manager.end_scope(&mut encoder, scope);
        manager.submit(encoder);
        assert!(manager.get_results_blocking().unwrap()[0].timestamp_valid);

        // Readback failure must also retire/cancel the owner before slot reuse.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = manager.begin_scope(&mut encoder, "batch.map_failure");
        manager.end_scope(&mut encoder, scope);
        manager.submit(encoder);
        let batch = manager.submitted.front_mut().unwrap();
        batch.start_mapping();
        batch.readback.unmap(); // Real cancellation delivers an error callback.
        device.poll(Maintain::Wait);
        assert!(manager.try_get_results().is_err());
        assert!(manager.submitted.is_empty());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = manager.begin_scope(&mut encoder, "batch.after_failure");
        manager.end_scope(&mut encoder, scope);
        manager.submit(encoder);
        assert!(manager.get_results_blocking().unwrap()[0].timestamp_valid);

        // One-shot callers use the same owner for live query submission/readback.
        let mut one_shot = OneShotTiming::for_device(device.clone(), queue.clone());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = one_shot.begin(&mut encoder, "batch.one_shot");
        encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 4096);
        one_shot.end(&mut encoder, scope, 1);
        one_shot.submit(&queue, encoder);
        assert!(one_shot.record_into_certificate());

        // The inert OneShotTiming path still submits executed GPU work.
        let mut inert = OneShotTiming::inert();
        let encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let submitted = inert.submit(&queue, encoder);
        device.poll(Maintain::WaitForSubmissionIndex(submitted));
        assert!(!inert.record_into_certificate());

        // Request a real device without optional features: requested timestamps
        // and statistics must degrade to absent instead of resolving unwritten queries.
        let (unsupported_device, unsupported_queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("gpu_timing_no_query_features"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
            },
            None,
        ))
        .expect("device without optional features");
        eprintln!(
            "[gpu_timing unsupported] adapter={:?}, features={:?}",
            adapter.get_info(),
            unsupported_device.features()
        );
        let unsupported_device = Arc::new(unsupported_device);
        let unsupported_queue = Arc::new(unsupported_queue);
        let mut unsupported = GpuTimingManager::new(
            unsupported_device.clone(),
            unsupported_queue.clone(),
            GpuTimingConfig {
                enable_pipeline_stats: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!unsupported.is_supported());
        let mut encoder =
            unsupported_device.create_command_encoder(&CommandEncoderDescriptor::default());
        let scope = unsupported.begin_scope(&mut encoder, "batch.unsupported");
        unsupported.end_scope(&mut encoder, scope);
        unsupported.submit(encoder);
        assert!(unsupported.try_get_results().unwrap().is_empty());
        assert!(unsupported.get_results_blocking().unwrap().is_empty());
        drop(unsupported);
        unsupported_device.poll(Maintain::Wait);
        drop(unsupported_queue);
        drop(unsupported_device);

        // Explicitly disabled queries and zero capacity preserve the marker-only path.
        for config in [
            GpuTimingConfig {
                enable_timestamps: false,
                enable_pipeline_stats: false,
                ..Default::default()
            },
            GpuTimingConfig {
                max_queries_per_frame: 0,
                ..Default::default()
            },
        ] {
            let mut disabled =
                GpuTimingManager::new(device.clone(), queue.clone(), config).unwrap();
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            let scope = disabled.begin_scope(&mut encoder, "batch.disabled");
            disabled.end_scope(&mut encoder, scope);
            disabled.submit(encoder);
            assert!(disabled.get_results_blocking().unwrap().is_empty());
        }
        assert!(GpuTimingManager::new(
            device.clone(),
            queue.clone(),
            GpuTimingConfig {
                max_queries_per_frame: u32::MAX,
                ..Default::default()
            }
        )
        .is_err());

        // Teardown: drop GPU resources explicitly and poll so the backend
        // processes destruction work, then deliberately leak the device/queue.
        // On this driver stack (wgpu 0.19 / Windows), dropping a device that
        // owned timestamp query sets hangs inside the backend even after a
        // final poll; the process is exiting anyway, so leaking is safe and
        // keeps the test harness from deadlocking.
        drop(results);
        drop(results2);
        drop(manager);
        drop(src);
        drop(dst);
        device.poll(Maintain::Wait);
        std::mem::forget(device);
        std::mem::forget(queue);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_timing_manager_preserves_native_thread_traits() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuTimingManager>();
    }

    #[test]
    fn timestamp_results_preserve_labels_and_reject_unwritten_or_reversed_pairs() {
        let scopes = [
            RecordedScope {
                name: "valid".into(),
                query_start: Some(0),
                finished: true,
                draw_calls: 3,
            },
            RecordedScope {
                name: "unwritten".into(),
                query_start: Some(2),
                finished: true,
                ..Default::default()
            },
            RecordedScope {
                name: "reversed".into(),
                query_start: Some(4),
                finished: true,
                ..Default::default()
            },
            RecordedScope {
                name: "overflow".into(),
                finished: true,
                ..Default::default()
            },
        ];
        let results = timing_results_from_timestamps(&scopes, &[10, 20, 0, 12, 20, 10], 1.0);
        assert_eq!(results.len(), scopes.len());
        assert!(results[0].timestamp_valid);
        assert_eq!(results[0].draw_calls, 3);
        for result in &results[1..] {
            assert!(!result.timestamp_valid);
            assert_eq!(result.gpu_time_ms, 0.0);
            assert_eq!(result.certificate_gpu_ms(), 0.0);
        }
    }

    #[test]
    fn certificate_gpu_ms_zeros_invalid_timestamps() {
        assert_eq!(
            TimingResult {
                gpu_time_ms: 3.25,
                timestamp_valid: false,
                ..Default::default()
            }
            .certificate_gpu_ms(),
            0.0
        );
        assert_eq!(
            TimingResult {
                gpu_time_ms: 3.25,
                timestamp_valid: true,
                ..Default::default()
            }
            .certificate_gpu_ms(),
            3.25
        );
    }

    #[test]
    fn one_shot_failure_is_render_local_degradation() {
        crate::core::degradation::clear_degradations();
        crate::core::degradation::begin_degradation_capture();
        let timing = OneShotTiming {
            manager: None,
            failure: Some("forced readback failure".to_string()),
            labels: vec!["test.pass".to_string()],
        };
        assert!(!timing.record_into_certificate());
        let first = crate::core::degradation::finish_degradation_capture();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].kind, "timing_unavailable");
        assert_eq!(first[0].name, "test.pass");
        assert_eq!(first[0].consequence, "forced readback failure");

        crate::core::degradation::begin_degradation_capture();
        assert!(crate::core::degradation::finish_degradation_capture().is_empty());
        crate::core::degradation::clear_degradations();
    }
}
