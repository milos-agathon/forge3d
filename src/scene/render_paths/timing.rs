// CENSOR Task 9: Scene render-path GPU-pass timing.
//
// Mirrors the terrain renderer's timing plumbing (take/store/record around a
// `Mutex<Option<GpuTimingManager>>`) so a `Scene.render_rgba` / `render_png`
// produces its own RenderCertificate capture with live per-pass GPU timings.
// One render = one capture: `render_*_impl` calls `begin_render_capture`,
// records every timed pass, and `finish_render_capture`, replacing the last
// completed capture regardless of which native path (terrain or scene)
// produced it.

impl Scene {
    pub(super) fn record_terrain_shader_use() {
        let label = if crate::core::gpu::ctx_if_initialized()
            .is_some_and(|ctx| ctx.device.limits().max_bind_groups >= 6)
        {
            "vf.Terrain.shader.full"
        } else {
            "vf.Terrain.shader.minimal"
        };
        crate::core::shader_registry::record_shader_use(label);
    }

    pub(super) fn begin_certificate_capture(
        &self,
        entry_point: &str,
    ) -> (
        crate::core::certificate::RenderCaptureGuard,
        crate::core::resource_tracker::AllocationOwnerGuard,
    ) {
        let allocation_scope = self.allocation_owner.activate();
        let render_capture = crate::core::certificate::begin_render_capture_with_resources(
            entry_point,
            &[self.allocation_owner.id()],
        );
        (render_capture, allocation_scope)
    }

    pub(super) fn finish_certificate_capture(
        &self,
        capture: crate::core::certificate::RenderCaptureGuard,
    ) {
        capture.finish();
    }

    /// Take the render-timing manager out of the scene, lazily constructing it
    /// the first time when the device granted `TIMESTAMP_QUERY`. Returns `None`
    /// when timestamps are unavailable (the certificate then reports the passes
    /// with `gpu_ms == 0`). Returned via [`Scene::store_render_timing`].
    pub(super) fn take_render_timing(&self) -> Option<crate::core::gpu_timing::GpuTimingManager> {
        let mut guard = self.render_timing.lock().ok()?;
        if guard.is_none() {
            let g = crate::core::gpu::ctx_if_initialized()?;
            if !g
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
            {
                return None;
            }
            // Timestamps only: this path never issues pipeline-statistics
            // queries, so enabling that query set would make `resolve_queries`
            // resolve a never-written statistics range and lose the device on
            // adapters that also advertise PIPELINE_STATISTICS_QUERY.
            let config = crate::core::gpu_timing::GpuTimingConfig {
                enable_timestamps: true,
                enable_pipeline_stats: false,
                enable_debug_markers: false,
                label_prefix: "forge3d".to_string(),
                max_queries_per_frame: 32,
            };
            match crate::core::gpu_timing::GpuTimingManager::new(
                g.device.clone(),
                g.queue.clone(),
                config,
            ) {
                Ok(manager) => return Some(manager),
                Err(e) => {
                    log::warn!("failed to create Scene GPU timing manager: {e}");
                    return None;
                }
            }
        }
        guard.take()
    }

    /// Return a timing manager taken by [`Scene::take_render_timing`].
    pub(super) fn store_render_timing(
        &self,
        manager: Option<crate::core::gpu_timing::GpuTimingManager>,
    ) {
        if let Some(manager) = manager {
            if let Ok(mut guard) = self.render_timing.lock() {
                *guard = Some(manager);
            }
        }
    }

    /// Finalize a render's timing: resolve+read the timestamps and record each
    /// timed pass into the global certificate capture. Must be called AFTER the
    /// encoder that recorded `resolve_queries` was submitted (the readback
    /// encoder does that here). Consumes the manager's current slot.
    pub(super) fn record_render_timings(
        &self,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    ) {
        if let Some(manager) = timing.as_mut() {
            match manager.get_results_blocking() {
                Ok(results) => {
                    for result in results {
                        // Invalid timestamps are recorded as 0.0, never as a
                        // garbage delta (CENSOR audit F-04).
                        crate::core::certificate::record_pass(
                            &result.name,
                            result.certificate_gpu_ms(),
                            result.draw_calls,
                        );
                    }
                }
                Err(e) => log::warn!("Scene GPU timing readback failed: {e}"),
            }
        }
    }
}

/// Open a timing scope around a Scene GPU pass when timing is active. The
/// returned id is threaded to [`scene_ts_end`]; `None` when timing is
/// unavailable. Must be called with the encoder outside any active render pass.
pub(super) fn scene_ts_begin(
    timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
) -> Option<crate::core::gpu_timing::TimingScopeId> {
    timing.as_mut().map(|t| t.begin_scope(encoder, label))
}

/// Close a timing scope opened by [`scene_ts_begin`], recording its draw count.
pub(super) fn scene_ts_end(
    timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    encoder: &mut wgpu::CommandEncoder,
    scope: Option<crate::core::gpu_timing::TimingScopeId>,
    draw_calls: u32,
) {
    if let (Some(t), Some(id)) = (timing.as_mut(), scope) {
        t.end_scope_with_draws(encoder, id, draw_calls);
    }
}
