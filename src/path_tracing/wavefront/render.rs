use super::*;
#[cfg(feature = "extension-module")]
use crate::path_tracing::TracerParams;

impl WavefrontScheduler {
    #[cfg(feature = "extension-module")]
    pub fn render_frame(
        &mut self,
        _scene: &Scene,
        _params: &TracerParams,
        _accum_buffer: &Buffer,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wavefront-frame"),
            });
        self.queue_buffers.reset_counters(&self.queue, &mut encoder);
        if self.restir_enabled {
            self.dispatch_restir_init(&mut encoder, uniforms_buffer, scene_bind_group)?;
            self.dispatch_restir_temporal(&mut encoder, uniforms_buffer, scene_bind_group)?;
            if self.restir_spatial_enabled {
                self.dispatch_restir_spatial(&mut encoder, uniforms_buffer, scene_bind_group)?;
                use std::mem::swap;
                swap(&mut self.restir_prev, &mut self.restir_out);
            }
        }
        self.dispatch_raygen(
            &mut encoder,
            uniforms_buffer,
            scene_bind_group,
            accum_bind_group,
        )?;
        let max_iterations = MAX_DEPTH * 2;
        for _iteration in 0..max_iterations {
            let ray_count =
                self.queue_buffers
                    .get_active_ray_count(&self.device, &self.queue, &mut encoder)?;
            if ray_count == 0 {
                break;
            }
            self.dispatch_intersect(&mut encoder, uniforms_buffer, scene_bind_group)?;
            self.dispatch_shade(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
            self.dispatch_shadow(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
            self.dispatch_scatter(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.frame_index += 1;
        Ok(())
    }

    /// Render one wavefront frame and return the number of wavefront
    /// iterations executed (each iteration = intersect + shade + shadow +
    /// scatter over the currently active ray wave). A genuine path-traced
    /// frame executes >= 2 iterations: the primary wave plus at least one
    /// BSDF-sampled bounce wave. Errors if the active-ray count is zero
    /// immediately after raygen — that means the queue state or the
    /// active-count readback is broken, and hiding it behind a one-iteration
    /// fallback would silently degrade the render into a primary-only pass.
    ///
    /// `timing`, when supplied as `(one_shot, label, draw_calls)`, opens a
    /// certificate timing scope (CENSOR F-04) around this frame's FIRST
    /// wavefront iteration (intersect + shade + shadow + scatter of the
    /// primary wave) — the representative encoder region for this pipeline;
    /// a whole frame spans multiple encoders because `read_ray_queue_header`
    /// swaps + submits mid-frame, so it cannot be bracketed on one encoder.
    /// The scope's queries are resolved on this frame's final encoder before
    /// its submit; the caller reads them back after all frames complete.
    pub fn render_frame_simple(
        &mut self,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
        mut timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str, u32)>,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wavefront-frame-simple"),
            });
        self.queue_buffers.reset_counters(&self.queue, &mut encoder);
        if self.restir_enabled {
            self.dispatch_restir_init(&mut encoder, uniforms_buffer, scene_bind_group)?;
            self.dispatch_restir_temporal(&mut encoder, uniforms_buffer, scene_bind_group)?;
            if self.restir_spatial_enabled {
                self.dispatch_restir_spatial(&mut encoder, uniforms_buffer, scene_bind_group)?;
                use std::mem::swap;
                swap(&mut self.restir_prev, &mut self.restir_out);
            }
        }
        self.dispatch_raygen(
            &mut encoder,
            uniforms_buffer,
            scene_bind_group,
            accum_bind_group,
        )?;
        let max_iterations = MAX_DEPTH * 2;
        let mut iterations_executed = 0u32;
        // Rays consumed in completed iterations. The ray queue is append-only
        // within a frame (raygen + each scatter wave push at increasing
        // indices); every iteration drains all rays present when it starts,
        // so the consumed prefix after iteration k is the in_count that was
        // current when iteration k was dispatched. out_count itself is NOT
        // trustworthy between iterations: the persistent-thread pop pattern
        // over-increments it past in_count once the wave drains, which is
        // exactly the corruption that used to silently reduce this loop to a
        // single (direct-lighting-only) iteration.
        let mut consumed_rays = 0u32;
        for iteration in 0..max_iterations {
            let header = self.queue_buffers.read_ray_queue_header(
                &self.device,
                &self.queue,
                &mut encoder,
            )?;
            if header.in_count > header.capacity {
                // Scatter's atomicAdd bumps in_count even when the capacity
                // guard drops the write, so this detects silently lost bounce
                // rays (a biased render) instead of hiding them.
                return Err(format!(
                    "wavefront ray queue overflow: {} rays pushed into capacity {}",
                    header.in_count, header.capacity
                )
                .into());
            }
            let active = header.in_count.saturating_sub(consumed_rays);
            if active == 0 {
                if iteration == 0 {
                    // Raygen always pushes width*height primary rays; zero
                    // active rays here means the queue state or the readback
                    // is broken. Fail loudly instead of hiding it behind a
                    // one-iteration fallback.
                    return Err("wavefront raygen produced zero active rays; \
                                queue state or active-count readback is broken"
                        .into());
                }
                break;
            }
            // read_ray_queue_header stalled on all prior GPU work, so these
            // header writes land before this iteration's dispatches.
            self.queue_buffers
                .begin_iteration(&self.queue, consumed_rays);
            // The first iteration's dispatches all land on the encoder that
            // read_ray_queue_header just swapped in, so begin/end bracket the
            // timed region on the same encoder that executes it.
            let timing_scope = if iteration == 0 {
                timing
                    .as_mut()
                    .and_then(|(t, label, _)| t.begin(&mut encoder, label))
            } else {
                None
            };
            self.dispatch_intersect(&mut encoder, uniforms_buffer, scene_bind_group)?;
            self.dispatch_shade(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
            self.dispatch_shadow(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
            self.dispatch_scatter(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
            // `end` is a no-op when `timing_scope` is None (iterations > 0).
            if let Some((t, _, draw_calls)) = timing.as_mut() {
                t.end(&mut encoder, timing_scope, *draw_calls);
            }
            // This iteration consumes every ray present when it started.
            consumed_rays = header.in_count;
            iterations_executed += 1;
            // NOTE: no queue compaction here — compaction would relocate
            // queue entries and invalidate the consumed-prefix bookkeeping
            // (see the removal note in dispatch.rs).
        }
        // Resolve on this frame's final encoder: the end timestamp was written
        // on an encoder already submitted (or on this one), so the resolved
        // range is complete once this submit lands.
        if let Some((t, _, _)) = timing.as_mut() {
            t.resolve(&mut encoder);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.frame_index += 1;
        Ok(iterations_executed)
    }
}
