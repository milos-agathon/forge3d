use super::*;
use crate::path_tracing::TracerParams;

impl WavefrontScheduler {
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
        for iteration in 0..max_iterations {
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
            if iteration % 2 == 0 && iteration > 0 {
                self.dispatch_compact(&mut encoder)?;
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.frame_index += 1;
        Ok(())
    }

    pub fn render_frame_simple(
        &mut self,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
        let mut did_any = false;
        for iteration in 0..max_iterations {
            let ray_count =
                self.queue_buffers
                    .get_active_ray_count(&self.device, &self.queue, &mut encoder)?;
            if ray_count == 0 && did_any {
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
            did_any = true;
            if iteration % 2 == 0 && iteration > 0 {
                self.dispatch_compact(&mut encoder)?;
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.frame_index += 1;
        Ok(())
    }
}
