use super::*;

impl WavefrontScheduler {
    pub(super) fn dispatch_shadow(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_shadow_bind_group(&self.device)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);
        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }

    pub(super) fn dispatch_raygen(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_raygen_bind_group(&self.device)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raygen-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.raygen);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);
        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }

    pub(super) fn dispatch_intersect(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self
            .queue_buffers
            .create_intersect_bind_group(&self.device)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("intersect-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.intersect);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }

    pub(super) fn dispatch_shade(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_shade_bind_group(&self.device)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shade-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.shade);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);
        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }

    pub(super) fn dispatch_scatter(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_scatter_bind_group(&self.device)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.scatter);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);
        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }

    pub(super) fn dispatch_compact(
        &self,
        encoder: &mut CommandEncoder,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let queue_bind_group = self.queue_buffers.create_compact_bind_group(&self.device)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compact-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.compact);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }
}
