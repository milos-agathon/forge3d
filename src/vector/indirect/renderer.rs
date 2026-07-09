/// Indirect drawing and GPU culling manager
pub struct IndirectRenderer {
    pub(super) draw_commands_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub(super) instances_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub(super) instances_capacity: usize,
    pub(super) culling_pipeline: wgpu::ComputePipeline,
    pub(super) culling_bind_group_layout: wgpu::BindGroupLayoutDescriptor<'static>,
    pub(super) culling_uniforms_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub(super) counter_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub(super) readback_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub(super) cpu_culling_enabled: bool,
}
