/// Indirect drawing and GPU culling manager
pub struct IndirectRenderer {
    pub(super) draw_commands_buffer: wgpu::Buffer,
    pub(super) instances_buffer: wgpu::Buffer,
    pub(super) instances_capacity: usize,
    pub(super) culling_pipeline: wgpu::ComputePipeline,
    pub(super) culling_bind_group_layout: wgpu::BindGroupLayoutDescriptor<'static>,
    pub(super) culling_uniforms_buffer: wgpu::Buffer,
    pub(super) counter_buffer: wgpu::Buffer,
    pub(super) readback_buffer: wgpu::Buffer,
    pub(super) cpu_culling_enabled: bool,
}
