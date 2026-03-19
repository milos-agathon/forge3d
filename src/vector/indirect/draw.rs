use super::IndirectRenderer;

impl IndirectRenderer {
    pub fn draw_indirect<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        draw_count: u32,
    ) {
        if draw_count > 0 {
            render_pass.draw_indirect(&self.draw_commands_buffer, 0);
        }
    }
}
