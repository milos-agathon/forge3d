use super::*;

impl WavefrontScheduler {
    pub(super) fn create_uniforms_bind_group(
        &self,
        uniforms_buffer: &Buffer,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms-bind-group"),
            layout: &self.pipelines.uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        }))
    }

    pub fn create_scene_bind_group(
        &self,
        spheres_buffer: &Buffer,
        mesh_vertices: &Buffer,
        mesh_indices: &Buffer,
        mesh_bvh: &Buffer,
        area_lights: &Buffer,
        directional_lights: &Buffer,
        object_importance: &Buffer,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wavefront-scene-bind-group"),
            layout: &self.pipelines.scene_bind_group_layout,
            entries: &[
                entry(0, spheres_buffer),
                entry(1, mesh_vertices),
                entry(2, mesh_indices),
                entry(3, mesh_bvh),
                entry(4, area_lights),
                entry(5, directional_lights),
                entry(6, object_importance),
                entry(7, &self.restir_prev),
                entry(8, &self.restir_diag_flags),
                entry(9, &self.restir_debug_aov),
                entry(10, &self.restir_gbuffer),
                entry(11, &self.restir_gbuffer_pos),
                entry(12, &self.restir_settings),
                entry(13, &self.restir_gbuffer_mat),
                entry(14, &self.instances_buffer),
                entry(15, &self.blas_descs),
                entry(16, &self.aov_albedo),
                entry(17, &self.aov_depth),
                entry(18, &self.aov_normal),
                entry(19, &self.medium_params),
                entry(20, &self.hair_segments),
            ],
        }))
    }

    pub fn create_accum_bind_group(&self, accum_buffer: &Buffer) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accum-bind-group"),
            layout: &self.pipelines.accum_bind_group_layout,
            entries: &[entry(0, accum_buffer)],
        })
    }
}

fn entry<'a>(binding: u32, buffer: &'a Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
