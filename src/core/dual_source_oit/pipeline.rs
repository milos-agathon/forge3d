use super::*;

impl DualSourceOITRenderer {
    pub fn create_dual_source_pipeline(
        &mut self,
        device: &wgpu::Device,
        vertex_buffers: &[wgpu::VertexBufferLayout],
        additional_bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> Result<(), String> {
        match self.get_operating_mode() {
            DualSourceOITMode::DualSource => self.create_true_dual_source_pipeline(
                device,
                vertex_buffers,
                additional_bind_group_layouts,
            ),
            DualSourceOITMode::WBOITFallback => Ok(()),
            _ => Ok(()),
        }
    }

    fn create_true_dual_source_pipeline(
        &mut self,
        device: &wgpu::Device,
        vertex_buffers: &[wgpu::VertexBufferLayout],
        additional_bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> Result<(), String> {
        let mut all_layouts = vec![&self.dual_source_bind_group_layout];
        all_layouts.extend(additional_bind_group_layouts);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DualSourceOIT.Pipeline.Layout"),
            bind_group_layouts: &all_layouts,
            push_constant_ranges: &[],
        });

        self.dual_source_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("DualSourceOIT.Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.dual_source_shader,
                    entry_point: "vs_main",
                    buffers: vertex_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.dual_source_shader,
                    entry_point: "fs_main",
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(Self::get_dual_source_color_blend()),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(Self::get_dual_source_alpha_blend()),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));

        Ok(())
    }

    fn get_dual_source_color_blend() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrc1Alpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrc1Alpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    fn get_dual_source_alpha_blend() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }
}
