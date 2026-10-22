//! Shared-depth specialization of the existing point sprite/OIT renderer.
use super::*;

impl PointRenderer {
    #[cfg(any(test, feature = "weighted-oit"))]
    pub(crate) fn shared_depth_source() -> Result<String, RenderError> {
        let mut source =
            include_str!("../../../shaders/point_instanced.wgsl").replace("\r\n", "\n");
        let replace = |source: &mut String, before: &str, after: &str| -> Result<(), RenderError> {
            if source.matches(before).count() != 1 {
                return Err(RenderError::Render(format!(
                    "shared point shader anchor {before:?} changed"
                )));
            }
            *source = source.replacen(before, after, 1);
            Ok(())
        };
        replace(
            &mut source,
            "    @location(0) position: vec2<f32>,",
            "    @location(0) position: vec2<f32>,\n    @location(5) authored_depth: f32,",
        )?;
        replace(&mut source,"    out.clip_position = uniforms.transform * vec4<f32>(world_pos, 0.0, 1.0);",
            "    out.clip_position = vec4<f32>(point_center + local_offset * 2.0 / uniforms.viewport_size, vertex.authored_depth, 1.0);")?;
        replace(
            &mut source,
            "@fragment\nfn fs_oit(in: VertexOutput) -> OITOutput {",
            "fn concordia_point_output(in: VertexOutput) -> OITOutput {",
        )?;
        replace(
            &mut source,
            "    let effective_alpha = in.color.a * alpha;",
            r#"
    let effective_alpha = in.color.a * alpha;
    if (effective_alpha <= 0.0) { discard; }
    let pixel=vec2<u32>(in.clip_position.xy);
    let opaque=textureLoad(concordia_point_opaque,pixel,0).r;
    if (opaque>0.0) {
        let clip=concordia_frame.proj*vec4<f32>(0.0,0.0,-opaque,1.0);
        let ndc=clip.z/clip.w;
        if (in.clip_position.z>ndc+max(abs(ndc),1.0)*0.00000011920928955078125) { discard; }
    }
"#,
        )?;
        source.push_str(
            &include_str!("../../../shaders/concordia_surface.wgsl")
                .replace("@group(7)", "@group(1)"),
        );
        source.push_str(
            r#"
@group(1) @binding(3) var concordia_point_opaque: texture_2d<f32>;
@fragment fn fs_concordia_oit(in: VertexOutput) -> OITOutput { return concordia_point_output(in); }
@fragment fn fs_concordia_depth(in: VertexOutput) { let shaded=concordia_point_output(in); }
"#,
        );
        Ok(source)
    }

    #[cfg(feature = "weighted-oit")]
    pub(crate) fn configure_shared_depth(
        &mut self,
        device: &wgpu::Device,
        shared: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, RenderError> {
        let source = Self::shared_depth_source()?;
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "concordia.points.shader",
            &source,
        );
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("concordia.points.layout"),
            bind_group_layouts: &[&self.bind_group_layout, shared],
            push_constant_ranges: &[],
        });
        let mut attributes = super::pipelines::INSTANCE_ATTRIBUTES.to_vec();
        attributes.push(wgpu::VertexAttribute {
            offset: 40,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32,
        });
        let vertex = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &attributes,
        }];
        let pipeline =
            |entry: &str, targets: &[Option<wgpu::ColorTargetState>], write_depth: bool| {
                crate::core::shader_registry::create_render_pipeline_scoped(
                    device,
                    &wgpu::RenderPipelineDescriptor {
                        label: Some(entry),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: "vs_main",
                            buffers: &vertex,
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: entry,
                            targets,
                        }),
                        primitive: super::pipelines::point_primitive_state(),
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: wgpu::TextureFormat::Depth32Float,
                            depth_write_enabled: write_depth,
                            depth_compare: if write_depth {
                                wgpu::CompareFunction::LessEqual
                            } else {
                                wgpu::CompareFunction::Always
                            },
                            stencil: Default::default(),
                            bias: Default::default(),
                        }),
                        multisample: Default::default(),
                        multiview: None,
                    },
                )
            };
        self.oit_pipeline = pipeline(
            "fs_concordia_oit",
            &crate::vector::oit::get_mrt_color_targets(),
            false,
        );
        self.oit_shader_label = "concordia.points.shader";
        Ok(pipeline("fs_concordia_depth", &[], true))
    }

    #[cfg(feature = "weighted-oit")]
    pub(crate) fn render_representative_depth<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        pipeline: &'pass wgpu::RenderPipeline,
        count: u32,
    ) {
        if let Some(buffer) = &self.instance_buffer {
            crate::core::shader_registry::record_shader_use("concordia.points.shader");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, buffer.slice(..));
            pass.draw(0..4, 0..count);
        }
    }
}
