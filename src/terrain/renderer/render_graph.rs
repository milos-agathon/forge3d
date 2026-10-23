use crate::core::framegraph_impl::{
    PassType, RendererGraphBuilder, RendererGraphPlan, ResourceDesc, ResourceHandle, ResourceType,
};

pub(super) struct TerrainGraphHandles {
    pub(super) height: ResourceHandle,
    pub(super) prepared: ResourceHandle,
    pub(super) shadow: ResourceHandle,
    pub(super) beauty: ResourceHandle,
    #[allow(dead_code)] // consumed when the additive media attachment enables this graph
    pub(super) media: Option<TerrainMediaGraphHandles>,
    pub(super) resolved: ResourceHandle,
}

#[derive(Clone, Copy)]
pub(in crate::terrain) struct TerrainMediaGraphHandles {
    /// Authoritative camera depth supplied by the terrain depth prepass/adapter.
    pub(in crate::terrain) scene_depth: ResourceHandle,
    /// Directional linear-HDR radiance populated while the forward sky is prepared.
    pub(in crate::terrain) radiance_provider: ResourceHandle,
    pub(in crate::terrain) extinction: ResourceHandle,
    pub(in crate::terrain) light_transmittance: ResourceHandle,
    pub(in crate::terrain) world_medium: ResourceHandle,
    pub(in crate::terrain) world_sun: ResourceHandle,
    pub(in crate::terrain) surface_cloudlight: ResourceHandle,
    pub(in crate::terrain) directional_buffers: Option<[ResourceHandle; 6]>,
    pub(in crate::terrain) directional_textures: Option<[ResourceHandle; 4]>,
    pub(in crate::terrain) in_scatter: ResourceHandle,
    pub(in crate::terrain) integrated: ResourceHandle,
    pub(in crate::terrain) transmittance: ResourceHandle,
    pub(in crate::terrain) cloud_shadow: ResourceHandle,
    pub(in crate::terrain) optical_depth: ResourceHandle,
    pub(in crate::terrain) history_previous: ResourceHandle,
    pub(in crate::terrain) history_depth_previous: ResourceHandle,
    pub(in crate::terrain) history_current: ResourceHandle,
    pub(in crate::terrain) history_depth_current: ResourceHandle,
    pub(in crate::terrain) composite: ResourceHandle,
}

pub(super) struct TerrainRenderGraph {
    pub(super) plan: RendererGraphPlan,
    pub(super) handles: TerrainGraphHandles,
}

pub(super) struct TerrainPassDeclarations {
    pub(super) prepare: Vec<u8>,
    pub(super) shadow: Vec<u8>,
    pub(super) forward: Vec<u8>,
    pub(super) resolve: Vec<u8>,
    pub(super) prepared_output_size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::terrain) struct TerrainMediaGraphConfig {
    pub(in crate::terrain) enabled: bool,
    pub(in crate::terrain) directional: bool,
    pub(in crate::terrain) froxel_grid: super::media::FroxelGrid,
    pub(in crate::terrain) resource_version: u64,
}

impl TerrainMediaGraphConfig {
    pub(in crate::terrain) fn disabled(output_width: u32, output_height: u32) -> Self {
        Self {
            enabled: false,
            directional: false,
            froxel_grid: super::media::FroxelGrid::for_viewport(output_width, output_height),
            resource_version: 0,
        }
    }
}

fn texture_resource(
    name: &str,
    resource_type: ResourceType,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
    is_transient: bool,
    can_alias: bool,
) -> ResourceDesc {
    ResourceDesc {
        name: name.into(),
        resource_type,
        format: Some(format),
        extent: Some(wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        }),
        size: None,
        usage: Some(usage),
        can_alias,
        is_transient,
    }
}

fn pipeline_material(label: &str, color_format: wgpu::TextureFormat, aov: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"forge3d.terrain.pipeline-declaration/v1\0");
    bytes.extend_from_slice(label.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(format!("{color_format:?}").as_bytes());
    bytes.push(u8::from(aov));
    bytes
}

pub(super) fn build_terrain_render_graph(
    output_width: u32,
    output_height: u32,
    internal_width: u32,
    internal_height: u32,
    height_width: u32,
    height_height: u32,
    shadow_resolution: u32,
    shadow_layers: u32,
    color_format: wgpu::TextureFormat,
    aov: bool,
    media: TerrainMediaGraphConfig,
    declarations: TerrainPassDeclarations,
    cacheable: bool,
) -> crate::core::error::RenderResult<TerrainRenderGraph> {
    let mut builder = RendererGraphBuilder::new();
    let height = builder.add_resource(texture_resource(
        "terrain.height.input",
        ResourceType::SampledTexture,
        wgpu::TextureFormat::R32Float,
        height_width,
        height_height,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        false,
        false,
    ));
    let prepared = builder.add_resource(ResourceDesc {
        name: "terrain.prepared.uniforms".into(),
        resource_type: ResourceType::UniformBuffer,
        format: None,
        extent: None,
        size: Some(declarations.prepared_output_size.max(1)),
        usage: None,
        can_alias: false,
        is_transient: true,
    });
    let mut shadow_desc = texture_resource(
        "terrain.shadow.depth",
        ResourceType::DepthStencilAttachment,
        wgpu::TextureFormat::Depth32Float,
        shadow_resolution,
        shadow_resolution,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        true,
        false,
    );
    if let Some(extent) = shadow_desc.extent.as_mut() {
        extent.depth_or_array_layers = shadow_layers.max(1);
    }
    let shadow = builder.add_resource(shadow_desc);
    let beauty = builder.add_resource(texture_resource(
        "terrain.forward.beauty",
        ResourceType::ColorAttachment,
        color_format,
        internal_width,
        internal_height,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        true,
        false,
    ));
    let media_handles = media.enabled.then(|| {
        let scene_depth = builder.add_resource(texture_resource(
            "terrain.depth.authoritative",
            ResourceType::SampledTexture,
            wgpu::TextureFormat::Depth32Float,
            internal_width,
            internal_height,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            false,
            false,
        ));
        let mut ext = texture_resource(
            "nephele.media.froxel.extinction",
            ResourceType::SampledTexture,
            wgpu::TextureFormat::Rgba16Float,
            media.froxel_grid.width,
            media.froxel_grid.height,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            false,
            false,
        );
        ext.extent = Some(wgpu::Extent3d {
            width: media.froxel_grid.width,
            height: media.froxel_grid.height,
            depth_or_array_layers: media.froxel_grid.depth,
        });
        let extinction = builder.add_resource(ext);
        let radiance_provider = builder.add_resource(texture_resource(
            "nephele.media.radiance_provider",
            ResourceType::SampledTexture,
            wgpu::TextureFormat::Rgba16Float,
            internal_width,
            internal_height,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            false,
            false,
        ));
        let mut light = texture_resource(
            "nephele.media.froxel.light_transmittance",
            ResourceType::SampledTexture,
            wgpu::TextureFormat::Rgba16Float,
            media.froxel_grid.width,
            media.froxel_grid.height,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            false,
            false,
        );
        light.extent = Some(wgpu::Extent3d {
            width: media.froxel_grid.width,
            height: media.froxel_grid.height,
            depth_or_array_layers: media.froxel_grid.depth,
        });
        let light_transmittance = builder.add_resource(light);
        let world_resource = |name, resource_type| {
            let mut resource = texture_resource(
                name, resource_type, wgpu::TextureFormat::Rgba16Float,
                media.froxel_grid.depth, media.froxel_grid.depth,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                false, false,
            );
            resource.extent = Some(wgpu::Extent3d {
                width: media.froxel_grid.depth, height: media.froxel_grid.depth,
                depth_or_array_layers: media.froxel_grid.depth,
            });
            resource
        };
        let world_medium = builder.add_resource(world_resource("nephele.media.world.medium", ResourceType::SampledTexture));
        let world_sun = builder.add_resource(world_resource("nephele.media.world.sun", ResourceType::SampledTexture));
        let surface_cloudlight = builder.add_resource(texture_resource(
            "nephele.media.surface.cloudlight", ResourceType::SampledTexture,
            wgpu::TextureFormat::Rgba16Float, internal_width, internal_height,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            false, false,
        ));
        let mut scatter = texture_resource(
            "nephele.media.froxel.in_scatter",
            ResourceType::SampledTexture,
            wgpu::TextureFormat::Rgba16Float,
            media.froxel_grid.width,
            media.froxel_grid.height,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            false,
            false,
        );
        scatter.extent = Some(wgpu::Extent3d {
            width: media.froxel_grid.width,
            height: media.froxel_grid.height,
            depth_or_array_layers: media.froxel_grid.depth,
        });
        let in_scatter = builder.add_resource(scatter);
        let history_depth_previous = builder.add_resource(texture_resource(
            "nephele.media.history.previous.depth",
            ResourceType::SampledTexture,
            wgpu::TextureFormat::Depth32Float,
            internal_width,
            internal_height,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            false,
            false,
        ));
        let history_depth_current = builder.add_resource(texture_resource(
            "nephele.media.history.current.depth",
            ResourceType::SampledTexture,
            wgpu::TextureFormat::Depth32Float,
            internal_width,
            internal_height,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            false,
            false,
        ));
        let mut output = |name: &str| {
            builder.add_resource(texture_resource(
                name,
                ResourceType::SampledTexture,
                wgpu::TextureFormat::Rgba16Float,
                internal_width,
                internal_height,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                false,
                false,
            ))
        };
        let integrated = output("nephele.media.aov.in_scatter");
        let transmittance = output("nephele.media.aov.transmittance");
        let cloud_shadow = output("nephele.media.aov.cloud_shadow");
        let optical_depth = output("nephele.media.aov.optical_depth");
        let history_previous = output("nephele.media.history.previous.in_scatter");
        let history_current = output("nephele.media.history.current.in_scatter");
        let composite = output("nephele.media.composite.linear_hdr");
        let directional_buffers = media.directional.then(|| std::array::from_fn(|index| {
            builder.add_resource(ResourceDesc {
                name: format!("nephele.media.directional.buffer.{index}"),
                resource_type: ResourceType::StorageBuffer, format: None, extent: None,
                size: None, usage: None, can_alias: false, is_transient: false,
            })
        }));
        let directional_textures=media.directional.then(||std::array::from_fn(|index| {
            builder.add_resource(ResourceDesc {
                name:format!("nephele.media.directional.texture.{index}"),resource_type:ResourceType::SampledTexture,
                format:Some(match index { 2=>wgpu::TextureFormat::Rgba32Uint,3=>wgpu::TextureFormat::R32Uint,_=>wgpu::TextureFormat::Rgba16Float }),
                extent:None,size:None,usage:Some(wgpu::TextureUsages::TEXTURE_BINDING|if index==3 {wgpu::TextureUsages::COPY_DST}else{wgpu::TextureUsages::STORAGE_BINDING}),can_alias:false,is_transient:false,
            })
        }));
        TerrainMediaGraphHandles {
            scene_depth,
            radiance_provider,
            extinction,
            light_transmittance,
            world_medium,
            world_sun,
            surface_cloudlight,
            directional_buffers,
            directional_textures,
            in_scatter,
            integrated,
            transmittance,
            cloud_shadow,
            optical_depth,
            history_previous,
            history_depth_previous,
            history_current,
            history_depth_current,
            composite,
        }
    });
    let aov_resource = aov.then(|| {
        builder.add_resource(texture_resource(
            "terrain.forward.aov",
            ResourceType::ColorAttachment,
            wgpu::TextureFormat::Rgba32Float,
            internal_width,
            internal_height,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            true,
            false,
        ))
    });
    let resolved = builder.add_resource(texture_resource(
        "terrain.resolve.output",
        ResourceType::ColorAttachment,
        color_format,
        output_width,
        output_height,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        false,
        false,
    ));
    builder.add_pass("terrain.prepare", PassType::Transfer, |pass| {
        pass.read(height)
            .write(prepared)
            .pipeline_descriptor(pipeline_material("terrain.prepare", color_format, aov))
            .uniform_bytes(declarations.prepare.clone());
        if !cacheable {
            pass.disable_cache("native terrain cache declaration is unavailable for this render");
        }
        Ok(())
    })?;
    builder.add_pass("terrain.shadow", PassType::Graphics, |pass| {
        pass.read(height)
            .write(shadow)
            .pipeline_descriptor(pipeline_material("terrain.shadow", color_format, aov))
            .uniform_bytes(declarations.shadow);
        // Depth32Float cannot be restored by Queue::write_texture. Recompute
        // this producer explicitly; dependent pass keys still describe its inputs.
        pass.disable_cache(
            "Depth32Float shadow resources are recomputed; queue depth uploads are unsupported",
        );
        Ok(())
    })?;
    if let Some(m) = media_handles {
        builder.add_pass("nephele.media.cloudlight", PassType::Compute, |pass| {
            pass.read(height).read(prepared).read(m.world_medium)
                .write(m.world_sun).write(m.surface_cloudlight)
                .pipeline_descriptor(pipeline_material("nephele.media.cloudlight", color_format, aov))
                .uniform_bytes(declarations.forward.clone());
            if let Some(buffers) = m.directional_buffers {
                pass.read(m.extinction).read(m.radiance_provider).read(m.light_transmittance)
                    .write(m.radiance_provider).write(m.in_scatter);
                for (index,buffer) in buffers.into_iter().enumerate() {
                    pass.read(buffer);
                    if index!=2 { pass.write(buffer); }
                }
                for (index,texture) in m.directional_textures.unwrap().into_iter().enumerate() { pass.read(texture); if index!=3 {pass.write(texture);} }
            }
            Ok(())
        })?;
    }
    let forward_label = if aov {
        "terrain.forward_aov"
    } else {
        "terrain.forward"
    };
    builder.add_pass(forward_label, PassType::Graphics, |pass| {
        pass.read(height)
            .read(prepared)
            .read(shadow)
            .write(beauty)
            .pipeline_descriptor(pipeline_material(forward_label, color_format, aov))
            .uniform_bytes(declarations.forward);
        if !cacheable {
            pass.disable_cache("native terrain cache declaration is unavailable for this render");
        }
        if let Some(resource) = aov_resource {
            pass.write(resource);
        }
        if let Some(m) = media_handles {
            // Terrain direct light samples the complete canonical surface-to-sun
            // segment before view-ray integration. Directional cloudlight already
            // populated the radiance provider; only the legacy path writes it here.
            pass.read(m.light_transmittance)
                .read(m.surface_cloudlight)
                .write(m.scene_depth);
            if m.directional_buffers.is_some() { pass.read(m.radiance_provider); }
            else { pass.write(m.radiance_provider); }
        }
        Ok(())
    })?;
    if let Some(m) = media_handles {
        // Legacy injection follows forward's radiance-provider write. In the
        // directional path both passes read cloudlight's provider, so injection
        // may precede forward. Integration consumes both results and scene depth.
        builder.add_pass("nephele.media.inject", PassType::Compute, |pass| {
            pass.read(prepared)
                .read(shadow)
                .read(m.radiance_provider)
                .read(m.extinction)
                .read(m.light_transmittance)
                .read(m.world_medium)
                .read(m.world_sun)
                .pipeline_descriptor(pipeline_material("nephele.media.inject", color_format, aov))
                .uniform_bytes(media.resource_version.to_le_bytes().to_vec());
            if let Some(buffers) = m.directional_buffers {
                for buffer in buffers { pass.read(buffer); }
                for texture in m.directional_textures.unwrap() {pass.read(texture);}
                pass.read(m.in_scatter);
            } else {pass.write(m.in_scatter);}
            Ok(())
        })?;
        builder.add_pass("nephele.media.integrate", PassType::Compute, |pass| {
            pass.read(m.extinction)
                .read(m.light_transmittance)
                .read(m.in_scatter)
                .read(m.scene_depth)
                .read(m.history_previous)
                .read(m.history_depth_previous)
                .write(m.integrated)
                .write(m.transmittance)
                .write(m.cloud_shadow)
                .write(m.optical_depth)
                .pipeline_descriptor(pipeline_material(
                    "nephele.media.integrate",
                    color_format,
                    aov,
                ))
                .uniform_bytes(media.resource_version.to_le_bytes().to_vec());
            Ok(())
        })?;
        builder.add_pass("nephele.media.composite", PassType::Compute, |pass| {
            pass.read(beauty)
                .read(m.integrated)
                .read(m.transmittance)
                .write(m.composite)
                .pipeline_descriptor(pipeline_material(
                    "nephele.media.composite.linear_hdr",
                    color_format,
                    aov,
                ))
                .uniform_bytes(media.resource_version.to_le_bytes().to_vec());
            Ok(())
        })?;
        builder.add_pass("nephele.media.history.commit", PassType::Transfer, |pass| {
            pass.read(m.integrated)
                .read(m.scene_depth)
                .write(m.history_current)
                .write(m.history_depth_current)
                .pipeline_descriptor(pipeline_material(
                    "nephele.media.history.commit",
                    color_format,
                    aov,
                ))
                .uniform_bytes(media.resource_version.to_le_bytes().to_vec());
            Ok(())
        })?;
    }
    let resolve_label = if aov {
        "terrain.resolve_aov"
    } else {
        "terrain.resolve"
    };
    builder.add_pass(resolve_label, PassType::Transfer, |pass| {
        pass.read(media_handles.map_or(beauty, |m| m.composite))
            .write(resolved)
            .pipeline_descriptor(pipeline_material(resolve_label, color_format, aov))
            .uniform_bytes(declarations.resolve);
        if !cacheable {
            pass.disable_cache("native terrain cache declaration is unavailable for this render");
        }
        if let Some(resource) = aov_resource {
            pass.read(resource);
        }
        Ok(())
    })?;
    Ok(TerrainRenderGraph {
        plan: builder.compile()?,
        handles: TerrainGraphHandles {
            height,
            prepared,
            shadow,
            beauty,
            media: media_handles,
            resolved,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn declarations() -> TerrainPassDeclarations {
        TerrainPassDeclarations {
            prepare: vec![],
            shadow: vec![],
            forward: vec![],
            resolve: vec![],
            prepared_output_size: 64,
        }
    }
    #[test]
    fn directional_media_declares_each_transport_buffer_and_source_read() {
        let graph = build_terrain_render_graph(
            64, 64, 64, 64, 16, 16, 256, 4, wgpu::TextureFormat::Rgba16Float, false,
            TerrainMediaGraphConfig { enabled: true, directional: true,
                froxel_grid: super::super::media::FroxelGrid::for_viewport(64,64), resource_version: 7 },
            declarations(), false,
        ).unwrap();
        let media = graph.handles.media.unwrap();
        let pass = graph.plan.pass("nephele.media.cloudlight").unwrap();
        assert!(pass.reads.contains(&media.extinction));
        assert!(pass.reads.contains(&media.radiance_provider));
        for (index,buffer) in media.directional_buffers.unwrap().into_iter().enumerate() {
            assert!(pass.reads.contains(&buffer));
            assert_eq!(pass.writes.contains(&buffer),index!=2);
            assert_eq!(graph.plan.resource(buffer).unwrap().desc.resource_type, ResourceType::StorageBuffer);
        }
        for (index,texture) in media.directional_textures.unwrap().into_iter().enumerate() {
            assert!(pass.reads.contains(&texture));
            assert_eq!(pass.writes.contains(&texture),index!=3);
        }
    }
    #[test]
    fn directional_aov_execution_follows_the_compiled_plan() {
        let graph = build_terrain_render_graph(
            64, 64, 64, 64, 16, 16, 256, 4, wgpu::TextureFormat::Rgba16Float, true,
            TerrainMediaGraphConfig { enabled: true, directional: true,
                froxel_grid: super::super::media::FroxelGrid::for_viewport(64,64), resource_version: 7 },
            declarations(), false,
        ).unwrap();
        let shadow = graph.handles.shadow;
        let mut plan = graph.plan;
        let main = super::super::aov::ordered_aov_main_passes(&plan);
        let position = |name| main.iter().position(|label| label == name).unwrap();
        assert!(position("nephele.media.inject") < position("terrain.forward_aov"));
        assert!(super::super::aov::observe_aov_shadow_transition(
            "terrain.forward_aov", &[], shadow, &mut false,
        ).is_err());
        let mut executed = Vec::new();
        let mut shadow_transition_observed = false;
        for label in ["terrain.prepare", "nephele.media.cloudlight", "terrain.shadow"]
            .into_iter().chain(main.iter().map(String::as_str)).chain(["terrain.resolve_aov"])
        {
            plan.execute_with_barriers(label, |barriers| {
                if matches!(label, "nephele.media.inject" | "terrain.forward_aov") {
                    super::super::aov::observe_aov_shadow_transition(
                        label, barriers, shadow, &mut shadow_transition_observed,
                    ).unwrap();
                }
                executed.push(label.to_string());
                Ok::<(), crate::core::error::RenderError>(())
            }).unwrap();
        }
        assert!(shadow_transition_observed);
        assert_eq!(executed, plan.labels);
        plan.finish().unwrap();
    }
    #[test]
    fn media_disabled_preserves_graph() {
        let g = build_terrain_render_graph(
            640,
            480,
            640,
            480,
            32,
            32,
            256,
            4,
            wgpu::TextureFormat::Rgba16Float,
            false,
            TerrainMediaGraphConfig::disabled(640, 480),
            declarations(),
            false,
        )
        .unwrap();
        assert_eq!(
            g.plan.labels,
            [
                "terrain.prepare",
                "terrain.shadow",
                "terrain.forward",
                "terrain.resolve"
            ]
        );
        assert!(g.handles.media.is_none());
    }
    #[test]
    fn media_order_includes_temporal_commit_before_resolve() {
        let g = build_terrain_render_graph(
            640,
            480,
            640,
            480,
            32,
            32,
            256,
            4,
            wgpu::TextureFormat::Rgba16Float,
            false,
            TerrainMediaGraphConfig {
                enabled: true,
                directional: false,
                froxel_grid: super::super::media::FroxelGrid::for_viewport(640, 480),
                resource_version: 7,
            },
            declarations(),
            false,
        )
        .unwrap();
        let position = |label| {
            g.plan
                .labels
                .iter()
                .position(|candidate| candidate == label)
                .unwrap()
        };
        assert!(position("terrain.prepare") < position("nephele.media.cloudlight"));
        assert!(position("nephele.media.cloudlight") < position("terrain.forward"));
        assert!(position("terrain.forward") < position("nephele.media.inject"));
        assert!(position("nephele.media.inject") < position("nephele.media.integrate"));
        assert!(position("nephele.media.integrate") < position("nephele.media.composite"));
        assert!(position("nephele.media.composite") < position("nephele.media.history.commit"));
        assert!(position("nephele.media.history.commit") < position("terrain.resolve"));
    }

    #[test]
    fn media_aov_injection_reuses_forward_shadow_transition_and_order_stays_strict() {
        let g = build_terrain_render_graph(
            64,
            64,
            64,
            64,
            32,
            32,
            256,
            4,
            wgpu::TextureFormat::Rgba16Float,
            true,
            TerrainMediaGraphConfig {
                enabled: true,
                directional: false,
                froxel_grid: super::super::media::FroxelGrid::for_viewport(64, 64),
                resource_version: 7,
            },
            declarations(),
            false,
        )
        .unwrap();
        let media = g.handles.media.unwrap();
        let position = |label| {
            g.plan
                .labels
                .iter()
                .position(|candidate| candidate == label)
                .unwrap()
        };
        assert!(position("terrain.prepare") < position("nephele.media.cloudlight"));
        assert!(position("nephele.media.cloudlight") < position("terrain.forward_aov"));
        assert!(position("terrain.forward_aov") < position("nephele.media.inject"));
        assert!(position("nephele.media.inject") < position("nephele.media.integrate"));
        assert!(position("nephele.media.integrate") < position("nephele.media.composite"));
        assert!(position("nephele.media.composite") < position("nephele.media.history.commit"));
        assert!(position("nephele.media.history.commit") < position("terrain.resolve_aov"));
        assert!(g.plan.barriers_before("nephele.media.inject").is_empty());
        let forward = g.plan.pass("terrain.forward_aov").unwrap();
        let inject = g.plan.pass("nephele.media.inject").unwrap();
        assert!(inject.reads.contains(&media.world_medium));
        assert!(inject.reads.contains(&media.world_sun));
        assert!(g
            .plan
            .pass("terrain.forward_aov")
            .unwrap()
            .reads
            .contains(&g.handles.shadow));
        assert!(forward.writes.contains(&media.radiance_provider));
        assert!(inject.reads.contains(&media.radiance_provider));
        assert!(g
            .plan
            .pass("nephele.media.integrate")
            .unwrap()
            .reads
            .contains(&media.scene_depth));
        assert!(g
            .plan
            .barriers_before("terrain.forward_aov")
            .iter()
            .any(|barrier| barrier.resource == g.handles.shadow));
        assert!(g.plan.barriers_before("nephele.media.integrate").is_empty());
        assert!(g
            .plan
            .barriers_before("nephele.media.composite")
            .iter()
            .any(|barrier| barrier.resource == g.handles.beauty));
        assert!(g
            .plan
            .barriers_before("nephele.media.history.commit")
            .is_empty());
        assert!(g
            .plan
            .barriers_before("terrain.resolve_aov")
            .iter()
            .any(|barrier| {
                g.plan
                    .resource(barrier.resource)
                    .is_some_and(|resource| resource.desc.name == "terrain.forward.aov")
            }));

        let mut plan = g.plan;
        plan.execute_with_barriers("terrain.prepare", |_| {
            Ok::<(), crate::core::error::RenderError>(())
        })
        .unwrap();
        plan.execute_with_barriers("nephele.media.cloudlight", |_| {
            Ok::<(), crate::core::error::RenderError>(())
        })
        .unwrap();
        plan.execute_with_barriers("terrain.shadow", |_| {
            Ok::<(), crate::core::error::RenderError>(())
        })
        .unwrap();
        let error = plan
            .execute_with_barriers("nephele.media.inject", |_| {
                Ok::<(), crate::core::error::RenderError>(())
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("expected \"terrain.forward_aov\""));
    }
}
