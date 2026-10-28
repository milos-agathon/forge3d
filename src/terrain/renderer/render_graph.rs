use crate::core::framegraph_impl::{
    PassType, RendererGraphBuilder, RendererGraphPlan, ResourceDesc, ResourceHandle, ResourceType,
};

pub(super) struct TerrainGraphHandles {
    pub(super) height: ResourceHandle,
    pub(super) prepared: ResourceHandle,
    pub(super) shadow: ResourceHandle,
    pub(super) beauty: ResourceHandle,
    pub(super) resolved: ResourceHandle,
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
    // v2 separated format-correct color blobs from RGBA8-only serialization.
    // v3 retires every entry written while shadow depth was restored through
    // a forbidden Depth32Float upload: a forward pass encoded after that
    // failed restore may have read a stale shadow map.
    bytes.extend_from_slice(b"forge3d.terrain.pipeline-declaration/v3\0");
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
        if !cacheable {
            pass.disable_cache("native terrain cache declaration is unavailable for this render");
        } else {
            // Depth32Float cannot be written by Queue::write_texture, so the
            // shadow map is keyed but never stored or served. It is encoded
            // freshly whenever terrain.forward executes and elided when the
            // forward output is restored (its key commits to this one).
            pass.disable_cache(
                "terrain.shadow depth is keyed but never served: Depth32Float has no queue upload path",
            )
            .materialize_on_demand();
        }
        Ok(())
    })?;
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
        Ok(())
    })?;
    let resolve_label = if aov {
        "terrain.resolve_aov"
    } else {
        "terrain.resolve"
    };
    builder.add_pass(resolve_label, PassType::Transfer, |pass| {
        pass.read(beauty)
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
            resolved,
        },
    })
}

// Validate before any GPU work, including actions the normal scheduler is
// forbidden to request. This boundary can be exercised without a GPU texture.
pub(super) fn validate_terrain_pass_action(
    label: &str,
    action: &crate::core::anamnesis::GraphPassAction<'_>,
) -> anyhow::Result<()> {
    use crate::core::anamnesis::GraphPassAction;
    match (label, action) {
        ("terrain.shadow", GraphPassAction::Restore { .. }) => anyhow::bail!(
            "terrain.shadow depth is keyed but never served; the scheduler requested a restore"
        ),
        (
            "terrain.shadow",
            GraphPassAction::Execute {
                capture_output: true,
                ..
            },
        ) => anyhow::bail!(
            "terrain.shadow depth is keyed but never stored; it cannot capture a cache blob"
        ),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod action_tests {
    use super::validate_terrain_pass_action;
    use crate::core::anamnesis::GraphPassAction;

    #[test]
    fn shadow_restore_and_capture_are_rejected_before_gpu_work() {
        let restore = GraphPassAction::Restore {
            blob: &[],
            barriers: &[],
        };
        let error = validate_terrain_pass_action("terrain.shadow", &restore).unwrap_err();
        assert!(error.to_string().contains("scheduler requested a restore"));
        assert!(validate_terrain_pass_action("terrain.forward", &restore).is_ok());
        let capture = GraphPassAction::Execute {
            barriers: &[],
            capture_output: true,
        };
        assert!(validate_terrain_pass_action("terrain.shadow", &capture)
            .unwrap_err()
            .to_string()
            .contains("cannot capture a cache blob"));
        let execute = GraphPassAction::Execute {
            barriers: &[],
            capture_output: false,
        };
        assert!(validate_terrain_pass_action("terrain.shadow", &execute).is_ok());
    }
}
