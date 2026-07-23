use crate::core::framegraph_impl::{
    PassType, RendererGraphBuilder, RendererGraphPlan, ResourceDesc, ResourceType,
};

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
    let hashes = crate::core::shader_registry::shader_hashes_snapshot();
    bytes.extend_from_slice(
        &serde_json::to_vec(&hashes).expect("shader hash snapshot is infallibly serializable"),
    );
    bytes
}

pub(super) fn build_terrain_render_graph(
    output_width: u32,
    output_height: u32,
    height_width: u32,
    height_height: u32,
    color_format: wgpu::TextureFormat,
    aov: bool,
    declaration_uniforms: Vec<u8>,
) -> crate::core::error::RenderResult<RendererGraphPlan> {
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
        size: Some(declaration_uniforms.len().max(1) as u64),
        usage: None,
        can_alias: false,
        is_transient: true,
    });
    let shadow = builder.add_resource(texture_resource(
        "terrain.shadow.depth",
        ResourceType::DepthStencilAttachment,
        wgpu::TextureFormat::Depth32Float,
        output_width,
        output_height,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        true,
        false,
    ));
    let beauty = builder.add_resource(texture_resource(
        "terrain.forward.beauty",
        ResourceType::ColorAttachment,
        color_format,
        output_width,
        output_height,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        true,
        false,
    ));
    let aov_resource = aov.then(|| {
        builder.add_resource(texture_resource(
            "terrain.forward.aov",
            ResourceType::ColorAttachment,
            wgpu::TextureFormat::Rgba32Float,
            output_width,
            output_height,
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
            | wgpu::TextureUsages::COPY_SRC,
        false,
        false,
    ));
    let cache_reason =
        "native intermediate restore is disabled until every mutable terrain binding is keyed";
    builder.add_pass("terrain.prepare", PassType::Transfer, |pass| {
        pass.read(height)
            .write(prepared)
            .pipeline_descriptor(pipeline_material("terrain.prepare", color_format, aov))
            .uniform_bytes(declaration_uniforms.clone())
            .disable_cache(cache_reason);
        Ok(())
    })?;
    builder.add_pass("terrain.shadow", PassType::Graphics, |pass| {
        pass.read(height)
            .read(prepared)
            .write(shadow)
            .pipeline_descriptor(pipeline_material("terrain.shadow", color_format, aov))
            .uniform_bytes(declaration_uniforms.clone())
            .disable_cache(cache_reason);
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
            .uniform_bytes(declaration_uniforms.clone())
            .disable_cache(cache_reason);
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
            .uniform_bytes(declaration_uniforms)
            .disable_cache(cache_reason);
        if let Some(resource) = aov_resource {
            pass.read(resource);
        }
        Ok(())
    })?;
    builder.compile()
}
