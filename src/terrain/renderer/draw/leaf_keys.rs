use super::*;
use crate::core::anamnesis::PassKey;
use crate::terrain::renderer::{media::MediaGraphInputKeys, render_graph::TerrainGraphHandles};

pub(super) fn terrain_graph_leaf_keys(
    handles: &TerrainGraphHandles,
    source_key: PassKey,
    media: Option<MediaGraphInputKeys>,
) -> crate::core::error::RenderResult<BTreeMap<ResourceHandle, PassKey>> {
    let mut keys = BTreeMap::from([(handles.height, source_key)]);
    if let Some(handles) = handles.media {
        let media = media.ok_or_else(|| crate::core::error::RenderError::render(
            "media graph requires identities for its external GPU inputs"))?;
        keys.extend([
            (handles.extinction, media.extinction),
            (handles.light_transmittance, media.light_transmittance),
            (handles.world_medium, media.world_medium),
            (handles.history_previous, media.history),
            (handles.history_depth_previous, media.history_depth),
        ]);
        if handles.directional_buffers.is_some() {
            keys.insert(handles.radiance_provider, media.radiance_provider);
        }
        if let Some(handles) = handles.directional_buffers {
            let contents = media.directional_buffers.ok_or_else(||
                crate::core::error::RenderError::render("directional graph requires buffer content identities"))?;
            keys.extend(handles.into_iter().zip(contents));
        }
        if let Some(handles) = handles.directional_textures {
            let contents = media.directional_textures.ok_or_else(||
                crate::core::error::RenderError::render("directional graph requires texture content identities"))?;
            keys.extend(handles.into_iter().zip(contents));
        }
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::renderer::render_graph::{
        build_terrain_render_graph, TerrainMediaGraphConfig, TerrainPassDeclarations,
    };
    use crate::terrain::renderer::media::FroxelGrid;

    #[test]
    fn draw_media_leaf_keys_complete_the_scheduler_key_pass() {
        for directional in [false, true] {
            let graph = build_terrain_render_graph(
                64, 64, 64, 64, 16, 16, 256, 4, wgpu::TextureFormat::Rgba16Float, false,
                TerrainMediaGraphConfig {
                    enabled: true, directional, froxel_grid: FroxelGrid::for_viewport(64, 64),
                    resource_version: 7,
                },
                TerrainPassDeclarations {
                    prepare: vec![], shadow: vec![], forward: vec![], resolve: vec![],
                    prepared_output_size: 64,
                }, false,
            ).unwrap();
            let keys = MediaGraphInputKeys {
                extinction: leaf_key(&[0x10, 0x34]),
                light_transmittance: leaf_key(&[0x00, 0x3c]),
                world_medium: leaf_key(&[0x20, 0x34]),
                radiance_provider: leaf_key(&[0x00, 0x38]),
                history: leaf_key(&[0; 8]), history_depth: leaf_key(&[0; 4]),
                directional_buffers: directional.then(|| std::array::from_fn(|i: usize| leaf_key(&i.to_le_bytes()))),
                directional_textures: directional.then(|| std::array::from_fn(|i: usize| leaf_key(&i.to_le_bytes()))),
            };
            let leaves = terrain_graph_leaf_keys(&graph.handles, leaf_key(&[1, 2, 3, 4]), Some(keys)).unwrap();
            let mut executed = Vec::new();
            let mut shadow_transition_observed = false;
            assert!(observe_shadow_transition("terrain.forward", &[], graph.handles.shadow,
                &mut shadow_transition_observed).is_err());
            let result = GraphScheduler::disabled(vec![], vec![]).execute_graph_with(
                &graph.plan, &leaves, |pass, action| {
                    assert!(matches!(action, GraphPassAction::Execute { capture_output: false, .. }));
                    if let GraphPassAction::Execute { barriers, .. } = action {
                        if matches!(pass.name.as_str(), "terrain.forward" | "nephele.media.inject") {
                            observe_shadow_transition(&pass.name, barriers, graph.handles.shadow,
                                &mut shadow_transition_observed).map_err(io_error)?;
                        }
                        if pass.name == "terrain.resolve" {
                            validate_resolve_transition(&graph.plan, &graph.handles, barriers).map_err(io_error)?;
                        }
                    }
                    executed.push(pass.name.clone());
                    Ok(GraphPassOutcome::Executed(vec![]))
                },
            ).unwrap_or_else(|error| panic!("directional={directional}: {error}"));
            assert_eq!(executed, graph.plan.labels);
            assert!(shadow_transition_observed);
            assert!(result.contains_key(&graph.handles.resolved));
            // Every external input participates in the real downstream key;
            // changed media/history must not alias a prior resolved frame.
            for (handle, previous) in &leaves {
                let mut changed = leaves.clone();
                changed.insert(*handle, leaf_key(&previous.0));
                let changed_result = GraphScheduler::disabled(vec![], vec![]).execute_graph_with(
                    &graph.plan, &changed, |_, _| Ok(GraphPassOutcome::Executed(vec![])),
                ).unwrap();
                assert_ne!(result[&graph.handles.resolved], changed_result[&graph.handles.resolved],
                    "directional={directional}, input={:?}", graph.plan.resource(*handle).unwrap().desc.name);
            }
        }
    }

    #[test]
    fn draw_resolve_validation_uses_the_compiled_input_type() {
        for configuration in [None, Some(false), Some(true)] {
            let media = configuration.map_or(TerrainMediaGraphConfig::disabled(64, 64), |directional|
                TerrainMediaGraphConfig { enabled: true, directional,
                    froxel_grid: FroxelGrid::for_viewport(64, 64), resource_version: 7 });
            let graph = build_terrain_render_graph(
                64, 64, 64, 64, 16, 16, 256, 4, wgpu::TextureFormat::Rgba16Float, false,
                media, TerrainPassDeclarations {
                    prepare: vec![], shadow: vec![], forward: vec![], resolve: vec![],
                    prepared_output_size: 64,
                }, false,
            ).unwrap();
            let barriers = graph.plan.barriers_before("terrain.resolve");
            validate_resolve_transition(&graph.plan, &graph.handles, &barriers).unwrap();
            if configuration.is_some() {
                assert_eq!(graph.plan.resource(graph.handles.media.unwrap().composite).unwrap()
                    .desc.resource_type, crate::core::framegraph_impl::ResourceType::SampledTexture);
                validate_resolve_transition(&graph.plan, &graph.handles, &[]).unwrap();
            } else {
                assert!(validate_resolve_transition(&graph.plan, &graph.handles, &[]).is_err());
                let unrelated = graph.plan.barriers_before("terrain.forward");
                assert!(validate_resolve_transition(&graph.plan, &graph.handles, &unrelated).is_err());
            }
        }
    }
}
