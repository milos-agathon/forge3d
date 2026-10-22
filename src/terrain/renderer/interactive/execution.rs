use super::*;

impl TerrainScene {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_interactive_internal(
        &mut self,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<f32>,
        water_mask: Option<PyReadonlyArray2<f32>>,
        time_seconds: f32,
        clouds: Option<&Bound<'_, PyAny>>,
        label_batches: Option<&Bound<'_, PyAny>>,
        vector_batches: Option<&Bound<'_, PyAny>>,
        furniture_batches: Option<&Bound<'_, PyAny>>,
        point_batches: Option<&Bound<'_, PyAny>>,
    ) -> Result<crate::Frame> {
        // Reuse the real terrain grid under an orthographic camera. Hardware
        // raster depth then occludes buildings without a fragment-depth write.
        let screen_params;
        let params = if params.camera_mode == "screen" {
            screen_params = {
                let mut geometry = params.clone();
                geometry.camera_mode = SCREEN_GEOMETRY_MODE.to_owned();
                geometry
            };
            &screen_params
        } else {
            params
        };
        if params.render_scale != 1.0 {
            return Err(anyhow!(
                "CONCORDIA equal-resolution TAA requires render_scale=1; TAAU is unavailable"
            ));
        }
        if params.msaa_samples != 1 {
            return Err(anyhow!("CONCORDIA shared G-buffer requires msaa_samples=1; multisample resolve is unavailable"));
        }
        if params.shading != "forward" {
            return Err(anyhow!("CONCORDIA requires forward shading: the visibility-buffer resolve has no shared GBuffer outputs"));
        }
        #[cfg(feature = "enable-globe")]
        if self.orbis_capture_request.is_some() {
            return Err(anyhow!("CONCORDIA shared MRT outputs cannot include the queued ORBIS coverage attachment; use the dedicated capture entry"));
        }
        let image_transfers = crate::core::texture_readback::ImageTransferScope::begin();
        let (capture, _allocation_scope) =
            self.begin_certificate_capture("terrain.render_interactive");
        let mut timing = self.take_render_timing();
        let mut stages = ExecutedStages::default();
        let decoded = params.decoded();
        if decoded.height_ao.enabled
            && params.height_curve_strength > 0.0
            && matches!(
                params.height_curve_mode.as_str(),
                "pow" | "smoothstep" | "lut"
            )
        {
            return Err(anyhow!("height AO has no curved-height binding; nonlinear height curves cannot join the shared raw-DEM horizon pass"));
        }
        self.ensure_shadow_atlas(&decoded.shadow)?;
        self.prepare_frame_lighting(decoded)?;
        use sha2::Digest;
        let mut radiance = sha2::Sha256::new();
        // Hash authored inputs before upload; no GPU readback is involved.
        radiance.update([u8::from(water_mask.is_some())]);
        if let Some(mask) = water_mask.as_ref() {
            for size in mask.as_array().shape() {
                radiance.update(size.to_le_bytes());
            }
            for value in mask.as_array() {
                radiance.update(value.to_bits().to_le_bytes());
            }
        }
        let mut height_inputs = self.upload_height_inputs(heightmap, water_mask, params)?;
        self.prepare_geometry(
            params,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            height_inputs.terrain_data_hash,
        )?;
        let world_span = params.terrain_span.max(f32::MIN_POSITIVE);
        super::super::probes::prepare_probes(
            self,
            &decoded.probes,
            world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            params.z_scale,
            height_inputs.terrain_data_hash,
        )?;
        super::super::probes::prepare_reflection_probes(
            self,
            &decoded.reflection_probes,
            material_set,
            env_maps,
            params,
            decoded,
            world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            params.z_scale,
            height_inputs.terrain_data_hash,
        )?;
        let materials =
            self.prepare_material_context_with_mode(material_set, params, decoded, true)?;
        let ibl = self.prepare_ibl_bind_group(env_maps)?;
        let curve = if params.height_curve_mode == "lut" {
            params
                .height_curve_lut
                .as_ref()
                .map(|lut| self.upload_height_curve_lut(lut.as_ref().as_slice()))
                .transpose()?
        } else {
            None
        };
        let curve_view = curve
            .as_ref()
            .map(|(texture, _)| texture.create_view(&Default::default()))
            .unwrap_or_else(|| {
                self._height_curve_identity_texture
                    .create_view(&Default::default())
            });
        self.ensure_pipeline_sample_count(1, is_clipmap_camera_mode(&params.camera_mode))?;
        let mut targets = self.create_render_targets_for_format(
            params,
            params.msaa_samples,
            1,
            wgpu::TextureFormat::Rgba16Float,
        )?;
        let mut state = match self.concordia.take() {
            Some(state) if state.matches(params) => state,
            _ => ConcordiaFrame::new(self, params)?,
        };
        env_maps.configure_screen_space_environment(&self.device, &mut state.effects)?;
        let cloud_enabled =
            state.configure_clouds(&self.device, &self.queue, clouds, time_seconds)?;
        radiance.update(format!("{:?}|{:?}", decoded.fog, decoded.volumetrics));
        let sky = &decoded.sky;
        radiance.update([u8::from(sky.enabled), u8::from(sky.aerial_perspective)]);
        radiance.update(sky.model.to_le_bytes());
        radiance.update(bytemuck::cast_slice(&[
            sky.turbidity,
            sky.ground_albedo,
            sky.ozone_du,
            sky.mie_g,
            sky.sun_intensity,
            sky.sun_size,
            sky.aerial_density,
            sky.sky_exposure,
        ]));
        if let Some(lut) = &sky.lut_handle {
            radiance.update(lut.deterministic_sha256());
        }
        radiance.update(bytemuck::bytes_of(&materials.overlay_binding.uniform));
        if let Some(lut) = &materials.overlay_binding.lut {
            radiance.update(lut.texture.width().to_le_bytes());
            radiance.update(lut.texture.height().to_le_bytes());
            radiance.update(lut.content_hash);
        }
        radiance.update(bytemuck::cast_slice(&self.build_shading_uniforms(
            material_set,
            &materials.gpu_materials,
            params,
            decoded,
        )?));
        radiance.update(bytemuck::cast_slice(&material_set.materials));
        radiance.update(bytemuck::cast_slice(&decoded.light.direction));
        radiance.update((Arc::as_ptr(&materials.gpu_materials) as usize).to_le_bytes());
        radiance.update((Arc::as_ptr(&env_maps.gpu_state) as usize).to_le_bytes());
        radiance.update(env_maps.intensity.to_le_bytes());
        radiance.update(env_maps.rotation_deg.to_le_bytes());
        radiance.update(bytemuck::bytes_of(&state.cloud.uniforms));
        radiance.update([u8::from(cloud_enabled)]);
        let history_reset = state.prepare(
            self,
            params,
            cloud_enabled,
            height_inputs.terrain_data_hash,
            radiance.finalize().into(),
        )?;
        let mut uniforms = Self::build_uniforms_with_matrices(
            params,
            decoded,
            glam::Mat4::from_cols_array_2d(&state.current_uniforms.view),
            glam::Mat4::from_cols_array_2d(&state.current_uniforms.proj),
        );
        if let Some(blend) = self.height_detail_blend_override {
            uniforms[39] = -(blend.clamp(0.0, 1.0) + 1.0);
        }
        let uniform_buffer = Arc::new(tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("concordia.terrain.uniform"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?);
        let cartography_requested = vector_batches
            .map(|value| value.len())
            .transpose()?
            .unwrap_or(0)
            > 0
            || point_batches
                .map(|value| value.len())
                .transpose()?
                .unwrap_or(0)
                > 0;
        if cartography_requested {
            if state.cartography_depth.is_none() {
                state.cartography_depth = Some(super::cartographic_depth::CartographyDepth::new(
                    self, params, &state,
                )?);
            }
        } else {
            state.cartography_depth = None;
        }
        targets.concordia = Some(state);
        let labels =
            super::labels::prepare_labels(self, label_batches, params.size_px.0, params.size_px.1)?;
        let vectors = super::overlays::prepare(
            self,
            vector_batches,
            params,
            targets
                .concordia
                .as_ref()
                .expect("frame owner")
                .cartographic_depth_view(),
            targets
                .concordia
                .as_ref()
                .and_then(|frame| frame.cartography_depth.as_ref())
                .map(|depth| &depth.surface_kind_view),
            wgpu::TextureFormat::Rgba8Unorm,
        )?;
        let furniture = super::overlays::prepare(
            self,
            furniture_batches,
            params,
            targets
                .concordia
                .as_ref()
                .expect("frame owner")
                .cartographic_depth_view(),
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        )?;
        let points = super::points::prepare(
            self,
            point_batches,
            targets.concordia.as_ref().expect("frame owner"),
            params,
        )?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("concordia.interactive.frame"),
            });
        let resources_scope = stages.begin(&mut timing, &mut encoder, "concordia.resources");
        let material_vt = self.prepare_material_vt_frame(
            &mut encoder,
            params,
            decoded,
            materials.gpu_materials.layer_count,
            targets.internal_width,
            targets.internal_height,
        )?;
        if cloud_enabled {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.cloud_shadows");
            targets
                .concordia
                .as_ref()
                .expect("frame owner")
                .cloud
                .generate_shadows(&mut encoder);
            stages.end(&mut timing, &mut encoder, scope, 0);
        }
        stages.end(&mut timing, &mut encoder, resources_scope, 0);
        let shadow_scope = stages.begin(&mut timing, &mut encoder, "concordia.shadows");
        let shadow = self.prepare_interactive_shadow_setup(
            &mut encoder,
            params,
            decoded,
            &height_inputs.heightmap_view,
            &curve_view,
            height_inputs.width,
            height_inputs.height,
        )?;
        let sun_visibility = self.compute_sun_visibility_pass(
            &mut encoder,
            &height_inputs.heightmap_view,
            targets.internal_width,
            targets.internal_height,
            height_inputs.width,
            height_inputs.height,
            params,
            decoded,
        )?;
        stages.end(&mut timing, &mut encoder, shadow_scope, 0);
        let geometry_scope = stages.begin(&mut timing, &mut encoder, "concordia.geometry");
        let mut cartography = targets
            .concordia
            .as_mut()
            .expect("frame owner")
            .cartography_depth
            .take();
        if let Some(depth) = cartography.as_mut() {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.cartography_depth");
            let draws = depth.encode(
                self,
                &mut encoder,
                params,
                &height_inputs,
                &materials,
                &ibl,
                &curve_view,
                &mut targets,
                &shadow,
                time_seconds,
                material_vt,
                sun_visibility,
                &mut timing,
            )?;
            stages.end(&mut timing, &mut encoder, scope, draws);
        }
        targets
            .concordia
            .as_mut()
            .expect("frame owner")
            .cartography_depth = cartography;
        super::cartographic_depth::clear_shared_geometry(&mut encoder, &targets);
        let mut staged_selection = None;
        self.encode_forward_pass(
            &mut encoder,
            params,
            decoded,
            &height_inputs,
            &materials,
            &uniform_buffer,
            &ibl,
            &curve_view,
            &targets,
            &shadow,
            material_vt,
            false,
            sun_visibility,
            time_seconds,
            &mut timing,
            &mut staged_selection,
        )?;
        stages.end(&mut timing, &mut encoder, geometry_scope, 1);
        let mut state = targets.concordia.take().expect("frame owner");

        let settings = &decoded.screen_space;
        let mut source = targets.internal_texture.create_view(&Default::default());
        if settings.ssao_enabled || settings.ssgi_enabled || settings.ssr_enabled {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.depth_pyramid");
            let depth_view = state
                .gbuffer()
                .depth_texture
                .create_view(&Default::default());
            state
                .effects
                .build_hzb(&self.device, &mut encoder, &depth_view, false);
            stages.end(&mut timing, &mut encoder, scope, 0);
        }
        if decoded.height_ao.enabled || settings.ssao_enabled {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.ao");
            self.compute_height_ao_pass_with_pipeline(
                &mut encoder,
                &height_inputs.heightmap_view,
                targets.internal_width,
                targets.internal_height,
                height_inputs.width,
                height_inputs.height,
                params,
                decoded,
                Some(&state.height_ao_pipeline),
            )?;
            if settings.ssao_enabled {
                state.effects.execute_effect(
                    &self.device,
                    &mut encoder,
                    ScreenSpaceEffect::SSAO,
                    timing.as_mut(),
                )?;
            }
            let horizon = self
                .height_ao_sample_view
                .lock()
                .map_err(|_| anyhow!("height AO view mutex poisoned"))?;
            let horizon = if decoded.height_ao.enabled {
                horizon.as_ref().unwrap_or(&self.height_ao_fallback_view)
            } else {
                &self.height_ao_fallback_view
            };
            state.compose(self, &mut encoder, &source, horizon, ComposePass::Ao)?;
            source = state.scratch[state.current_scratch].create_view(&Default::default());
            stages.end(&mut timing, &mut encoder, scope, 0);
        }
        if settings.ssgi_enabled {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.ssgi");
            state.effects.set_ssgi_scene_color_view(
                targets.internal_texture.create_view(&Default::default()),
            );
            state.effects.execute_effect(
                &self.device,
                &mut encoder,
                ScreenSpaceEffect::SSGI,
                timing.as_mut(),
            )?;
            let identity = state.white.create_view(&Default::default());
            state.compose(self, &mut encoder, &source, &identity, ComposePass::Gi)?;
            source = state.scratch[state.current_scratch].create_view(&Default::default());
            stages.end(&mut timing, &mut encoder, scope, 0);
        }
        if settings.ssr_enabled {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.ssr");
            state.effects.set_ssr_scene_color_view(
                targets.internal_texture.create_view(&Default::default()),
            );
            state.effects.execute_effect(
                &self.device,
                &mut encoder,
                ScreenSpaceEffect::SSR,
                timing.as_mut(),
            )?;
            let identity = state.white.create_view(&Default::default());
            state.compose(
                self,
                &mut encoder,
                &source,
                &identity,
                ComposePass::Reflections,
            )?;
            source = state.scratch[state.current_scratch].create_view(&Default::default());
            stages.end(&mut timing, &mut encoder, scope, 0);
        }
        if decoded.fog.density > 0.0
            || (decoded.sky.enabled && decoded.sky.aerial_perspective)
            || cloud_enabled
        {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.atmosphere");
            source = state.atmosphere(self, &mut encoder, &source)?;
            stages.end(&mut timing, &mut encoder, scope, 0);
        }

        if let Some(fog) = state.froxels.as_mut() {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.froxels");
            fog.encode(
                self,
                &mut encoder,
                params,
                &state.effects.gbuffer().depth_view,
                &state.frame_bind_group,
                state.frame_index,
                {
                    let view = glam::Mat4::from_cols_array_2d(&state.current_uniforms.view);
                    (
                        view.inverse().w_axis.truncate(),
                        view,
                        glam::Mat4::from_cols_array_2d(&state.current_uniforms.proj),
                    )
                },
            )?;
            let identity = state.white.create_view(&Default::default());
            state.compose(self, &mut encoder, &source, &identity, ComposePass::Fog)?;
            source = state.scratch[state.current_scratch].create_view(&Default::default());
            stages.end(&mut timing, &mut encoder, scope, 0);
        }
        if settings.taa_enabled {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.temporal");
            state.taa.execute(
                &self.device,
                &mut encoder,
                &self.queue,
                &source,
                &state.effects.gbuffer().velocity_view,
                &state.effects.gbuffer().depth_view,
            );
            // execute swaps the history index; history_view is the written frame.
            source = state.taa.history_texture().create_view(&Default::default());
            stages.end(&mut timing, &mut encoder, scope, 0);
        }

        let (width, height) = params.size_px;
        let output = Arc::new(tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.final.color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        )?);
        let output_view = output.create_view(&Default::default());
        let output_scope = stages.begin(&mut timing, &mut encoder, "concordia.output_transform");
        self.dispatch_offline_tonemap_pass(
            &mut encoder,
            &source,
            &output_view,
            width,
            height,
            Self::resolved_offline_tonemap_operator(decoded),
            decoded.tonemap.white_point,
            params.gamma,
        )?;
        stages.end(&mut timing, &mut encoder, output_scope, 0);
        if !points.is_empty() {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.transparent_points");
            super::points::encode(&mut encoder, &output_view, &points);
            stages.end(&mut timing, &mut encoder, scope, points.len() as u32);
        }
        if !vectors.is_empty() {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.vectors");
            super::overlays::draw(&mut encoder, &output_view, &vectors);
            stages.end(&mut timing, &mut encoder, scope, vectors.len() as u32);
        }
        let label_scope = if labels.is_empty() {
            None
        } else {
            stages.begin(&mut timing, &mut encoder, "concordia.labels")
        };
        let output = super::labels::render_labels(self, &mut encoder, output, &labels)?;
        stages.end(&mut timing, &mut encoder, label_scope, labels.len() as u32);
        if !furniture.is_empty() {
            let scope = stages.begin(&mut timing, &mut encoder, "concordia.furniture");
            super::overlays::draw(
                &mut encoder,
                &output.create_view(&Default::default()),
                &furniture,
            );
            stages.end(&mut timing, &mut encoder, scope, furniture.len() as u32);
        }
        self.stage_material_vt_feedback_readback(&mut encoder)?;
        self.queue.submit(Some(encoder.finish()));
        let submitted_provenance = staged_selection.map(|selection| selection.provenance);
        if let Some(ticket) = staged_selection.and_then(|selection| selection.ticket) {
            if !self
                .geometry_provider()?
                .mark_lod_selection_submitted(ticket)
            {
                return Err(anyhow!(
                    "CONCORDIA submission lost its exact LOD readback ticket"
                ));
            }
        }
        // These observation/feedback maps occur only after the final image was
        // encoded; there is no image readback or CPU upload between GPU passes.
        self.refresh_cpu_visibility_oracle_from_gpu_selection(
            params,
            std::mem::take(&mut height_inputs.heightmap_data),
            (height_inputs.width, height_inputs.height),
            submitted_provenance,
        )?;
        self.finish_material_vt_frame()?;
        let gpu_timestamps_available = self.record_render_timings(&mut timing);
        if !gpu_timestamps_available {
            for (label, draws) in &stages.passes {
                crate::core::certificate::record_pass(label, 0.0, *draws);
            }
        }
        self.store_render_timing(timing);
        state.effects.clear_scene_color_overrides();
        state.effects.advance_frame(&self.queue);
        state.geometry_hdr = Some(targets.internal_texture.clone());
        state.point_layers = points;
        let point_layer_count = state.point_layers.len();
        let point_edl_layer_count = super::points::edl_layer_count(&state.point_layers);
        let label_batch_count = labels.len();
        let vector_batch_count = vectors.len();
        // Retain the exact submitted cartographic batches for explicit, after-
        // beauty validation. The routine frame allocates no coverage targets.
        state.label_batches = labels;
        state.vector_batches = vectors;
        let cloud_parameters = serde_json::json!({"coverage":state.cloud.uniforms.cloud_coverage,
            "density":state.cloud.uniforms.cloud_density,"shadow_strength":state.cloud.uniforms.shadow_intensity,
            "offset":state.cloud.uniforms._padding,"time_seconds":state.cloud.uniforms.time});
        let cloud_size = [
            state.cloud.shadow_texture().width(),
            state.cloud.shadow_texture().height(),
        ];
        let height_ao_size = *self
            .height_ao_size
            .lock()
            .map_err(|_| anyhow!("height AO size mutex poisoned"))?;
        let shadow_extent = texture_extent(&self.csm_renderer.shadow_maps);
        let sun_vis_size = *self
            .sun_vis_size
            .lock()
            .map_err(|_| anyhow!("sun visibility size mutex poisoned"))?;
        let auxiliary_sizes = serde_json::json!({
            "heightmap":[height_inputs.width,height_inputs.height],
            "cartography_depth":state.cartography_depth.as_ref().map(|depth|texture_extent(&depth.linear)),
            "height_visibility":if decoded.height_ao.enabled {Some([height_ao_size.0,height_ao_size.1])} else {None},
            "sun_visibility":if sun_visibility {Some([sun_vis_size.0,sun_vis_size.1])} else {None},
            "shadow_depth":if decoded.shadow.enabled {Some(shadow_extent)} else {None},
            "shadow_moments":if decoded.shadow.enabled {self.csm_renderer.evsm_maps.as_ref().map(|t|texture_extent(t))} else {None},
            "cloud_field":if cloud_enabled {Some(cloud_size)} else {None},
            "depth_pyramid_mips":if settings.ssao_enabled || settings.ssgi_enabled || settings.ssr_enabled {
                state.effects.hzb.as_ref().map(|p|(0..p.tex.mip_level_count()).map(|level|
                    [(p.tex.width()>>level).max(1),(p.tex.height()>>level).max(1)]).collect::<Vec<_>>())
            } else {None},
            "gtao_visibility":state.effects.ao_resolved_texture().map(texture_extent),
            "ssr_radiance":state.effects.ssr_output_texture().map(texture_extent),
            "froxels":state.froxels.as_ref().map(|fog|fog.resource_sizes()),
            "temporal_history":if settings.taa_enabled {Some(texture_extent(state.taa.history_texture()))} else {None},
        });
        let executed_ao = state.effects.ssao_settings();
        let ao_parameters = if settings.ssao_enabled {
            Some(serde_json::json!({
            "radius":executed_ao.radius,"intensity":executed_ao.intensity,"bias":executed_ao.bias,
            "num_samples":executed_ao.num_samples,"technique":"gtao"}))
        } else {
            None
        };
        let raster_jitter = [
            state.current_uniforms.geometry[2],
            state.current_uniforms.geometry[3],
        ];
        let cartography_depth_model = state
            .cartography_depth
            .as_ref()
            .map(|_| "unjittered_production_geometry_replay");
        self.concordia = Some(state);
        self.concordia_report = serde_json::json!({
            "backend": "native_concordia", "input_size": [width, height], "output_size": [width, height],
            "requested_geometry_sample_count":params.msaa_samples,"geometry_sample_count":1,
            "image_readbacks_before_output": image_transfers.count(), "label_batches": label_batch_count, "vector_batches": vector_batch_count, "furniture_batches": furniture.len(), "transparent_point_layers": point_layer_count, "point_edl_layers": point_edl_layer_count,
            "effects": { "height_ao": decoded.height_ao.enabled, "gtao": settings.ssao_enabled,
                "ssgi": settings.ssgi_enabled, "ssr": settings.ssr_enabled, "taa": settings.taa_enabled,
                "cloud_shadows": cloud_enabled, "volumetrics": decoded.volumetrics.enabled,
                "atmosphere": decoded.fog.density > 0.0 || (decoded.sky.enabled && decoded.sky.aerial_perspective) },
            "depth_convention": "positive_linear_view_depth_zero_background",
            "cartography_depth_model":cartography_depth_model,
            "velocity_convention":"current_minus_previous_uv_positive_y_down",
            "temporal_mode":if settings.taa_enabled {"equal_resolution_taa"} else {"disabled"},
            "taau_available":false,
            "ao_resolve":"bounded_geometry_partitioned_visibility_product",
            "ao_parameters":ao_parameters,
            "gtao_scope":if decoded.height_ao.enabled && settings.ssao_enabled {"building_contacts_on_terrain"} else {"full_shared_depth"},
            "layer_stages":{"terrain_buildings":"pre_lighting_opaque","point_clouds":"post_resolve_depth_aware_unlit",
                "vectors":"post_resolve_depth_aware","labels_furniture":"output_resolution"},
            "gpu_timestamps_available":gpu_timestamps_available,
            "temporal_history_reset":history_reset,
            "temporal_history_used":settings.taa_enabled && !history_reset,
            "raster_jitter_px":raster_jitter,
            "stage_sizes":stages.passes.iter().map(|(label,_)|(label.to_string(),if *label=="concordia.cloud_shadows" {
                serde_json::json!({"input":null,"output":cloud_size})
            } else if *label=="concordia.resources" {
                serde_json::json!({"input":[height_inputs.width,height_inputs.height],"output":null})
            } else if *label=="concordia.shadows" {
                serde_json::json!({"input":[height_inputs.width,height_inputs.height],
                    "output":if decoded.shadow.enabled {Some([shadow_extent[0],shadow_extent[1]])} else {None}})
            } else if *label=="concordia.geometry" {
                serde_json::json!({"input":[height_inputs.width,height_inputs.height],"output":[width,height]})
            } else { serde_json::json!({"input":[width,height],"output":[width,height]}) })).collect::<serde_json::Map<_,_>>(),
            "auxiliary_sizes":auxiliary_sizes,
            "sun":{"direction":decoded.light.direction,"intensity":decoded.light.intensity},
            "cloud_parameters":cloud_parameters,
            "fallbacks": {"ssgi_miss": "shared_ibl", "ssr_miss": "retained_surface_ibl_probe"},
        }).to_string();
        crate::core::certificate::record_input("concordia.pipeline", self.concordia_report.clone());
        crate::core::certificate::record_model("concordia.ao","bounded height-horizon * GTAO-contact visibility product; combined GTAO excludes terrain-to-terrain samples; buildings use GTAO once");
        crate::core::certificate::record_model("concordia.readback","routine GPU frame has no intermediate image transfers; the returned Frame counts completed final beauty reads");
        self.finish_certificate_capture(capture);
        let readbacks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        self.concordia_readbacks = Some(readbacks.clone());
        Ok(crate::Frame::new(
            self.device.clone(),
            self.queue.clone(),
            output,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .with_readback_counter(readbacks))
    }
}

#[pymethods]
impl TerrainRenderer {
    /// Render the native shared-depth HDR pipeline. Requested unavailable
    /// resources fail before returning a frame; no CPU effect substitutes run.
    /// ANAMNESIS does not yet serialize this graph; a non-None cache is rejected.
    #[pyo3(signature = (material_set, env_maps, params, heightmap, water_mask=None, time_seconds=0.0, clouds=None, label_batches=None, vector_batches=None, furniture_batches=None, point_batches=None, certificate=None, cache=None))]
    fn render_interactive<'py>(
        &mut self,
        py: Python<'py>,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
        time_seconds: f32,
        clouds: Option<Bound<'py, PyAny>>,
        label_batches: Option<Bound<'py, PyAny>>,
        vector_batches: Option<Bound<'py, PyAny>>,
        furniture_batches: Option<Bound<'py, PyAny>>,
        point_batches: Option<Bound<'py, PyAny>>,
        certificate: Option<Bound<'py, PyAny>>,
        cache: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<crate::Frame>> {
        self.last_anamnesis_report = crate::core::anamnesis::CacheReport::default();
        self.last_anamnesis_graph_submissions = 0;
        if cache.as_ref().is_some_and(|value| !value.is_none()) {
            return Err(PyRuntimeError::new_err(
                "native terrain ANAMNESIS does not yet serialize the shared-depth HDR graph; use cache=None",
            ));
        }
        self.scene
            .validate_material_vt_request(params, material_set.materials().len() as u32)
            .map_err(|error| {
                PyRuntimeError::new_err(format!("CONCORDIA request failed: {error:#}"))
            })?;
        if self
            .scene
            .offline_session_active()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        {
            return Err(PyRuntimeError::new_err(
                "End the offline accumulation session before interactive rendering",
            ));
        }
        let frame = self
            .scene
            .render_interactive_internal(
                material_set,
                env_maps,
                params,
                heightmap,
                water_mask,
                time_seconds,
                clouds.as_ref(),
                label_batches.as_ref(),
                vector_batches.as_ref(),
                furniture_batches.as_ref(),
                point_batches.as_ref(),
            )
            .map_err(|error| {
                PyRuntimeError::new_err(format!("CONCORDIA rendering failed: {error:#}"))
            })?;
        crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
        Py::new(py, frame)
    }

    fn interactive_pipeline_info(&self) -> String {
        let mut report: serde_json::Value = serde_json::from_str(&self.scene.concordia_report)
            .unwrap_or_else(|_| serde_json::json!({}));
        if let Some(counter) = self.scene.concordia_readbacks.as_ref() {
            report["beauty_readbacks"] =
                serde_json::json!(counter.load(std::sync::atomic::Ordering::Relaxed));
        }
        report["compiled_features"] =
            serde_json::json!({"weighted_oit":cfg!(feature="weighted-oit")});
        report.to_string()
    }
}

#[derive(Default)]
struct ExecutedStages {
    passes: Vec<(&'static str, u32)>,
}

impl ExecutedStages {
    fn begin(
        &mut self,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
    ) -> Option<(usize, Option<crate::core::gpu_timing::TimingScopeId>)> {
        let index = self.passes.len();
        self.passes.push((label, 0));
        Some((index, ts_begin(timing, encoder, label)))
    }
    fn end(
        &mut self,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
        encoder: &mut wgpu::CommandEncoder,
        token: Option<(usize, Option<crate::core::gpu_timing::TimingScopeId>)>,
        draws: u32,
    ) {
        if let Some((index, scope)) = token {
            self.passes[index].1 = draws;
            ts_end(timing, encoder, scope, draws);
        }
    }
}
