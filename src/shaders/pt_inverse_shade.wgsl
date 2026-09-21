// src/shaders/pt_inverse_shade.wgsl
// DIFFERENTIA: reverse-mode shading for the shared terrain primal.
//
// Mapped terrain decomposes the direct-sun spectrum into a deterministic
// achromatic baseline plus a three-way importance-sampled spectral residual.
// The reservoir-selected channel is observable in the primal, its proposal
// probability depends on sun elevation and turbidity, and W = 1/q remains
// below the forward weight cap by construction. The pathwise dW term and the
// likelihood score account for different parts of the derivative. The
// pathwise pass differentiates the realized image; the score passes count
// every fresh candidate and spatial redraw using the affected pixel's
// complete nonlinear loss of the finite replicate mean.
//
// Visibility remains detached here. pt_edge_sample.wgsl currently estimates
// its moving cast-shadow boundary from neighboring pixel-center AOVs; that
// approximation does not establish complete boundary support.
// RELEVANT FILES: src/shaders/pt_inverse_loss.wgsl, src/path_tracing/inverse/mod.rs

struct InvSpectralState {
    spectrum: vec3<f32>,
    unit_spectrum: vec3<f32>,
    baseline: f32,
    baseline_unit: f32,
    d_spectrum_tau: vec3<f32>,
    d_baseline_tau: f32,
    d_spectrum_wy: vec3<f32>,
    d_baseline_wy: f32,
    probabilities: vec3<f32>,
    dlogq_tau: vec3<f32>,
    dlogq_wy: vec3<f32>,
}

fn inv_min_channel(v: vec3<f32>) -> u32 {
    if (v.x <= v.y && v.x <= v.z) { return 0u; }
    if (v.y <= v.z) { return 1u; }
    return 2u;
}

fn inv_spectral_state() -> InvSpectralState {
    let omega = normalize(lighting.light_dir);
    let tau = terrain_turbidity();
    let wy = max(omega.y, 0.05);
    let air_mass = 1.0 / wy;
    let transmittance = terrain_sun_transmit(omega.y);
    let dT_dtau = -TERRAIN_ATMOS_BETA * air_mass * transmittance;
    let dT_dwy = select(
        vec3<f32>(0.0),
        TERRAIN_ATMOS_BETA * (tau - 1.0) * transmittance / (wy * wy),
        omega.y > 0.05);

    let unit_spectrum = inv_params.sunv.rgb * transmittance;
    let spectrum = lighting.light_color * transmittance;
    let d_spectrum_tau = lighting.light_color * dT_dtau;
    let d_spectrum_wy = lighting.light_color * dT_dwy;
    let min_channel = inv_min_channel(spectrum);
    let baseline = spectrum[min_channel];
    let baseline_unit = unit_spectrum[min_channel];
    let d_baseline_tau = d_spectrum_tau[min_channel];
    let d_baseline_wy = d_spectrum_wy[min_channel];

    let residual = max(spectrum - vec3<f32>(baseline), vec3<f32>(0.0));
    let residual_active = vec3<f32>(
        select(0.0, 1.0, residual.x > 0.0),
        select(0.0, 1.0, residual.y > 0.0),
        select(0.0, 1.0, residual.z > 0.0));
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let spectral_target = luma * residual;
    let dtarget_tau = luma * (d_spectrum_tau - vec3<f32>(d_baseline_tau)) * residual_active;
    let dtarget_wy = luma * (d_spectrum_wy - vec3<f32>(d_baseline_wy)) * residual_active;
    let target_sum = spectral_target.x + spectral_target.y + spectral_target.z;

    let floor_probability = 1.0 / TERRAIN_RESTIR_W_CAP;
    let focused_mass = 1.0 - 3.0 * floor_probability;
    var probabilities = vec3<f32>(1.0 / 3.0);
    var dlogq_tau = vec3<f32>(0.0);
    var dlogq_wy = vec3<f32>(0.0);
    if (target_sum > 0.0) {
        let normalized = spectral_target / target_sum;
        probabilities = vec3<f32>(floor_probability) + focused_mass * normalized;
        let dsum_tau = dtarget_tau.x + dtarget_tau.y + dtarget_tau.z;
        let dsum_wy = dtarget_wy.x + dtarget_wy.y + dtarget_wy.z;
        let inv_sum2 = 1.0 / (target_sum * target_sum);
        let dnorm_tau = (dtarget_tau * target_sum - spectral_target * dsum_tau) * inv_sum2;
        let dnorm_wy = (dtarget_wy * target_sum - spectral_target * dsum_wy) * inv_sum2;
        dlogq_tau = focused_mass * dnorm_tau / probabilities;
        dlogq_wy = focused_mass * dnorm_wy / probabilities;
    }

    return InvSpectralState(
        spectrum,
        unit_spectrum,
        baseline,
        baseline_unit,
        d_spectrum_tau,
        d_baseline_tau,
        d_spectrum_wy,
        d_baseline_wy,
        probabilities,
        dlogq_tau,
        dlogq_wy);
}

// Spectral temporal and spatial reuse are pixel-local in this specialization,
// so decisions at one pixel cannot affect another pixel's loss. This removes
// cross-pixel zero-mean noise from the whole-image likelihood score. The host
// divides all scalar gradients by R; compensate that division here.
fn inv_add_trace_score(channel: u32, pix: u32) {
    let spectral_state = inv_spectral_state();
    let pixel_loss = bitcast<f32>(atomicLoad(&inv_u32[inv_u32_loss_base() + pix]));
    let reward = pixel_loss * inv_params.fl.z / inv_params.rsv0.x;
    inv_add_scalar(INV_GRAD_SUN_Y, reward * spectral_state.dlogq_wy[channel]);
    inv_add_scalar(INV_GRAD_TURBIDITY, reward * spectral_state.dlogq_tau[channel]);
}

// Dispatch once after each replayed frame chain. The fresh candidate remains
// in curr; binding 7 is rebound to the temporal output for this entry point.
// A spectral spatial self-redraw erases the candidate category before any
// later beauty frame, so that candidate carries no score in that case.
@compute @workgroup_size(256)
fn main_inverse_score_candidate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = gid.y * INV_DISPATCH_X + gid.x;
    if (pix >= uniforms.width * uniforms.height) { return; }
    // The final frame's candidate is never shaded by a later frame.
    if (uniforms.frame_index + 1u >= inv_params.dims.w) { return; }
    if ((inv_params.ctrl.x & INV_FLAG_SCORE) == 0u || !terrain_spectral_restir_enabled()) { return; }
    if ((inv_params.ctrl.x & INV_FLAG_SPATIAL) != 0u
        && terrain_gbuffer_pos[pix].w != 0.0) {
        let temporal = terrain_reservoirs_prev[pix];
        let q_sum = temporal.sample.params.x + temporal.sample.params.y
            + temporal.sample.params.z;
        if (temporal.sample.light_type == 1u && q_sum > 0.0) { return; }
    }
    let candidate = terrain_reservoirs_curr[pix];
    if (candidate.sample.light_type != 1u || candidate.sample.light_index >= 3u) { return; }
    inv_add_trace_score(candidate.sample.light_index, pix);
}

// Dispatch once immediately after each replayed spatial pass. Pixels without
// a G-buffer hit bypass that pass's spectral draw and contribute no score.
@compute @workgroup_size(256)
fn main_inverse_score_spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = gid.y * INV_DISPATCH_X + gid.x;
    if (pix >= uniforms.width * uniforms.height) { return; }
    // The final frame's redraw is likewise outside the realized image.
    if (uniforms.frame_index + 1u >= inv_params.dims.w) { return; }
    if ((inv_params.ctrl.x & INV_FLAG_SCORE) == 0u || !terrain_spectral_restir_enabled()) { return; }
    if (terrain_gbuffer_pos[pix].w == 0.0) { return; }
    // This binding is rebound to the temporal output for this score pass.
    // The spatial shader tests that input before drawing a new category.
    let input = terrain_reservoirs_curr[pix];
    let input_sum = input.sample.params.x + input.sample.params.y + input.sample.params.z;
    if (input.sample.light_type != 1u || !(input_sum > 0.0)) { return; }
    let redraw = terrain_reservoirs_prev[pix];
    let spectral_sum = redraw.sample.params.x + redraw.sample.params.y + redraw.sample.params.z;
    if (redraw.sample.light_type != 1u || !(spectral_sum > 0.0)
        || redraw.sample.light_index >= 3u) { return; }
    inv_add_trace_score(redraw.sample.light_index, pix);
}

@compute @workgroup_size(8, 8, 1)
fn main_inverse_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) { return; }
    let gx = gid.x;
    let gy = gid.y;
    let gpix = gy * uniforms.width + gx;
    let adj = inv_v4[gpix].rgb;
    let slots_k = inv_whist_slots();
    let f_end = uniforms.frame_index;
    let tile_base = f_end - (f_end % slots_k);
    let tile_count = f_end - tile_base + 1u;
    let spp = max(terrain.extra.x, 1u);
    let inv_fs = inv_params.fl.w;
    let omega = normalize(lighting.light_dir);
    let spectral = terrain_spectral_restir_enabled();
    let spectral_state = inv_spectral_state();
    let haze = terrain_env_haze();

    var d_sun = vec3<f32>(0.0);
    var d_intensity = 0.0;
    var d_tau = 0.0;

    for (var slot = 0u; slot < tile_count; slot = slot + 1u) {
        let f = tile_base + slot;
        let wh = inv_v4[inv_whist_index(slot, gpix)];
        let reservoir_weight = wh.x;
        let channel = u32(clamp(wh.y, 0.0, 3.0));
        let sampled = spectral && channel < 3u;
        var channel_mask = vec3<f32>(0.0);
        var sampled_spectrum = spectral_state.spectrum;
        var d_sampled_intensity = spectral_state.unit_spectrum;
        var d_sampled_tau = spectral_state.d_spectrum_tau;
        var d_sampled_wy = spectral_state.d_spectrum_wy;
        var dlogq_tau = 0.0;
        var dlogq_wy = 0.0;
        var weighted_residual_reward = 0.0;
        if (sampled) {
            channel_mask = terrain_spectral_channel_mask(channel);
            let residual = max(
                spectral_state.spectrum - vec3<f32>(spectral_state.baseline),
                vec3<f32>(0.0));
            let residual_unit = max(
                spectral_state.unit_spectrum - vec3<f32>(spectral_state.baseline_unit),
                vec3<f32>(0.0));
            sampled_spectrum = vec3<f32>(spectral_state.baseline)
                + channel_mask * residual * reservoir_weight;
            d_sampled_intensity = vec3<f32>(spectral_state.baseline_unit)
                + channel_mask * residual_unit * reservoir_weight;
            d_sampled_tau = vec3<f32>(spectral_state.d_baseline_tau)
                + channel_mask
                    * (spectral_state.d_spectrum_tau
                        - vec3<f32>(spectral_state.d_baseline_tau))
                    * reservoir_weight;
            d_sampled_wy = vec3<f32>(spectral_state.d_baseline_wy)
                + channel_mask
                    * (spectral_state.d_spectrum_wy
                        - vec3<f32>(spectral_state.d_baseline_wy))
                    * reservoir_weight;
            dlogq_tau = spectral_state.dlogq_tau[channel];
            dlogq_wy = spectral_state.dlogq_wy[channel];
        }

        var st = inv_seed(gpix, f);
        for (var s = 0u; s < spp; s = s + 1u) {
            let jx = terrain_tent_offset(xorshift32(&st)) * 0.5;
            let jy = terrain_tent_offset(xorshift32(&st)) * 0.5;
            let ray = inv_cam_ray(gx, gy, jx, jy);
            let hit = intersect_hybrid(ray);
            if (hit.hit == 0u) {
                d_tau += dot(adj, terrain_env_radiance(ray.direction))
                    * TERRAIN_ATMOS_HAZE * inv_fs;
                continue;
            }

            let n = hit.normal;
            let is_terrain = hit.hit_type == 3u;
            var albedo = vec3<f32>(0.0);
            var taps: TerrainAlbedoTaps;
            if (is_terrain) {
                taps = terrain_albedo_taps_at(hit.point);
                albedo = taps.alb;
            } else {
                albedo = get_surface_properties(hit);
            }

            let env_u1 = xorshift32(&st);
            let env_u2 = xorshift32(&st);
            let env_dir = terrain_cosine_dir(n, env_u1, env_u2);
            var env_visibility = 1.0;
            if (intersect_ibl_occlusion_ray(
                Ray(hit.point + n * 1e-3, 1e-3, env_dir, 1e30), 1e30)) {
                env_visibility = 0.0;
            }
            let env_radiance = terrain_env_radiance(env_dir);

            let nd_raw = dot(n, omega);
            let nd = max(nd_raw, 0.0);
            var sun_visibility = 1.0;
            if (nd > 0.0 && lighting.shadows_enabled != 0u
                && intersect_shadow_ray(
                    Ray(hit.point + n * 1e-3, 1e-3, omega, 1e30), 1e30)) {
                sun_visibility = 0.0;
            }

            d_sun += n * dot(adj, albedo * sampled_spectrum * sun_visibility)
                * select(0.0, 1.0, nd_raw > 0.0) * inv_fs;
            d_sun.y += dot(
                adj,
                albedo * d_sampled_wy * nd * sun_visibility) * inv_fs;
            d_intensity += dot(
                adj,
                albedo * d_sampled_intensity * nd * sun_visibility) * inv_fs;
            d_tau += dot(
                adj,
                albedo * d_sampled_tau * nd * sun_visibility) * inv_fs;
            d_tau += dot(adj, albedo * env_radiance * env_visibility)
                * TERRAIN_ATMOS_HAZE * inv_fs;

            if (sampled) {
                let residual = max(
                    spectral_state.spectrum - vec3<f32>(spectral_state.baseline),
                    vec3<f32>(0.0));
                weighted_residual_reward += dot(
                    adj,
                    albedo * channel_mask * residual * reservoir_weight
                        * nd * sun_visibility) * inv_fs;
            }

            if (is_terrain && taps.texels.x != 0xffffffffu) {
                let albedo_gradient = adj * (
                    sampled_spectrum * nd * sun_visibility
                    + env_radiance * haze * env_visibility) * inv_fs;
                for (var tap = 0u; tap < 4u; tap = tap + 1u) {
                    let tap_weight = taps.weights[tap];
                    if (tap_weight > 0.0) {
                        inv_add_albedo(taps.texels[tap], albedo_gradient * tap_weight);
                    }
                }
            }
        }

        if (sampled) {
            d_sun.y -= weighted_residual_reward * dlogq_wy;
            d_tau -= weighted_residual_reward * dlogq_tau;
        }
    }

    inv_add_scalar(INV_GRAD_SUN_X, d_sun.x);
    inv_add_scalar(INV_GRAD_SUN_Y, d_sun.y);
    inv_add_scalar(INV_GRAD_SUN_Z, d_sun.z);
    inv_add_scalar(INV_GRAD_INTENSITY, d_intensity);
    inv_add_scalar(INV_GRAD_TURBIDITY, d_tau);
}
