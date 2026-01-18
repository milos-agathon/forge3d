// src/shaders/viewer_volumetrics.wgsl
// P5: Volumetric fog and light shafts for terrain viewer

struct VolumetricsUniforms {
    // Camera (64 bytes)
    inv_view_proj: mat4x4<f32>,
    // camera_pos.xyz, pad (16 bytes)
    camera_pos: vec4<f32>,
    // near, far, density, height_falloff (16 bytes)
    near_far: vec4<f32>,
    // scattering, absorption, pad, pad (16 bytes)
    scatter_absorb: vec4<f32>,
    // sun_direction.xyz, sun_intensity (16 bytes)
    sun_direction: vec4<f32>,
    // shaft_intensity, light_shafts_enabled, steps, mode (16 bytes)
    shaft_params: vec4<f32>,
    // screen_width, screen_height, pad, pad (16 bytes)
    screen_dims: vec4<f32>,
    // terrain_width, min_h, z_scale, h_range (16 bytes)
    terrain_params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: VolumetricsUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var depth_sampler: sampler;
@group(0) @binding(5) var heightmap_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

// Reconstruct world position from depth
fn world_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world_h = u.inv_view_proj * ndc;
    return world_h.xyz / world_h.w;
}

// Extract uniform fields for clarity
fn get_density() -> f32 { return u.near_far.z; }
fn get_height_falloff() -> f32 { return u.near_far.w; }
fn get_scattering() -> f32 { return u.scatter_absorb.x; }
fn get_absorption() -> f32 { return u.scatter_absorb.y; }
fn get_sun_intensity() -> f32 { return u.sun_direction.w; }
fn get_shaft_intensity() -> f32 { return u.shaft_params.x; }
fn get_light_shafts_enabled() -> bool { return u.shaft_params.y > 0.5; }
fn get_steps() -> u32 { return u32(u.shaft_params.z); }
fn get_mode() -> u32 { return u32(u.shaft_params.w); }
fn get_near() -> f32 { return u.near_far.x; }
fn get_far() -> f32 { return u.near_far.y; }

// Helper to get terrain height at world pos (xz)
fn get_terrain_height(world_pos: vec3<f32>) -> f32 {
    let terrain_width = u.terrain_params.x;
    let min_h = u.terrain_params.y;
    let z_scale = u.terrain_params.z;
    
    // UV coordinates: 0..1 maps to 0..terrain_width
    let uv = world_pos.xz / terrain_width;
    
    // Check bounds
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return -99999.0;
    }
    
    // Sample heightmap (R32Float) - use dimensions to convert UV to pixel coords
    let dims = vec2<f32>(textureDimensions(heightmap_tex));
    let texel = vec2<i32>(uv * dims);
    let h_raw = textureLoad(heightmap_tex, texel, 0).r;
    
    // Convert to world height
    return (h_raw - min_h) * z_scale; // Assuming world logic matches shader_pbr
}

// Height-based fog density
fn fog_density_at(world_pos: vec3<f32>) -> f32 {
    let density = get_density();
    if get_mode() == 0u {
        // Uniform fog
        return density;
    } else {
        // Height-based exponential falloff
        // Use height relative to terrain center (0.5 * terrain height range)
        // This makes fog denser in valleys and thinner at peaks
        let terrain_width = u.terrain_params.x;
        let z_scale = u.terrain_params.z;
        let h_range = u.terrain_params.w;
        let terrain_max_height = h_range * z_scale;
        
        // Normalize height: 0 at base, 1 at max terrain height
        let height_normalized = clamp(world_pos.y / max(terrain_max_height, 1.0), 0.0, 2.0);
        
        // Use height_falloff as a meaningful parameter (0.01 = weak falloff, 1.0 = strong)
        let falloff_rate = get_height_falloff() * 5.0;
        return density * exp(-height_normalized * falloff_rate);
    }
}

// Henyey-Greenstein phase function for anisotropic scattering
fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// Light shaft shadow estimation using sun ray march through volume
// Returns visibility factor [0,1] for light shafts with soft penumbra
fn estimate_light_shaft_shadow(world_pos: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    // March toward sun to estimate shadow from terrain
    // We check if the line to sun is occluded by terrain geometry
    
    // Use terrain dimensions for proper shadow ray length
    let terrain_width = u.terrain_params.x;
    let z_scale = u.terrain_params.z;
    let h_range = u.terrain_params.w;
    let terrain_max_height = h_range * z_scale;
    
    // Higher step count for sharper beam edges
    let shaft_steps = 24u;
    let shaft_distance = max(terrain_width * 0.6, terrain_max_height * 2.5);
    let step_size = shaft_distance / f32(shaft_steps);
    
    var occlusion = 0.0;
    var partial_shadow = 0.0;  // Track near-misses for soft penumbra
    
    // Starting offset to avoid self-occlusion artifacts (relative to terrain scale)
    let start_offset = terrain_width * 0.003;
    
    for (var i = 0u; i < shaft_steps; i++) {
        let t = start_offset + (f32(i) + 0.5) * step_size;
        let sample_pos = world_pos + sun_dir * t;
        
        // Check terrain occlusion
        let terrain_h = get_terrain_height(sample_pos);
        if terrain_h > -90000.0 {
            let height_margin = sample_pos.y - terrain_h;
            
            if height_margin < 0.0 {
                // Fully occluded by terrain - hard shadow
                return 0.0;
            }
            
            // Soft penumbra: track how close we are to terrain silhouette
            // This creates subtle gradients at beam edges
            let penumbra_range = terrain_max_height * 0.05;
            if height_margin < penumbra_range {
                let penumbra_factor = height_margin / penumbra_range;
                partial_shadow = max(partial_shadow, 1.0 - penumbra_factor);
            }
        }
        
        // Volumetric self-shadowing from fog density
        let local_density = fog_density_at(sample_pos);
        occlusion += local_density * step_size * 0.008 / max(terrain_width * 0.1, 100.0);
        
        if occlusion > 2.5 {
            break;
        }
    }
    
    // Combine terrain penumbra with volumetric occlusion
    let volumetric_shadow = exp(-occlusion * get_absorption() * 4.0);
    let penumbra_shadow = 1.0 - partial_shadow * 0.6;
    
    return volumetric_shadow * penumbra_shadow;
}

// Ray march through fog volume with depth-aware termination
// Returns fog color (rgb) and fog amount (a) for proper compositing
fn raymarch_fog_with_depth(ray_origin: vec3<f32>, ray_dir: vec3<f32>, scene_dist: f32) -> vec4<f32> {
    let steps = get_steps();
    let sun_dir = normalize(u.sun_direction.xyz);
    let scattering = get_scattering();
    let absorption = get_absorption();
    let shaft_intensity = get_shaft_intensity();
    let sun_intensity = get_sun_intensity();
    let base_density = get_density();
    
    // Use terrain dimensions for proper scale reference
    let terrain_width = u.terrain_params.x;
    let h_range = u.terrain_params.w;
    let z_scale = u.terrain_params.z;
    let terrain_max_height = h_range * z_scale;
    
    // Reference scale: typical viewing distance as fraction of terrain
    let reference_dist = terrain_width * 0.5;
    let density_coeff = base_density * 100.0;
    
    // Limit fog distance to scene geometry
    let fog_max_dist = scene_dist;
    let step_size = fog_max_dist / f32(steps);
    
    // ===== ENHANCED COLOR PALETTE FOR GOD RAYS =====
    // Cool shadow color (blue-gray for shadowed fog)
    let fog_shadow_color = vec3<f32>(0.65, 0.72, 0.85);
    // Warm lit color (golden-orange for sun-lit fog)  
    let fog_lit_color = vec3<f32>(1.0, 0.88, 0.65);
    // Intense shaft color (bright golden-white for beam highlights)
    let shaft_highlight = vec3<f32>(1.0, 0.95, 0.75);
    // Sky ambient (subtle blue contribution from sky)
    let sky_ambient = vec3<f32>(0.5, 0.6, 0.8);
    
    var accumulated_fog = 0.0;
    var light_accumulation = 0.0;
    var shaft_luminance = 0.0;
    var shadow_accumulation = 0.0;  // Track shadowed regions for contrast
    
    // Calculate view-to-sun angle for forward scattering
    let cos_view_sun = dot(ray_dir, sun_dir);
    
    // ===== RADIAL BEAM PATTERN =====
    // Boost intensity when looking toward sun (classic radial god ray look)
    // Uses a sharper falloff for more defined beams
    let radial_core = smoothstep(0.6, 0.999, cos_view_sun);  // Intense core
    let radial_halo = smoothstep(0.2, 0.8, cos_view_sun);     // Soft halo
    let forward_boost = radial_core * 5.0 + radial_halo * 2.0 + 1.0;
    
    // Cornette-Shanks phase for more realistic Mie scattering
    // Better approximation for aerosol/fog scattering than pure HG
    let g = clamp(scattering + 0.35, 0.0, 0.96);
    let g2 = g * g;
    let cs_denom = 1.0 + g2 - 2.0 * g * cos_view_sun;
    let phase_cs = 1.5 * (1.0 - g2) * (1.0 + cos_view_sun * cos_view_sun) / 
                   ((2.0 + g2) * pow(cs_denom, 1.5));
    
    // Jitter start position to reduce banding
    let jitter = fract(sin(dot(ray_dir.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 0.5;
    
    for (var i = 0u; i < steps; i++) {
        let t = (f32(i) + 0.5 + jitter) * step_size;
        if t >= scene_dist {
            break;
        }
        
        let pos = ray_origin + ray_dir * t;
        
        // Get local fog density with distance normalization
        let local_density = fog_density_at(pos) * density_coeff / max(reference_dist, 100.0);
        
        if local_density < 0.00001 {
            continue;
        }
        
        // Light shaft shadow estimation (terrain occlusion check)
        let shadow = estimate_light_shaft_shadow(pos, sun_dir);
        
        // ===== 10/10 GOD RAYS =====
        // Multi-component light accumulation for rich, realistic beams
        
        let transmittance = 1.0 - accumulated_fog;
        
        // 1. Primary shaft contribution (direct sunlight through fog)
        let primary_shaft = shadow * phase_cs * sun_intensity * shaft_intensity * forward_boost;
        
        // 2. Secondary scatter (multi-scatter approximation for richer atmosphere)
        // Light that bounces once in the fog - creates softer fill
        let secondary_scatter = shadow * 0.15 * sun_intensity * (1.0 + radial_halo * 0.5);
        
        // 3. Ambient sky contribution (always present, gives depth to shadows)
        let ambient_contrib = 0.08;
        
        // Accumulate shaft brightness with depth falloff
        // Beams become less intense further from camera (atmospheric perspective)
        let depth_factor = 1.0 - smoothstep(0.0, 1.0, t / max(scene_dist, 1.0)) * 0.3;
        shaft_luminance += (primary_shaft + secondary_scatter) * local_density * step_size * transmittance * depth_factor * 2.5;
        
        // Track light vs shadow for color mixing
        light_accumulation += shadow * local_density * step_size * transmittance;
        shadow_accumulation += (1.0 - shadow) * local_density * step_size * transmittance;
        
        // Accumulated fog density via Beer-Lambert
        let fog_step = local_density * step_size * (1.0 + absorption);
        accumulated_fog += fog_step * (1.0 - accumulated_fog);
        
        // Early exit if fog is saturated
        if accumulated_fog > 0.88 {
            accumulated_fog = 0.88;
            break;
        }
    }
    
    // ===== FINAL COLOR COMPOSITION (10/10 QUALITY) =====
    
    // Normalize light/shadow accumulation
    let total_contrib = light_accumulation + shadow_accumulation + 0.001;
    let lit_ratio = light_accumulation / total_contrib;
    let shadow_ratio = shadow_accumulation / total_contrib;
    
    // Temperature-based color gradation: 
    // - Lit regions are warm (golden)
    // - Shadowed regions are cool (blue-gray)
    var fog_color = mix(fog_shadow_color, fog_lit_color, lit_ratio);
    
    // Add subtle sky ambient to create depth
    fog_color = fog_color + sky_ambient * shadow_ratio * 0.15;
    
    // Add bright shaft highlights (creates the "glow" effect)
    let clamped_shaft = clamp(shaft_luminance, 0.0, 2.0);
    fog_color = fog_color + shaft_highlight * clamped_shaft * 0.5;
    
    // Subtle bloom simulation for very bright shafts
    if clamped_shaft > 1.0 {
        let bloom = (clamped_shaft - 1.0) * 0.2;
        fog_color = fog_color + vec3<f32>(bloom, bloom * 0.9, bloom * 0.7);
    }
    
    // Final HDR-safe clamp (allow slight overexposure for dramatic effect)
    fog_color = clamp(fog_color, vec3<f32>(0.0), vec3<f32>(2.0));
    
    return vec4<f32>(fog_color, accumulated_fog);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    
    // Sample scene color
    let scene_color = textureSample(color_tex, color_sampler, uv).rgb;
    
    // Get density - if zero, just pass through (baseline preservation)
    let density = get_density();
    if density < 0.0001 {
        return vec4<f32>(scene_color, 1.0);
    }
    
    // Sample depth buffer to get scene distance
    let ndc_depth = textureSample(depth_tex, depth_sampler, uv);
    let near = get_near();
    let far = get_far();
    
    // Detect sky pixels (at or very close to far plane in Standard Z)
    // In Standard Z (used by wgpu default): ndc_depth near 1.0 means far plane (sky)
    let is_sky = ndc_depth > 0.999;
    
    // For sky pixels, skip volumetric processing to avoid artifacts
    if is_sky {
        return vec4<f32>(scene_color, 1.0);
    }
    
    // Reconstruct world position and calculate ray parameters
    let camera_pos = u.camera_pos.xyz;
    let world_pos = world_pos_from_depth(uv, ndc_depth);
    let ray_dir = normalize(world_pos - camera_pos);
    let scene_dist = length(world_pos - camera_pos);
    
    // Use terrain dimensions for proper scale reference
    let terrain_width = u.terrain_params.x;
    let h_range = u.terrain_params.w;
    let z_scale = u.terrain_params.z;
    let terrain_max_height = h_range * z_scale;
    
    // Reference distance: fraction of terrain width for fog calculations
    let reference_dist = terrain_width * 0.3;
    
    // Use raymarching for proper volumetric effects (especially light shafts)
    let light_shafts = get_light_shafts_enabled();
    
    if light_shafts {
        // Full volumetric raymarching with light shafts
        let vol_result = raymarch_fog_with_depth(camera_pos, ray_dir, scene_dist);
        let fog_color = vol_result.rgb;
        let fog_amount = vol_result.a;
        
        // Composite with scene
        let final_color = mix(scene_color, fog_color, fog_amount);
        return vec4<f32>(final_color, 1.0);
    }
    
    // Simplified fog for non-light-shaft modes (faster)
    let mode = get_mode();
    var fog_amount: f32;
    
    // Normalized distance: how far we are relative to a reference viewing distance
    let dist_normalized = scene_dist / max(reference_dist, 100.0);
    
    // ===== 10/10 QUALITY FOG ENHANCEMENTS =====
    
    // Camera and world space calculations for advanced effects
    let camera_height = camera_pos.y;
    let world_height = world_pos.y;
    let height_norm = clamp(world_pos.y / max(terrain_max_height, 1.0), -0.5, 2.5);
    let camera_height_norm = clamp(camera_pos.y / max(terrain_max_height, 1.0), 0.0, 2.0);
    
    // Ray direction vertical component for horizon detection
    let ray_vertical = ray_dir.y;
    let is_looking_down = ray_vertical < -0.1;
    let horizon_factor = 1.0 - abs(ray_vertical);  // 1.0 at horizon, 0 looking straight up/down
    
    // Subtle pseudo-noise for organic appearance (based on world position)
    let noise_input = world_pos.x * 0.001 + world_pos.z * 0.0013 + world_pos.y * 0.0007;
    let density_noise = 1.0 + (sin(noise_input * 17.3) * 0.15 + sin(noise_input * 31.7) * 0.1);
    
    if mode == 0u {
        // ===== UNIFORM FOG (10/10 QUALITY) =====
        // Multi-layer atmospheric simulation with aerial perspective
            
        // Layer 1: Ground-level haze (primary fog)
        // Scalar 1.0 provides subtle atmospheric depth at density=0.01
        let fog_optical_depth = density * 1.0 * dist_normalized * density_noise;
        let primary_fog = 1.0 - exp(-fog_optical_depth);
            
        // Layer 2: Aerial perspective (increases with distance for depth perception)
        // Creates the characteristic "bluing" of distant objects
        let aerial_factor = 1.0 - exp(-dist_normalized * 0.02);
            
        // Layer 3: Horizon enhancement (fog is denser near horizon)
        let horizon_density = horizon_factor * horizon_factor * 0.25;
            
        // Combine layers with proper weighting
        fog_amount = primary_fog * (1.0 + horizon_density) + aerial_factor * 0.05;
            
    } else {
        // ===== HEIGHT FOG (10/10 QUALITY) =====
        // Realistic low-lying fog with layering and organic variation
            
        let height_falloff = get_height_falloff();
            
        // Layer 1: Dense ground fog (concentrated in valleys)
        // Exponential falloff from terrain base
        let ground_fog_height = 0.25;  // Fog concentrated in bottom 25% of terrain
        let ground_factor = exp(-max(height_norm - 0.0, 0.0) / ground_fog_height * height_falloff * 2.0);
            
        // Layer 2: Mid-level haze (subtle, extends higher)
        let mid_haze_factor = exp(-max(height_norm, 0.0) * height_falloff * 0.8);
            
        // Layer 3: Fog density variation (thicker in low spots, thinner on ridges)
        // Creates more organic, cloud-like appearance
        let layer_variation = density_noise * (0.7 + 0.3 * ground_factor);
            
        // Distance-based optical depth
        // Scalar 1.0 provides consistent behavior with uniform mode
        let fog_optical_depth = density * 1.0 * dist_normalized * layer_variation;
        let base_fog = 1.0 - exp(-fog_optical_depth);
            
        // Combine height factors: heavy ground fog + subtle mid-level haze
        let combined_height_factor = ground_factor * 0.85 + mid_haze_factor * 0.15;
            
        // Apply height modulation with minimum visibility at all levels
        fog_amount = base_fog * (0.12 + 0.88 * combined_height_factor);
            
        // Enhance fog when looking down into valleys
        if is_looking_down && height_norm < 0.5 {
            let valley_enhance = (0.5 - height_norm) * 0.3;
            fog_amount = fog_amount + valley_enhance * base_fog;
        }
    }
        
    // Clamp fog to preserve scene visibility while allowing dramatic effect
    fog_amount = clamp(fog_amount, 0.0, 0.6);
        
    // ===== ADVANCED FOG COLOR CALCULATION =====
    let sun_dir = normalize(u.sun_direction.xyz);
    let cos_angle = dot(ray_dir, sun_dir);
    let scattering = get_scattering();
        
    // Cornette-Shanks phase function for realistic scattering
    let g = clamp(scattering + 0.1, 0.0, 0.9);
    let g2 = g * g;
    let cs_denom = 1.0 + g2 - 2.0 * g * cos_angle;
    let phase = 1.5 * (1.0 - g2) * (1.0 + cos_angle * cos_angle) / ((2.0 + g2) * pow(cs_denom, 1.5));
        
    // ===== MULTI-COLOR ATMOSPHERE =====
        
    // Cool distant color (Rayleigh-like blue for aerial perspective)
    let aerial_color = vec3<f32>(0.55, 0.68, 0.88);
        
    // Neutral mid-distance fog color
    let fog_mid_color = vec3<f32>(0.72, 0.78, 0.86);
        
    // Warm near color (subtle warmth from ground-reflected light)
    let fog_near_color = vec3<f32>(0.82, 0.82, 0.80);
        
    // Sun-influenced color (warm golden for forward scatter)
    let sun_color = vec3<f32>(1.0, 0.92, 0.78);
        
    // Sky contribution (adds subtle blue from above)
    let sky_contribution = vec3<f32>(0.4, 0.55, 0.75);
        
    // ===== COLOR BLENDING =====
        
    // Distance-based color gradation (aerial perspective)
    let distance_blend = clamp(dist_normalized, 0.0, 1.0);
    var fog_color = mix(fog_near_color, fog_mid_color, distance_blend * 0.6);
    // Subtle aerial perspective (5%) for depth without washout
    fog_color = mix(fog_color, aerial_color, distance_blend * distance_blend * 0.05);
        
    // Height-based color modulation (higher = bluer, lower = neutral/warm)
    if mode == 1u {
        // Height fog: add subtle warmth in valleys, cooler at elevation
        let height_color_factor = clamp(height_norm, 0.0, 1.0);
        fog_color = mix(fog_color, fog_color * vec3<f32>(1.02, 0.98, 0.94), 1.0 - height_color_factor);
        
        // Add subtle sky contribution from above
        fog_color = fog_color + sky_contribution * height_color_factor * 0.08;
    }
    
    // Sun scattering contribution
    let sun_intensity = get_sun_intensity();
    let scatter_strength = phase * sun_intensity * scattering * 1.2;
    fog_color = mix(fog_color, sun_color, clamp(scatter_strength, 0.0, 0.5));
    
    // Horizon glow enhancement (subtle warm tint at horizon)
    let horizon_glow = horizon_factor * horizon_factor * 0.12;
    fog_color = fog_color + vec3<f32>(0.08, 0.05, 0.02) * horizon_glow * sun_intensity;
    
    // Apply absorption (subtle density-based darkening)
    let absorption = get_absorption();
    fog_color *= (1.0 - absorption * 0.15);
    
    // Ensure valid color range
    fog_color = clamp(fog_color, vec3<f32>(0.0), vec3<f32>(1.2));
    
    // Composite: blend scene with fog
    let final_color = mix(scene_color, fog_color, fog_amount);
    
    return vec4<f32>(final_color, 1.0);
}
