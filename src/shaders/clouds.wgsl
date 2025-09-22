// src/shaders/clouds.wgsl
// Procedural realtime cloud shader sampling noise volumes and IBL lighting.
// Provides hybrid billboard/volumetric shading paths for the realtime cloud pass.
// RELEVANT FILES: src/core/clouds.rs, src/scene/mod.rs, tests/test_b8_clouds.py, examples/clouds_demo.py

struct CloudUniforms {
    view_proj: mat4x4<f32>,                    // View-projection matrix
    camera_pos: vec4<f32>,                     // Camera position (xyz) + accumulated time (w)
    sky_params: vec4<f32>,                     // Sky color (rgb) + sun intensity (w)
    sun_direction: vec4<f32>,                  // Sun direction (xyz) + density scale (w)
    cloud_params: vec4<f32>,                   // coverage (x), scale (y), height (z), fade_distance (w)
    wind_params: vec4<f32>,                    // wind_dir (xy), wind_strength (z), animation_speed (w)
    scattering_params: vec4<f32>,              // scatter_strength, absorption, phase_g, ambient
    render_params: vec4<f32>,                  // max_steps, step_size, billboard_threshold, mode flag
};

@group(0) @binding(0) var<uniform> cloud_uniforms : CloudUniforms;

@group(1) @binding(0) var cloud_noise_tex : texture_3d<f32>;
@group(1) @binding(1) var cloud_noise_samp : sampler;
@group(1) @binding(2) var cloud_shape_tex : texture_2d<f32>;
@group(1) @binding(3) var cloud_shape_samp : sampler;

@group(2) @binding(0) var ibl_irradiance_tex : texture_cube<f32>;
@group(2) @binding(1) var ibl_irradiance_samp : sampler;
@group(2) @binding(2) var ibl_prefilter_tex : texture_cube<f32>;
@group(2) @binding(3) var ibl_prefilter_samp : sampler;

struct VsIn {
    @location(0) position: vec3<f32>;
    @location(1) uv: vec2<f32>;
    @location(2) normal: vec3<f32>;
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>;
    @location(0) uv: vec2<f32>;
};

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn noise3d(p: vec3<f32>) -> f32 {
    return textureSampleLevel(cloud_noise_tex, cloud_noise_samp, p, 0.0).r;
}

fn fbm3d(pos: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var position = pos;

    for (var i = 0; i < octaves; i = i + 1) {
        value += amplitude * noise3d(position * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
        position = position * 1.73 + vec3<f32>(23.17, 7.33, 11.91);
    }

    return saturate(value * 1.1);
}

fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return (1.0 - g2) / max(denom, 1e-3) * (1.0 / (4.0 * 3.14159265));
}

fn compute_cloud_density(tex_uv: vec2<f32>) -> f32 {
    let time = cloud_uniforms.camera_pos.w * cloud_uniforms.wind_params.w;
    let wind_offset = cloud_uniforms.wind_params.xy * cloud_uniforms.wind_params.z * time;
    let base_scale = cloud_uniforms.cloud_params.y * 0.01;
    let sample_pos = vec3<f32>(tex_uv * base_scale + wind_offset, time * 0.05);
    let octaves = clamp(i32(cloud_uniforms.render_params.x / 16.0), 2, 6);

    let base = fbm3d(sample_pos, octaves);
    let detail = fbm3d(sample_pos * 2.7 + vec3<f32>(17.3, 9.1, 3.7), octaves + 1);
    let combined = saturate(base * 0.65 + detail * 0.35);

    let coverage = cloud_uniforms.cloud_params.x;
    let density_scale = cloud_uniforms.sun_direction.w;
    let density = smoothstep(1.0 - coverage, 1.0, combined) * density_scale;

    // Shape texture acts as soft alpha mask for billboard mode
    let shape_sample = textureSample(cloud_shape_tex, cloud_shape_samp, (tex_uv * 0.5 + 0.5));
    let shape_alpha = saturate(shape_sample.r * 1.15);

    return saturate(density * mix(0.85, 1.0, shape_alpha));
}

fn evaluate_scattering(density: f32, view_dir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(cloud_uniforms.sun_direction.xyz);
    let cos_theta = dot(view_dir, sun_dir);
    let phase = henyey_greenstein_phase(cos_theta, cloud_uniforms.scattering_params.z);
    let sun_intensity = cloud_uniforms.sky_params.w;
    let scatter_strength = cloud_uniforms.scattering_params.x * sun_intensity;
    let sun_light = scatter_strength * phase * density;

    let absorption = exp(-cloud_uniforms.scattering_params.y * density);
    let ambient = cloud_uniforms.scattering_params.w;

    // Sample low frequency irradiance and a subtle reflection tint
    let irradiance = textureSampleLevel(ibl_irradiance_tex, ibl_irradiance_samp, normal, 0.0).rgb;
    let reflection = textureSampleLevel(ibl_prefilter_tex, ibl_prefilter_samp, reflect(-view_dir, normal), 0.0).rgb;

    let sky_color = cloud_uniforms.sky_params.rgb;
    return (sky_color * (sun_light + ambient) + irradiance * 0.35 + reflection * 0.15) * absorption;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(in.position.xy, 0.0, 1.0);
    out.uv = in.uv * 2.0 - 1.0; // Map to [-1,1] for procedural lookups
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let density = compute_cloud_density(in.uv);
    if density < 0.01 {
        return vec4<f32>(0.0);
    }

    let mode = cloud_uniforms.render_params.w;
    var density_mod = density;
    if mode < 0.5 {
        density_mod *= 0.85; // Billboard emphasis
    } else if mode > 1.5 {
        let radial = length(in.uv);
        let falloff = saturate(1.0 - radial);
        density_mod = mix(density_mod * 0.75, density_mod, falloff);
    } else {
        density_mod = saturate(density_mod * (cloud_uniforms.render_params.x / 32.0));
    }

    let blend_alpha = saturate(density_mod);
    if blend_alpha <= 0.001 {
        return vec4<f32>(0.0);
    }

    let view_dir = normalize(vec3<f32>(in.uv, 1.5));
    let normal = normalize(vec3<f32>(0.15 * in.uv.x, 1.0, 0.15 * in.uv.y));
    let color = evaluate_scattering(density_mod, view_dir, normal);

    return vec4<f32>(color * blend_alpha, blend_alpha);
}
