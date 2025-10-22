// src/shaders/hybrid_kernel.wgsl
// Main compute kernel for hybrid path tracing combining SDF raymarching with BVH traversal
// Extends pt_kernel.wgsl functionality with SDF support

#include "hybrid_traversal.wgsl"
#include "sampling/sobol.wgsl"
#include "sampling/cmj.wgsl"
#include "common/hdri.wgsl"

// Base uniforms (Group 0) - FROZEN layout, do not modify
struct Uniforms {
  width: u32,
  height: u32,
  frame_index: u32,
  aov_flags: u32,
  cam_origin: vec3<f32>,
  cam_fov_y: f32,
  cam_right: vec3<f32>,
  cam_aspect: f32,
  cam_up: vec3<f32>,
  cam_exposure: f32,
  cam_forward: vec3<f32>,
  seed_hi: u32,
  seed_lo: u32,
  sampling_mode: u32,        // 0=rng, 1=sobol, 2=cmj
  lighting_type: u32,
  lighting_intensity: f32,
  lighting_azimuth: f32,
  lighting_elevation: f32,
  shadow_intensity: f32,
  hdri_intensity: f32,
  hdri_rotation: f32,
  hdri_exposure: f32,
  hdri_enabled: f32,
}

// TileParams (Group 2) - for true-resolution tiled rendering
struct TileParams {
  tile_origin_size: vec4<u32>,  // (tile_x_px, tile_y_px, tile_w, tile_h)
  img_spp_counts: vec4<u32>,    // (img_w, img_h, spp_batch, spp_done)
  seeds_misc: vec4<u32>,        // (seed, reserved0, reserved1, reserved2)
}

// Sphere primitive for legacy support
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    _pad0: f32
}

// Bind groups
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(1) @binding(5) var hdri_tex: texture_2d<f32>;
@group(1) @binding(6) var hdri_samp: sampler;
@group(2) @binding(0) var<uniform> tile_params: TileParams;
@group(2) @binding(1) var<storage, read_write> accum_hdr: array<vec4<f32>>;
@group(3) @binding(0) var out_tex: texture_storage_2d<rgba16float, write>;

// AOV Output textures (moved into Group 3 to stay within max_bind_groups)
@group(3) @binding(1) var aov_albedo: texture_storage_2d<rgba16float, write>;
@group(3) @binding(2) var aov_normal: texture_storage_2d<rgba16float, write>;
@group(3) @binding(3) var aov_depth: texture_storage_2d<r32float, write>;
@group(3) @binding(4) var aov_direct: texture_storage_2d<rgba16float, write>;
@group(3) @binding(5) var aov_indirect: texture_storage_2d<rgba16float, write>;
@group(3) @binding(6) var aov_emission: texture_storage_2d<rgba16float, write>;
@group(3) @binding(7) var aov_visibility: texture_storage_2d<rgba8unorm, write>;

// AOV flag constants
const AOV_ALBEDO_BIT: u32 = 0u;
const AOV_NORMAL_BIT: u32 = 1u;
const AOV_DEPTH_BIT: u32 = 2u;
const AOV_DIRECT_BIT: u32 = 3u;
const AOV_INDIRECT_BIT: u32 = 4u;
const AOV_EMISSION_BIT: u32 = 5u;
const AOV_VISIBILITY_BIT: u32 = 6u;

fn aov_enabled(bit: u32) -> bool {
    return (uniforms.aov_flags & (1u << bit)) != 0u;
}

// Random number generation
fn xorshift32(state: ptr<function, u32>) -> f32 {
  var x = *state;
  x ^= (x << 13u);
  x ^= (x >> 17u);
  x ^= (x << 5u);
  *state = x;
  return f32(x) / 4294967296.0;
}

// Unified sampling interface - selects between RNG, Sobol, or CMJ
fn get_sample_2d(pixel: vec2<u32>, sample_idx: u32, dim: u32, rng_state: ptr<function, u32>) -> vec2<f32> {
  if (uniforms.sampling_mode == 1u) {
    // Sobol sequence
    return sobol_sample(pixel, sample_idx, dim, uniforms.seed_hi);
  } else if (uniforms.sampling_mode == 2u) {
    // CMJ sampling (assuming 8x8 stratification for 64 samples)
    return cmj_sample(pixel, sample_idx, 8u, uniforms.seed_hi);
  } else {
    // Default: pseudo-random (for backward compatibility)
    return vec2<f32>(xorshift32(rng_state), xorshift32(rng_state));
  }
}

// Tent filter for antialiasing
fn tent_filter(u: f32) -> f32 {
  let t = 2.0 * u - 1.0;
  return select(1.0 + t, 1.0 - t, t < 0.0);
}

// Ray-sphere intersection (for legacy sphere support)
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  let oc = ro - c;
  let b = dot(oc, rd);
  let cterm = dot(oc, oc) - r * r;
  let disc = b * b - cterm;
  if (disc <= 0.0) { return 1e30; }
  let s = sqrt(disc);
  let t0 = -b - s;
  let t1 = -b + s;
  if (t0 > 1e-3) { return t0; }
  if (t1 > 1e-3) { return t1; }
  return 1e30;
}

// Simple tonemap function
fn reinhard_tonemap(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = color * exposure;
    return exposed / (vec3<f32>(1.0) + exposed);
}

fn procedural_sky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(0.5 * (dir.y + 1.0), 0.0, 1.0);
    let sky = mix(vec3<f32>(0.85, 0.9, 1.0), vec3<f32>(0.2, 0.35, 0.7), t);
    let ground = vec3<f32>(0.04, 0.05, 0.06);
    return mix(ground, sky, t);
}

// Robust sky sampling: prefer HDRI when enabled, but fall back to procedural sky
// if the sampled HDRI is effectively black (e.g., missing/invalid texel) to avoid
// fully black tiles in miss paths. Also enforce a tiny luminance floor for stability.
fn sample_sky(dir: vec3<f32>) -> vec3<f32> {
    var sky_rgb = vec3<f32>(0.0);
    if (uniforms.hdri_enabled > 0.5) {
        let env_rgb = sample_environment(dir, uniforms.hdri_rotation, hdri_tex, hdri_samp);
        let env_lit = env_rgb * uniforms.hdri_intensity;
        let tonemapped = reinhard_tonemap(env_lit, uniforms.hdri_exposure);
        let lum = dot(tonemapped, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (lum < 1e-5) {
            // Degenerate/black HDRI sample: fall back to procedural sky
            sky_rgb = procedural_sky(dir);
        } else {
            sky_rgb = tonemapped;
        }
    } else {
        sky_rgb = procedural_sky(dir);
    }
    // Prevent fully black sky due to edge cases; minimal brightness floor
    let floor_val = 0.02;
    sky_rgb = max(sky_rgb, vec3<f32>(floor_val, floor_val, floor_val));
    return sky_rgb;
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lx = gid.x;
  let ly = gid.y;
  let tile_w = tile_params.tile_origin_size.z;
  let tile_h = tile_params.tile_origin_size.w;
  
  // Early exit if outside tile bounds
  if (lx >= tile_w || ly >= tile_h) { return; }
  
  // Compute GLOBAL pixel coordinates for seamless tiling
  let gx = tile_params.tile_origin_size.x + lx;
  let gy = tile_params.tile_origin_size.y + ly;
  let img_w = tile_params.img_spp_counts.x;
  let img_h = tile_params.img_spp_counts.y;
  let spp_batch = tile_params.img_spp_counts.z;
  let debug_mode = tile_params.seeds_misc.y;
  
  // Safety check against full image bounds
  if (gx >= img_w || gy >= img_h) { return; }
  
  // P1 DIAGNOSTIC MODE: Test texture writes and TileParams binding
  if (debug_mode == 1u) {
    let u = f32(gx) / f32(max(img_w, 1u));
    let v = f32(gy) / f32(max(img_h, 1u));
    let checker = f32(((gx/32u + gy/32u) & 1u) == 1u);
    let tx = tile_params.tile_origin_size.x;
    let ty = tile_params.tile_origin_size.y;
    let tile_id = (ty / tile_h) * ((img_w + tile_w - 1u) / tile_w) + (tx / tile_w);
    let stripe = (f32(tile_id & 7u) + 1.0) / 8.0;
    
    let diag_color = vec3<f32>(
      u * 0.7 + checker * 0.3,
      v * 0.7,
      stripe * 0.8
    );
    
    let pixel_coord = vec2<i32>(i32(lx), i32(ly));
    textureStore(out_tex, pixel_coord, vec4<f32>(diag_color, 1.0));
    return;
  }

  // P1 DIAGNOSTIC MODE: HDRI/procedural sky only (miss path verification)
  // Renders the environment for each pixel using the current camera without any
  // geometry intersection or lighting. Useful to validate HDRI binding,
  // rotation, exposure and texture writes, independent of BVH traversal.
  if (debug_mode == 3u) {
    // Build camera ray (center of pixel, no jitter) in GLOBAL coordinates
    let global_px = f32(gx) + 0.5;
    let global_py = f32(gy) + 0.5;
    let ndc_x = (global_px / f32(img_w)) * 2.0 - 1.0;
    let ndc_y = (1.0 - global_py / f32(img_h)) * 2.0 - 1.0;
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    var rd = normalize(
      uniforms.cam_forward +
      ndc_x * half_w * uniforms.cam_right +
      ndc_y * half_h * uniforms.cam_up
    );

    // Sample environment (HDRI when enabled, otherwise procedural sky)
    let sky_rgb = sample_sky(rd);

    // Optional tile border visualization (faint grid every 64px)
    let border = select(1.0, 0.85, ((lx % 64u == 0u) || (ly % 64u == 0u)));
    let pixel_coord = vec2<i32>(i32(lx), i32(ly));
    textureStore(out_tex, pixel_coord, vec4<f32>(sky_rgb * border, 1.0));
    return;
  }
  
  if (debug_mode == 2u) {
    let u = f32(gx) / f32(max(img_w, 1u));
    let v = f32(gy) / f32(max(img_h, 1u));

    // Split image into three vertical bands to highlight channel order
    let band = u * 3.0;
    var color = vec3<f32>(0.0);
    if (band < 1.0) {
      // Red band: horizontal ramp
      color = vec3<f32>(band, 0.0, 0.0);
    } else if (band < 2.0) {
      // Green band: vertical ramp
      color = vec3<f32>(0.0, v, 0.0);
    } else {
      // Blue band: inverse horizontal ramp
      color = vec3<f32>(0.0, 0.0, 1.0 - (band - 2.0));
    }

    // Draw tile borders to confirm copy/paste alignment
    let border = select(1.0, 0.15, ((lx % 64u == 0u) || (ly % 64u == 0u)));
    let pixel_coord = vec2<i32>(i32(lx), i32(ly));
    textureStore(out_tex, pixel_coord, vec4<f32>(color * border, 1.0));
    return;
  }

  // P1 DIAGNOSTIC MODE 4: XY gradient using GLOBAL coordinates
  // Pure gradient visualization to verify tile addressing and seamless writes
  if (debug_mode == 4u) {
    let u = f32(gx) / f32(max(img_w, 1u));
    let v = f32(gy) / f32(max(img_h, 1u));
    // Simple horizontal/vertical gradient with tile borders
    let border = select(1.0, 0.5, ((gx % 64u == 0u) || (gy % 64u == 0u)));
    let pixel_coord = vec2<i32>(i32(lx), i32(ly));
    textureStore(out_tex, pixel_coord, vec4<f32>(u * border, v * border, 0.5, 1.0));
    return;
  }

  let global_coord = vec2<u32>(gx, gy);
  let pixel_coord = vec2<i32>(i32(lx), i32(ly));  // LOCAL tile coordinates for texture writes

  // Seed per-pixel RNG using GLOBAL coordinates for consistency across tiles
  var st: u32 = uniforms.seed_hi ^ (gx * 1664525u) ^ (gy * 1013904223u) ^ uniforms.frame_index;
  
  // Accumulate multiple samples per pixel
  var final_color_accum = vec3<f32>(0.0);
  let spp_count = max(spp_batch, 1u);  // Safety: at least 1 sample
  
  // Initialize ray tracing results (used after SPP loop for AOVs)
  var t_best = 1e30;
  var hit_albedo = vec3<f32>(0.7, 0.7, 0.8);
  var hit_normal = vec3<f32>(0.0, 0.0, 1.0);
  var hit_point = vec3<f32>(0.0);
  var is_hit = false;
  var sample_final_color = vec3<f32>(0.0);
  var sample_base_albedo = vec3<f32>(0.0);
  var sample_direct_light = vec3<f32>(0.0);
  var sample_indirect_light = vec3<f32>(0.0);
  
  for (var sample_idx: u32 = 0u; sample_idx < spp_count; sample_idx = sample_idx + 1u) {
    // Get sample for pixel jitter using GLOBAL coordinates
    let sample_2d = get_sample_2d(global_coord, sample_idx, 0u, &st);
    let jx = (sample_2d.x - 0.5);  // [-0.5, 0.5] range
    let jy = (sample_2d.y - 0.5);

    // Generate camera ray using GLOBAL coordinates for seamless perspective
    let global_px = f32(gx) + 0.5 + jx;
    let global_py = f32(gy) + 0.5 + jy;
    let ndc_x = (global_px / f32(img_w)) * 2.0 - 1.0;
    let ndc_y = (1.0 - global_py / f32(img_h)) * 2.0 - 1.0;
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    // Generate ray in world space using camera basis, matching pt_kernel.wgsl convention
    var rd = normalize(
      uniforms.cam_forward +
      ndc_x * half_w * uniforms.cam_right +
      ndc_y * half_h * uniforms.cam_up
    );
    let ro = uniforms.cam_origin;

    // Create ray
    let camera_ray = Ray(ro, 1e-3, rd, 1e30);

    // Reset per-sample results
    t_best = 1e30;
    hit_albedo = vec3<f32>(0.7, 0.7, 0.8);
    hit_normal = vec3<f32>(0.0, 0.0, 1.0);
    var hit_material_type = 0u; // 0 = mesh, 1 = sphere, 2 = SDF
    hit_point = vec3<f32>(0.0);

  // Test legacy spheres first
  let sphere_count = arrayLength(&scene_spheres);
  for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
    let s = scene_spheres[i];
    let t = ray_sphere(ro, rd, s.center, s.radius);
    if (t < t_best) {
      t_best = t;
      hit_point = ro + rd * t;
      hit_normal = normalize(hit_point - s.center);
      hit_albedo = s.albedo;
      hit_material_type = 1u;
    }
  }

  // Test hybrid scene (SDF + mesh)
  let hybrid_hit = intersect_hybrid(camera_ray);
  if (hybrid_hit.hit != 0u && hybrid_hit.t < t_best) {
    t_best = hybrid_hit.t;
    hit_point = hybrid_hit.point;
    hit_normal = hybrid_hit.normal;
    hit_albedo = get_surface_properties(hybrid_hit);
    hit_material_type = hybrid_hit.hit_type + 2u; // 2 = mesh, 3 = SDF
  }

  // Update is_hit for last sample (used for AOVs)
  is_hit = t_best < 1e20;

  // Sky color for miss cases - use magenta marker for background pixels
  var sky_color = vec3<f32>(0.0);
  if (!is_hit) {
    sky_color = sample_sky(rd);
  }

  // Compute light direction from azimuth and elevation
  let az = uniforms.lighting_azimuth;
  let el = uniforms.lighting_elevation;
  let light_dir = normalize(vec3<f32>(
    cos(el) * sin(az),
    sin(el),
    cos(el) * cos(az)
  ));
  
  sample_final_color = sky_color;
  sample_base_albedo = clamp(hit_albedo, vec3<f32>(0.0), vec3<f32>(1.0));
  sample_direct_light = vec3<f32>(0.0);
  sample_indirect_light = vec3<f32>(0.0);

  if (is_hit) {
    let surf_albedo = clamp(hit_albedo, vec3<f32>(0.0), vec3<f32>(1.0));
    sample_base_albedo = surf_albedo;

    // Compute shadow factor with aggressive bias to completely eliminate shadow spikes
    // Shadow ray bias tuned for normalized mesh scale (~2.0 units total extent)
    let ndotl_s = max(dot(hit_normal, light_dir), 0.0);
    // Aggressive bias values to prevent self-shadowing on normalized terrain
    let bias_normal = 0.008;       // Strong normal offset for normalized coordinates
    let bias_light = 0.004;        // Substantial offset along light direction
    // Strong adaptive bias at grazing angles to prevent shadow acne
    let grazing_bias = 0.008 * (1.0 - ndotl_s);  // Much stronger at grazing angles
    // Combined bias: use both normal and light offsets with grazing adjustment
    let shadow_origin = hit_point + hit_normal * (bias_normal + grazing_bias) + light_dir * bias_light;
    let shadow_ray = Ray(shadow_origin, 0.001, light_dir, 1000.0);
    // Shadow test with backface culling to prevent self-shadowing
    let in_shadow = intersect_shadow_ray(shadow_ray, 1000.0);
    let shadow_factor = select(1.0, 1.0 - uniforms.shadow_intensity, in_shadow);

    // Compute lighting based on type
    var ndotl = 0.0;
    if (uniforms.lighting_type == 0u) {
      // Lambertian: simple diffuse
      ndotl = max(dot(hit_normal, light_dir), 0.0);
    } else {
      // Blinn-Phong: diffuse + specular
      ndotl = max(dot(hit_normal, light_dir), 0.0);
      let view_dir = normalize(-rd);
      let half_vec = normalize(light_dir + view_dir);
      let ndoth = max(dot(hit_normal, half_vec), 0.0);
      let specular = pow(ndoth, 32.0) * 0.1;  // Subtle specular
      ndotl = ndotl + specular;
    }

    // Apply hue-preserving overlay lighting for categorical colors
    // Narrow range (0.4 to 1.15) prevents harsh contrast and spikes
    // while maintaining proper 3D shading and color recognition
    
    // Compute shading value (0=shadow, 1=full light)
    let shading = ndotl * shadow_factor;
    
    // Proven values: ambient 0.4, max 1.15 for clean terrain rendering
    let light_min = 0.4;  // Higher ambient floor prevents crushed blacks
    let light_max = 1.15; // Gentle highlights avoid HDR spikes
    let light_factor = light_min + shading * uniforms.lighting_intensity * (light_max - light_min);
    
    // Apply lighting preserving hue and saturation
    let lit_color = surf_albedo * light_factor;
    
    // Clamp to reasonable range (no HDR needed with narrow lighting range)
    sample_final_color = clamp(lit_color, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Compute AOV values
    sample_direct_light = surf_albedo * (shading * uniforms.lighting_intensity);
    sample_indirect_light = surf_albedo * light_min; // Ambient component
  } else {
    // Miss case: use sky color (magenta marker for background pixels)
    sample_base_albedo = sky_color;
    sample_final_color = sky_color;
  }

    // Accumulate this sample's color
    final_color_accum += sample_final_color;
  }  // End SPP loop
  
  // Average accumulated samples
  let final_color_avg = final_color_accum / f32(spp_count);
  
  // Write final averaged output
  // CRITICAL FIX: Ensure we never write pure black (0,0,0) to distinguish from uninitialized texture
  // If final color is very dark, add a tiny offset to make shader writes visible
  var output_color = final_color_avg;
  let luminance = dot(output_color, vec3<f32>(0.2126, 0.7152, 0.0722));
  if (luminance < 0.001) {
    // Very dark or black - add tiny offset to prove shader executed
    output_color = max(output_color, vec3<f32>(0.001, 0.001, 0.001));
  }
  textureStore(out_tex, pixel_coord, vec4<f32>(output_color, 1.0));
  
  // Write AOVs if enabled (using last sample's data)
  if (aov_enabled(AOV_ALBEDO_BIT)) {
    let albedo_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(hit_albedo, 1.0), is_hit);
    textureStore(aov_albedo, pixel_coord, albedo_val);
  }

  if (aov_enabled(AOV_NORMAL_BIT)) {
    let normal_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(hit_normal, 1.0), is_hit);
    textureStore(aov_normal, pixel_coord, normal_val);
  }

  if (aov_enabled(AOV_DEPTH_BIT)) {
    let depth_val: f32 = select(bitcast<f32>(0x7fc00000u), t_best, is_hit); // qNaN for miss
    textureStore(aov_depth, pixel_coord, vec4<f32>(depth_val, 0.0, 0.0, 0.0));
  }

  if (aov_enabled(AOV_DIRECT_BIT)) {
    let direct_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(sample_direct_light, 1.0), is_hit);
    textureStore(aov_direct, pixel_coord, direct_val);
  }

  if (aov_enabled(AOV_INDIRECT_BIT)) {
    let indirect_val = select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(sample_indirect_light, 1.0), is_hit);
    textureStore(aov_indirect, pixel_coord, indirect_val);
  }

  if (aov_enabled(AOV_EMISSION_BIT)) {
    // No emission in this simple implementation
    textureStore(aov_emission, pixel_coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
  }

  if (aov_enabled(AOV_VISIBILITY_BIT)) {
    let visibility_val: f32 = select(0.0, 1.0, is_hit);
    textureStore(aov_visibility, pixel_coord, vec4<f32>(visibility_val, 0.0, 0.0, 1.0));
  }

  // Final output already written above after averaging
}