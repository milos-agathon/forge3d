// Terrain draping shader: DEM displacement + categorical land-cover texture
// Vertex stage samples height from DEM (R32F/R16F), fragment samples RGBA8 land-cover
// Shadow mapping with PCF for soft, controllable shadows

// Import PCF shadow utilities and HDRI sampling
#include "common/shadows_pcf.wgsl"
#include "common/hdri.wgsl"

// ============================================================================
// Bind Groups
// ============================================================================

// Group 0: Camera & globals (expanded to 336 bytes for shadow support)
struct Globals {
    view: mat4x4<f32>,           // 64 bytes (offset 0)
    proj: mat4x4<f32>,           // 64 bytes (offset 64)
    camera_pos: vec4<f32>,       // 16 bytes (offset 128) - xyz = position, w = unused
    
    // Packed parameters - all in vec4s for predictable alignment
    params0: vec4<f32>,          // 16 bytes (offset 144) - [z_dir, zscale, light_type_f32, light_elevation]
    params1: vec4<f32>,          // 16 bytes (offset 160) - [light_azimuth, light_intensity, ambient, shadow_intensity]
    params2: vec4<f32>,          // 16 bytes (offset 176) - [gamma, fov, shadow_bias, enable_shadows]
    params3: vec4<f32>,          // 16 bytes (offset 192) - [lighting_model, shininess, specular_strength, shadow_softness]
    params4: vec4<f32>,          // 16 bytes (offset 208) - [background_r, background_g, background_b, background_a]
    params5: vec4<f32>,          // 16 bytes (offset 224) - [shadow_map_res, tonemap_mode, gamma_correction, hdri_intensity]
    params6: vec4<f32>,          // 16 bytes (offset 240) - [hdri_rotation_rad, enable_hdri, _unused, _unused]
    
    // Shadow mapping
    light_view_proj: mat4x4<f32>, // 64 bytes (offset 256) - light space transform
    shadow_params: vec4<f32>,     // 16 bytes (offset 320) - Reserved for additional shadow parameters
}

@group(0) @binding(0) var<uniform> globals: Globals;

// Group 1: DEM height texture (R32Float or R16Float)
@group(1) @binding(0) var height_tex: texture_2d<f32>;
@group(1) @binding(1) var height_samp: sampler;  // Linear filtering

// Group 2: Land-cover texture (RGBA8) + UV transform + HDRI environment
struct UVTransform {
    scale: vec2<f32>,
    offset: vec2<f32>,
    y_flip: u32,
    _pad: u32,
}

@group(2) @binding(0) var landcover_tex: texture_2d<f32>;
@group(2) @binding(1) var landcover_samp: sampler;  // Nearest filtering for categorical
@group(2) @binding(2) var<uniform> uv_xform: UVTransform;
@group(2) @binding(3) var hdri_tex: texture_2d<f32>;  // HDRI environment map
@group(2) @binding(4) var hdri_samp: sampler;  // Linear filtering for HDRI

// Group 3: Shadow mapping
@group(3) @binding(0) var shadow_map: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexInput {
    @location(0) position: vec2<f32>,  // XZ plane position
    @location(1) uv: vec2<f32>,        // Texture coordinates
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) height: f32,
    @location(3) shadow_pos: vec4<f32>,  // Position in light clip space
}

// UV transformation function (from common/uv_transform.wgsl logic)
fn apply_uv_transform(uv: vec2<f32>, xform: UVTransform) -> vec2<f32> {
    let uv_y_adjusted = select(uv.y, 1.0 - uv.y, xform.y_flip != 0u);
    let uv_flipped = vec2<f32>(uv.x, uv_y_adjusted);
    return clamp(uv_flipped * xform.scale + xform.offset, vec2<f32>(0.0), vec2<f32>(1.0));
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Unpack parameters
    let z_dir = globals.params0.x;
    let zscale = globals.params0.y;
    
    // Transform UV coordinates
    let uv_transformed = apply_uv_transform(in.uv, uv_xform);
    
    // Sample height from DEM (bilinear filtering for smooth terrain)
    let height = textureSampleLevel(height_tex, height_samp, uv_transformed, 0.0).r;
    
    // Build world position: Y-up coordinate system [X, Y, Z]
    // X and Z from input position (terrain plane), Y from DEM height
    // Apply z_dir (+1.0 = outward/up, -1.0 = inward/down) and zscale
    let world_pos = vec3<f32>(
        in.position.x,
        height * zscale * z_dir,
        in.position.y
    );
    
    // Transform to clip space
    out.clip_position = globals.proj * globals.view * vec4<f32>(world_pos, 1.0);
    out.uv = uv_transformed;
    out.world_pos = world_pos;
    out.height = height;
    
    // Transform to light clip space for shadow mapping
    out.shadow_pos = globals.light_view_proj * vec4<f32>(world_pos, 1.0);
    
    return out;
}

// ============================================================================
// Fragment Shader
// ============================================================================

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    // Sample land-cover texture with nearest filtering to preserve categorical colors
    let landcover_color = textureSample(landcover_tex, landcover_samp, in.uv);
    
    // Unpack lighting parameters from packed vec4s
    let light_type = u32(globals.params0.z);
    let light_elevation = globals.params0.w;
    let light_azimuth = globals.params1.x;
    let light_intensity = globals.params1.y;
    let ambient = globals.params1.z;
    let shadow_intensity = globals.params1.w;
    
    let shadow_bias = globals.params2.z;
    let enable_shadows = globals.params2.w > 0.5;  // Boolean encoded as float
    
    let lighting_model = u32(globals.params3.x);
    let shininess = globals.params3.y;
    let specular_strength = globals.params3.z;
    let shadow_softness = globals.params3.w;
    
    // If light_type is 0 (none), return unlit color
    if (light_type == 0u) {
        out.color = landcover_color;
        return out;
    }
    
    // Compute surface normal (Y-up flat terrain approximation)
    let normal = vec3<f32>(0.0, 1.0, 0.0);
    
    // Compute light direction from elevation and azimuth
    let light_elev_rad = radians(light_elevation);
    let light_azim_rad = radians(light_azimuth);
    
    // Spherical to Cartesian: azimuth from +X clockwise (viewed from above), elevation from horizon
    let light_dir = normalize(vec3<f32>(
        cos(light_elev_rad) * cos(light_azim_rad),  // X
        sin(light_elev_rad),                          // Y (up)
        -cos(light_elev_rad) * sin(light_azim_rad)  // Z
    ));
    
    // Compute view direction from camera to fragment
    let view_dir = normalize(globals.camera_pos.xyz - in.world_pos);
    
    // HDRI environment lighting (if enabled)
    let hdri_rotation_rad = globals.params6.x;
    let enable_hdri = globals.params6.y > 0.5;
    let hdri_intensity = globals.params5.w;
    
    var hdri_ambient = vec3<f32>(ambient);  // Default to constant ambient
    if (enable_hdri) {
        // Sample HDRI using surface normal for diffuse IBL
        let hdri_color = sample_environment(normal, hdri_rotation_rad, hdri_tex, hdri_samp);
        hdri_ambient = hdri_color * hdri_intensity;
    }
    
    // Base lighting calculation
    var lit_color: vec3<f32>;
    
    if (lighting_model == 0u) {
        // Lambert diffuse (simple)
        let ndotl = max(dot(normal, light_dir), 0.0);
        let diffuse = ndotl * light_intensity;
        // Use HDRI ambient if available, otherwise constant ambient
        let ambient_term = select(ambient, luminance(hdri_ambient), enable_hdri);
        lit_color = landcover_color.rgb * (hdri_ambient + diffuse * (1.0 - ambient_term));
    } else if (lighting_model == 1u) {
        // Phong reflection model
        let N = normalize(normal);
        let L = normalize(light_dir);
        let V = normalize(view_dir);
        
        // Diffuse
        let ndotl = max(dot(N, L), 0.0);
        let diffuse = landcover_color.rgb * ndotl;
        
        // Specular (reflected ray)
        let R = reflect(-L, N);
        let rdotv = max(dot(R, V), 0.0);
        let spec_factor = pow(rdotv, shininess);
        let specular = vec3<f32>(spec_factor * specular_strength);
        
        // Combine with HDRI or constant ambient
        lit_color = (hdri_ambient * landcover_color.rgb + (diffuse + specular) * light_intensity);
    } else {
        // Blinn-Phong (default for lighting_model == 2 or other)
        let N = normalize(normal);
        let L = normalize(light_dir);
        let V = normalize(view_dir);
        
        // Diffuse
        let ndotl = max(dot(N, L), 0.0);
        let diffuse = landcover_color.rgb * ndotl;
        
        // Specular (half-vector)
        let H = normalize(L + V);
        let ndoth = max(dot(N, H), 0.0);
        let spec_factor = pow(ndoth, shininess);
        let specular = vec3<f32>(spec_factor * specular_strength);
        
        // Combine with HDRI or constant ambient
        lit_color = (hdri_ambient * landcover_color.rgb + (diffuse + specular) * light_intensity);
    }
    
    // Clamp to reasonable range to avoid over-brightening
    lit_color = clamp(lit_color, vec3<f32>(0.0), vec3<f32>(2.0));
    
    // Apply shadow mapping if enabled
    if (enable_shadows && light_type != 0u) {
        // Convert from clip space to NDC
        let shadow_pos_ndc = in.shadow_pos.xyz / in.shadow_pos.w;
        
        // Sample shadow map with PCF
        let shadow_factor = sample_shadow_pcf(
            shadow_map,
            shadow_sampler,
            shadow_pos_ndc,
            shadow_bias,
            shadow_softness
        );
        
        // Mix between shadowed and lit based on shadow_intensity
        // shadow_factor: 1.0 = fully lit, 0.0 = fully in shadow
        // shadow_intensity: 0.0 = no darkening, 1.0 = maximum darkening
        let shadow_amount = 1.0 - (shadow_intensity * (1.0 - shadow_factor));
        lit_color = lit_color * shadow_amount;
    }
    
    // Apply lighting to RGB, preserve alpha
    out.color = vec4<f32>(lit_color, landcover_color.a);
    return out;
}
