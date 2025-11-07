// src/shaders/brdf_tile.wgsl
// P7-03: Simplified offscreen PBR shader for BRDF tile rendering
// Renders UV-sphere with direct BRDF evaluation for gallery generation
// RELEVANT FILES: src/offscreen/brdf_tile.rs, src/shaders/lighting.wgsl, src/shaders/brdf/dispatch.wgsl

// Milestone 0: Shader version stamp for CI diff tracking
// Milestone 1: Incremented to 2 for GGX donut fix and proper vector handling
// Milestone 2: Incremented to 3 for normalized Blinn-Phong with comparable mapping
// Milestone 3: Incremented to 4 for Disney Principled with IOR-based F0
// Milestone 7: Incremented to 5 for Clearcoat and Sheen extensions
const BRDF_SHADER_VERSION: u32 = 5u;

// Minimal inline BRDF constants to avoid include conflicts
// REQ-M2: SHADING STAGE (LINEAR) — all BRDF math stays in linear RGB
// sRGB/OETF happens only in the final write based on output_mode/debug flag
const PI: f32 = 3.141592653589793;
const INV_PI: f32 = 0.318309886;

fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn alpha_from_roughness(r: f32) -> f32 {
    let a = r * r;
    return max(1e-4, a);
}

fn G1_smith_height_ggx(nDotX: f32, roughness: f32) -> f32 {
    let nx = saturate(nDotX);
    let a = clamp(roughness, 1e-4, 1.0);
    let k = ((a + 1.0) * (a + 1.0)) * 0.125;
    return nx / (nx * (1.0 - k) + k);
}

fn ggx_ndf(NoH: f32, alpha: f32) -> f32 {
    let noh = saturate(NoH);
    let a2 = alpha * alpha;
    let noh2 = noh * noh;
    let d = noh2 * (a2 - 1.0) + 1.0;
    let denom = PI * d * d + 1e-8;
    return a2 / denom;
}

fn wi3_cosines(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>) -> vec3<f32> {
    let n = normalize(N);
    let v = normalize(V);
    let l = normalize(L);
    var h = v + l;
    if all(h == vec3<f32>(0.0)) {
        h = n;
    } else {
        h = normalize(h);
    }
    let NoV = saturate(dot(n, v));
    let NoL = saturate(dot(n, l));
    let NoH = saturate(dot(n, h));
    return vec3<f32>(NoV, NoL, NoH);
}

// sRGB helpers (piecewise exact curve)
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let a = 0.055;
    var outc = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 3; i = i + 1) {
        let x = c[i];
        outc[i] = select(
            x * 12.92,
            (1.0 + a) * pow(max(x, 0.0), 1.0 / 2.4) - a,
            x > 0.0031308
        );
    }
    return clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Small stamp to verify which branch rendered this pixel
fn apply_debug_stamp(uv: vec2<f32>, color_lin: vec3<f32>, stamp: vec3<f32>) -> vec3<f32> {
    // Draw a thin band at the bottom 5% of the sphere UV to avoid top-left label overlay
    if (uv.y < 0.05) {
        return stamp;
    }
    return color_lin;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let a = 0.055;
    var outc = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 3; i = i + 1) {
        let x = c[i];
        outc[i] = select(
            x / 12.92,
            pow((x + a) / (1.0 + a), 2.4),
            x > 0.04045
        );
    }
    return clamp(outc, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn burley_diffuse(base_color: vec3<f32>, n_dot_l: f32, n_dot_v: f32, v_dot_h: f32, roughness: f32) -> vec3<f32> {
    let r = clamp(roughness, 0.0, 1.0);
    let fd90 = 0.5 + 2.0 * r * v_dot_h * v_dot_h;
    let FL = pow(1.0 - n_dot_l, 5.0);
    let FV = pow(1.0 - n_dot_v, 5.0);
    let scatter = (1.0 + (fd90 - 1.0) * FL) * (1.0 + (fd90 - 1.0) * FV);
    return base_color * INV_PI * scatter;
}

fn energy_comp_factor(f0: vec3<f32>, roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    let gloss = 1.0 - r;
    let f_add = 0.04 * gloss + gloss * gloss * 0.5;
    let energy = clamp(f_add + 0.16 * r + 0.01, 0.0, 1.0);
    return clamp(1.0 - energy, 0.0, 1.0);
}

// BRDF model constants (matches lighting.wgsl and Rust BrdfModel enum)
const BRDF_LAMBERT: u32 = 0u;
const BRDF_PHONG: u32 = 1u;
const BRDF_COOK_TORRANCE_GGX: u32 = 4u;
const BRDF_DISNEY_PRINCIPLED: u32 = 6u;

// Minimal ShadingParamsGPU structure (matches lighting.wgsl)
struct ShadingParamsGPU {
    brdf: u32,
    metallic: f32,
    roughness: f32,
    sheen: f32,
    clearcoat: f32,
    subsurface: f32,
    anisotropy: f32,
    exposure: f32,    // Milestone 0: carry exposure (default 1.0 in tests)
    // M2: Output encoding selection (0=linear, 1=srgb)
    output_mode: u32,
    _pad_out0: u32,
    _pad_out1: u32,
    _pad_out2: u32,
}

// Camera and transform uniforms
struct Uniforms {
    model_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
}

// Material and lighting parameters
struct BrdfTileParams {
    light_dir: vec3<f32>,
    _pad0: f32,
    light_color: vec3<f32>,
    light_intensity: f32,
    camera_pos: vec3<f32>,
    _pad1: f32,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ndf_only: u32,  // Boolean: 1 = NDF-only mode, 0 = full BRDF
    g_only: u32,    // Milestone 0: 1 = output Smith G as grayscale
    dfg_only: u32,  // Milestone 0: 1 = output D*F*G (pre-division)
    spec_only: u32, // Milestone 0: 1 = output specular-only (Cook–Torrance)
    roughness_visualize: u32,  // Milestone 0: 1 = output vec3(r) for uniform validation
    f0: vec3<f32>,             // Milestone 0: explicitly provide F0
    _pad_f0: f32,
    // M4: Disney Principled BRDF extensions
    clearcoat: f32,              // Clearcoat layer intensity [0,1]
    clearcoat_roughness: f32,    // Clearcoat layer roughness [0,1]
    sheen: f32,                  // Sheen intensity [0,1] for fabric-like materials
    sheen_tint: f32,             // Sheen tint [0,1]: 0=white, 1=base color
    specular_tint: f32,          // Specular tint [0,1]: 0=achromatic, 1=base color tint
    // M2: Debug toggles
    debug_lambert_only: u32,   // 1 = lambert-only output (disable specular)
    debug_diffuse_only: u32,   // 1 = output physical diffuse term only
    debug_energy: u32,         // 1 = output kS/Kd diagnostics (packed R,G,B)
    debug_d: u32,              // 1 = output D only (grayscale)
    debug_g_dbg: u32,          // 1 = output correlated G only (grayscale)
    debug_spec_no_nl: u32,     // 1 = output spec without NL and without Li
    debug_angle_sweep: u32,    // 1 = override normal with sweep across uv.x and force V=L=+Z
    debug_angle_component: u32,// 0=spec,1=diffuse,2=combined
    debug_no_srgb: u32,        // 1 = bypass sRGB conversion at end
    debug_kind: u32,           // 0=full, 1=D-only, 2=G-only, 3=F-only
    _pad_debug_kind: vec3<u32>,
}
;

struct DebugPush {
    mode: u32,
    roughness: f32,
    _pad: vec2<f32>,
}

// Vertex input (matches TbnVertex from sphere.rs)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> params: BrdfTileParams;
@group(0) @binding(2) var<uniform> shading: ShadingParamsGPU;

// M1: Optional debug buffer for min/max N·L and N·V tracking
// Binding 3 is only used when debug_dot_products flag is enabled
@group(0) @binding(3) var<storage, read_write> debug_buffer: array<atomic<u32>, 4>;
// Layout: [0]=min_nl, [1]=max_nl, [2]=min_nv, [3]=max_nv
// Values are stored as atomicMin/Max of floatBitsToUint
@group(0) @binding(7) var<uniform> debug_push: DebugPush;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Transform to world space
    let world_pos = uniforms.model_matrix * vec4<f32>(input.position, 1.0);
    output.world_position = world_pos.xyz;
    
    // Transform to clip space
    output.clip_position = uniforms.projection_matrix * uniforms.view_matrix * world_pos;
    
    // Transform normal to world space (assuming uniform scale, no normal matrix needed)
    let world_normal = (uniforms.model_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    output.world_normal = normalize(world_normal);
    
    // Pass through UV
    output.uv = input.uv;
    
    return output;
}

// Milestone 1: Compute NDF (Normal Distribution Function) with correct GGX/GTR2 formula
// Matches the D term in Cook-Torrance BRDF
fn compute_ndf(normal: vec3<f32>, half_vec: vec3<f32>, roughness: f32) -> f32 {
    // Milestone 1: Single roughness convention with clamping
    let alpha = clamp(roughness * roughness, 1e-4, 1.0);
    let alpha2 = alpha * alpha;
    
    // Milestone 1: Use saturate for proper [0,1] clamping
    let n_dot_h = saturate(dot(normal, half_vec));
    let n_dot_h2 = n_dot_h * n_dot_h;
    
    let denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;
    let denom2 = denom * denom;
    
    // Milestone 1: Stable division guard
    if denom2 < 1e-6 {
        return 0.0;
    }
    
    return alpha2 / (PI * denom2);
}

// Inline BRDF implementations for main models

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// M1: Stable denominator for Cook–Torrance specular term
// den_spec = max(4 * NL * NV, 1e-4)
fn spec_den(n_dot_l: f32, n_dot_v: f32) -> f32 {
    return max(4.0 * n_dot_l * n_dot_v, 1e-4);
}

fn roughness_to_alpha(r: f32) -> f32 {
    let rr = max(r, 0.001);
    return rr * rr;
}

fn D_ggx(alpha: f32, n_dot_h: f32) -> f32 {
    let a2 = alpha * alpha;
    let nh = clamp(n_dot_h, 0.0, 1.0);
    let cos2 = nh * nh;
    let denom = cos2 * (a2 - 1.0) + 1.0;
    let denom2 = denom * denom;
    if denom2 < 1e-6 {
        return 0.0;
    }
    return a2 / (PI * denom2);
}

fn ndf_ggx(NdotH: f32, a2: f32) -> f32 {
    let nh = clamp(NdotH, 0.0, 1.0);
    let denom = (nh * nh) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-7);
}

// Milestone 1: GGX/GTR2 Normal Distribution Function with correct formula
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = clamp(roughness * roughness, 1e-4, 1.0);
    return D_ggx(alpha, n_dot_h);
}

fn G1_schlick(alpha: f32, n_dot_x: f32) -> f32 {
    let ndx = clamp(n_dot_x, 0.0, 1.0);
    let k = ((alpha + 1.0) * (alpha + 1.0)) / 8.0;
    let denom = ndx * (1.0 - k) + k;
    if denom < 1e-6 {
        return 0.0;
    }
    return ndx / denom;
}

fn G_schlick(alpha: f32, n_dot_l: f32, n_dot_v: f32) -> f32 {
    return G1_schlick(alpha, n_dot_l) * G1_schlick(alpha, n_dot_v);
}

// Helper: Heitz lambda(NdotX) for GGX
fn lambda_term(n_dot_x: f32, a2: f32) -> f32 {
    let ndx = clamp(n_dot_x, 1e-4, 1.0);
    let ndx2 = ndx * ndx;
    let t2 = (1.0 - ndx2) / ndx2;
    return 0.5 * (sqrt(1.0 + a2 * t2) - 1.0);
}

// M2: Heitz correlated Smith G (physically-based, correlated)
// Reference: "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs" (Heitz 2014)
fn geometry_smith_correlated(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    // Use alpha = roughness (Heitz parametrization), not roughness^2
    let a  = clamp(roughness, 1e-4, 1.0);
    let a2 = a * a;
    let lambda_v = lambda_term(n_dot_v, a2);
    let lambda_l = lambda_term(n_dot_l, a2);
    let g = 1.0 / (1.0 + lambda_v + lambda_l);
    return clamp(g, 0.0, 1.0);
}

// M7: Smith G for GGX (Schlick-GGX form, used for clearcoat)
fn geometry_smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = clamp(roughness, 1e-4, 1.0);
    let k = ((a + 1.0) * (a + 1.0)) / 8.0;
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return g1_v * g1_l;
}

// M7: Charlie NDF for sheen (Disney's form)
// D_charlie(α, θh) = ((2 + 1/α) * pow(cos(θh), 1/α)) / (2π)
fn distribution_charlie(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = max(roughness, 1e-3);
    let inv_alpha = 1.0 / alpha;
    let cos_theta_h = saturate(n_dot_h);
    let cos_power = pow(cos_theta_h, inv_alpha);
    let factor = (2.0 + inv_alpha) / (2.0 * PI);
    return factor * cos_power;
}

// M7: Scalar Fresnel for clearcoat (IOR=1.5 → F0≈0.04)
fn fresnel_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Lambert BRDF (diffuse)
fn brdf_lambert(base_color: vec3<f32>) -> vec3<f32> {
    return base_color * INV_PI;
}

// Milestone 2: Normalized Blinn-Phong BRDF
fn brdf_phong(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, roughness: f32) -> vec3<f32> {
    // Milestone 2: Normalize all input vectors
    let n = normalize(normal);
    let v = normalize(view);
    let l = normalize(light);
    
    // Milestone 2: Guard against degenerate half vector
    let v_plus_l = v + l;
    let v_plus_l_len = length(v_plus_l);
    if v_plus_l_len < 1e-6 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v_plus_l);
    
    // Milestone 2: Use saturate for proper [0,1] clamping
    let n_dot_h = saturate(dot(n, h));
    let n_dot_v = saturate(dot(n, v));
    let n_dot_l = saturate(dot(n, l));
    let v_dot_h = saturate(dot(v, h));
    
    // Early exit if surface not visible
    if n_dot_v < 1e-6 || n_dot_l < 1e-6 {
        return vec3<f32>(0.0);
    }
    
    // Milestone 2: Roughness → exponent mapping
    // Start from same alpha = r^2 as GGX for consistency
    let alpha = clamp(roughness * roughness, 1e-4, 1.0);
    let alpha2 = alpha * alpha;
    
    // Map to Phong exponent: s = max(1.0, 2.0/(alpha*alpha) - 2.0)
    let s = max(1.0, 2.0 / alpha2 - 2.0);
    
    // Milestone 2: Normalized Blinn-Phong NDF
    // Dp = (s + 2.0) / (2π) * (N·H)^s
    let INV_2PI = 1.0 / (2.0 * PI);
    let Dp = (s + 2.0) * INV_2PI * pow(n_dot_h, s);
    
    // Milestone 2: Use Schlick Fresnel (same as GGX for fair comparison)
    let dielectric_f0 = vec3<f32>(0.04);
    let F = fresnel_schlick(v_dot_h, dielectric_f0);
    
    // Milestone 0: Energy scaling to prevent clipping (peak < 0.95)
    // Phong produces very bright highlights at low roughness, so scale more aggressively
    let energy_scale = 0.5;
    let specular = Dp * F * energy_scale;
    
    // Diffuse component with energy conservation
    let kD = vec3<f32>(1.0) - F;
    let diffuse = kD * base_color * INV_PI;
    
    let result = diffuse + specular;
    
    // Milestone 2: NaN check
    if any(result != result) {
        return vec3<f32>(0.0);
    }
    
    return result;
}

// Milestone 1: Cook-Torrance GGX BRDF with proper vector handling and sanitation
fn brdf_ggx(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    // Milestone 1: Normalize all input vectors for vector integrity
    let n = normalize(normal);
    let v = normalize(view);
    let l = normalize(light);
    
    // Milestone 1: Guard against degenerate half vector (v + l near zero)
    let v_plus_l = v + l;
    let v_plus_l_len = length(v_plus_l);
    if v_plus_l_len < 1e-6 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v_plus_l);
    
    // Milestone 1: Use saturate to clamp all dot products to [0,1]
    let n_dot_v = saturate(dot(n, v));
    let n_dot_l = saturate(dot(n, l));
    let n_dot_h = saturate(dot(n, h));
    let v_dot_h = saturate(dot(v, h));
    
    // Early exit if surface not visible from view or light
    if n_dot_v < 1e-6 || n_dot_l < 1e-6 {
        return vec3<f32>(0.0);
    }
    
    // Calculate F0 (surface reflection at zero incidence)
    let dielectric_f0 = vec3<f32>(0.04);
    let f0 = mix(dielectric_f0, base_color, metallic);
    
    // Cook-Torrance BRDF components
    let D = distribution_ggx(n_dot_h, roughness);
    let F = fresnel_schlick(v_dot_h, f0);
    let G = geometry_smith_correlated(n_dot_v, n_dot_l, roughness);
    
    // Milestone 2: Proper denominator with max guard (spec = D*F*G / max(4*nl*nv, 1e-4))
    let numerator = D * F * G;
    let denominator = spec_den(n_dot_l, n_dot_v);
    var specular = numerator / denominator;
    
    // Milestone 1: Sanitize NaN/Inf - replace non-finite with zero
    // NaN check: if value != itself, it's NaN
    if any(specular != specular) {
        specular = vec3<f32>(0.0);
    }
    // Numerical safety only
    specular = max(specular, vec3<f32>(0.0));
    
    // Diffuse component (energy conservation)
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diffuse = kD * base_color * INV_PI;
    
    let result = diffuse + specular;
    
    // Milestone 1: Final NaN check on output
    if any(result != result) {
        return vec3<f32>(0.0);
    }
    
    return result;
}

// Milestone 3: Disney Principled BRDF (basic dielectric path)
// M7: Extended with Clearcoat and Sheen
fn brdf_disney(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, base_color: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    // Milestone 3: Normalize all input vectors
    let n = normalize(normal);
    let v = normalize(view);
    let l = normalize(light);
    
    // Milestone 3: Guard against degenerate half vector
    let v_plus_l = v + l;
    let v_plus_l_len = length(v_plus_l);
    if v_plus_l_len < 1e-6 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v_plus_l);
    
    // Milestone 3: Use saturate for proper [0,1] clamping
    let n_dot_v = saturate(dot(n, v));
    let n_dot_l = saturate(dot(n, l));
    let n_dot_h = saturate(dot(n, h));
    let v_dot_h = saturate(dot(v, h));
    let l_dot_h = saturate(dot(l, h));
    
    // Early exit if surface not visible
    if n_dot_v < 1e-6 || n_dot_l < 1e-6 {
        return vec3<f32>(0.0);
    }
    
    // Milestone 3: F0 from IOR (default IOR=1.5 → F0=0.04)
    // F0 = ((ior - 1) / (ior + 1))^2
    // For ior=1.5: F0 = ((1.5-1)/(1.5+1))^2 = (0.5/2.5)^2 = 0.04
    let ior = 1.5;
    let f0_dielectric_scalar = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
    let f0_dielectric = vec3<f32>(f0_dielectric_scalar);
    let f0 = mix(f0_dielectric, base_color, metallic);
    
    // Milestone 3: Disney uses same GGX specular as Milestone 1
    // Same alpha convention: alpha = clamp(r^2, 1e-4, 1.0)
    let D = distribution_ggx(n_dot_h, roughness);
    let F = fresnel_schlick(v_dot_h, f0);
    let G = geometry_smith_correlated(n_dot_v, n_dot_l, roughness);
    
    let numerator = D * F * G;
    let denominator = spec_den(n_dot_l, n_dot_v);
    var specular = numerator / denominator;
    
    if any(specular != specular) {
        specular = vec3<f32>(0.0);
    }
    specular = clamp(specular, vec3<f32>(0.0), vec3<f32>(10.0));
    
    let k_comp = energy_comp_factor(f0, roughness);
    specular = specular * k_comp;
    
    let diffuse = burley_diffuse(base_color, n_dot_l, n_dot_v, v_dot_h, roughness) * (vec3<f32>(1.0) - F) * (1.0 - metallic);
    
    var result = diffuse + specular;
    
    // M7: Clearcoat layer (secondary GGX lobe with fixed IOR=1.5, narrow roughness)
    let clearcoat_strength = params.clearcoat;
    if clearcoat_strength > 1e-4 {
        // Clearcoat roughness: clamp to [0.03, 0.2] range
        let r_clearcoat = clamp(params.clearcoat_roughness, 0.03, 0.2);
        let alpha_clearcoat = r_clearcoat * r_clearcoat;
        
        // Clearcoat uses separate half vector (same as base)
        let D_clearcoat = distribution_ggx(n_dot_h, r_clearcoat);
        let F0_clearcoat = 0.04; // Fixed IOR=1.5
        let F_clearcoat = fresnel_schlick_scalar(v_dot_h, F0_clearcoat);
        let G_clearcoat = geometry_smith_ggx(n_dot_v, n_dot_l, r_clearcoat);
        
        let num_clearcoat = D_clearcoat * F_clearcoat * G_clearcoat;
        let den_clearcoat = spec_den(n_dot_l, n_dot_v);
        var clearcoat_spec = num_clearcoat / den_clearcoat;
        
        // Energy-conserving mix: coat over base
        // Lo = coat * mix + base * (1 - mix)
        // where mix = clearcoat_strength * F_clearcoat
        let coat_mix = clearcoat_strength * F_clearcoat;
        let coat_contrib = vec3<f32>(clearcoat_spec) * coat_mix;
        result = result * (1.0 - coat_mix) + coat_contrib;
    }
    
    // M7: Sheen layer (Charlie NDF for grazing retro-reflection)
    let sheen_strength = params.sheen;
    if sheen_strength > 1e-4 {
        let D_sheen = distribution_charlie(n_dot_h, roughness);
        let sheen_tint = params.sheen_tint;
        let sheen_color = mix(vec3<f32>(1.0), base_color, sheen_tint);
        
        // Sheen is a grazing-angle effect, typically applied at grazing angles
        // Use a simple grazing factor: (1 - n_dot_v)^5
        let grazing_factor = pow(1.0 - n_dot_v, 5.0);
        let sheen_contrib = sheen_color * D_sheen * sheen_strength * grazing_factor * INV_PI;
        result = result + sheen_contrib;
    }
    
    // Milestone 3: Final NaN check
    if any(result != result) {
        return vec3<f32>(0.0);
    }
    
    return result;
}

fn finalize_output_linear(color_lin: vec3<f32>) -> vec4<f32> {
    // debug_no_srgb overrides and forces linear write
    if params.debug_no_srgb != 0u {
        return vec4<f32>(color_lin, 1.0);
    }
    // shading.output_mode selects 0=linear, 1=srgb
    if shading.output_mode == 1u {
        return vec4<f32>(linear_to_srgb(color_lin), 1.0);
    }
    return vec4<f32>(color_lin, 1.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var normal = normalize(input.world_normal);
    var view_dir = normalize(params.camera_pos - input.world_position);
    var light_dir = normalize(params.light_dir);
    
    // Milestone 0: Debug toggles (check in priority order)

    // Angle sweep override: replace normal, view, light
    if params.debug_angle_sweep != 0u {
        let nx = mix(0.0, 0.99, input.uv.x);
        normal = normalize(vec3<f32>(nx, 0.0, 1.0));
        view_dir = vec3<f32>(0.0, 0.0, 1.0);
        light_dir = vec3<f32>(0.0, 0.0, 1.0);
        // Recompute dot products after override
        let nl = dot(normal, light_dir);
        let nv = dot(normal, view_dir);
        // clamp to [0,1]
        _ = 0; // no-op to preserve structure; WGSL requires statements
        // overwrite locals
        // Note: reusing the same names; redeclare is not allowed, so recompute now
        // We will just shadow earlier calculations by reassigning below
    }
    // Derive N·L/N·V after potential override
    let NoL = saturate(dot(normal, light_dir));
    let NoV = saturate(dot(normal, view_dir));
    var h = view_dir + light_dir;
    if all(h == vec3<f32>(0.0)) {
        h = normal;
    } else {
        h = normalize(h);
    }
    let NoH = saturate(dot(normal, h));
    let VoH = saturate(dot(view_dir, h));
    var debug_d = 0.0;
    var debug_g = 0.0;
    var debug_f = vec3<f32>(0.0);
    if NoL > 0.0 && NoV > 0.0 {
        debug_d = distribution_ggx(NoH, shading.roughness);
        debug_g = geometry_smith_correlated(NoV, NoL, shading.roughness);
        let dielectric_f0 = vec3<f32>(0.04);
        let f0_mix = mix(dielectric_f0, params.base_color, shading.metallic);
        debug_f = fresnel_schlick(VoH, f0_mix);
    }

    if debug_push.mode != 0u {
        let cosines = wi3_cosines(normal, view_dir, light_dir);
        let dbgNoV = cosines.x;
        let dbgNoL = cosines.y;
        let dbgNoH = cosines.z;
        let alpha = alpha_from_roughness(debug_push.roughness);

        if debug_push.mode == 1u {
            var preview = 0.0;
            if dbgNoL > 0.0 && dbgNoV > 0.0 {
                let D = ggx_ndf(dbgNoH, alpha);
                let D_vis = saturate(D * 0.35);
                preview = D_vis;
            }
            return vec4<f32>(preview, preview, preview, 1.0);
        }

        if debug_push.mode == 2u {
            var vis = 0.0;
            if dbgNoL > 0.0 && dbgNoV > 0.0 {
                vis = geometry_smith_correlated(dbgNoV, dbgNoL, debug_push.roughness);
            }
            return vec4<f32>(vis, vis, vis, 1.0);
        }

        if debug_push.mode == 3u {
            if dbgNoL <= 0.0 || dbgNoV <= 0.0 {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            var dbg_h = view_dir + light_dir;
            if all(dbg_h == vec3<f32>(0.0)) {
                dbg_h = normal;
            } else {
                dbg_h = normalize(dbg_h);
            }
            let dbgNoH_spec = saturate(dot(normal, dbg_h));
            let dbgVoH = saturate(dot(view_dir, dbg_h));
            let D = distribution_ggx(dbgNoH_spec, debug_push.roughness);
            let F = fresnel_schlick(dbgVoH, params.f0);
            let G = geometry_smith_correlated(dbgNoV, dbgNoL, debug_push.roughness);
            let numerator = D * F * G;
            let denominator = spec_den(dbgNoL, dbgNoV);
            var specular = numerator / denominator;
            let radiance = params.light_color * params.light_intensity;
            let final_color = specular * radiance * dbgNoL * shading.exposure;
            return vec4<f32>(final_color, 1.0);
        }
    }

    let n_dot_l = NoL;
    let n_dot_v = NoV;
    let n_dot_h = NoH;

    // M1: Track min/max N·L and N·V for debug validation (after override)
    let nl_u32: u32 = u32(clamp(n_dot_l, 0.0, 1.0) * 4294967295.0);
    let nv_u32: u32 = u32(clamp(n_dot_v, 0.0, 1.0) * 4294967295.0);
    atomicMin(&debug_buffer[0], nl_u32);
    atomicMax(&debug_buffer[1], nl_u32);
    atomicMin(&debug_buffer[2], nv_u32);
    atomicMax(&debug_buffer[3], nv_u32);
    
    // 1. Roughness visualize: output vec3(r) to validate uniform flow
    if params.roughness_visualize != 0u {
        let r = params.roughness;
        return vec4<f32>(r, r, r, 1.0);
    }
    
    // Lambert-only path (T3): diffuse only, no specular
    if params.debug_lambert_only != 0u {
        // Standard Lambert diffuse
        let diffuse = params.base_color * INV_PI;
        let radiance = params.light_color * params.light_intensity;
        let final_color = diffuse * radiance * n_dot_l * shading.exposure;
        return finalize_output_linear(final_color);
    }

    if params.debug_diffuse_only != 0u {
        let n = normal;
        let v = view_dir;
        let l = light_dir;
        var h = v + l;
        if all(h == vec3<f32>(0.0)) {
            h = n;
        } else {
            h = normalize(h);
        }

        let n_dot_v = saturate(dot(n, v));
        let n_dot_l = saturate(dot(n, l));
        let v_dot_h = saturate(dot(v, h));

        if (n_dot_v < 1e-6 || n_dot_l < 1e-6) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }

        let dielectric_f0 = vec3<f32>(0.04);
        let f0 = mix(dielectric_f0, params.base_color, params.metallic);

        var diffuse = vec3<f32>(0.0);
        if shading.brdf == BRDF_DISNEY_PRINCIPLED {
            let F = fresnel_schlick(v_dot_h, f0);
            let burley = burley_diffuse(params.base_color, n_dot_l, n_dot_v, v_dot_h, shading.roughness);
            diffuse = burley * (vec3<f32>(1.0) - F) * (1.0 - shading.metallic);
        } else {
            let F = fresnel_schlick(v_dot_h, f0);
            let kD = (vec3<f32>(1.0) - F) * (1.0 - shading.metallic);
            diffuse = kD * params.base_color * INV_PI;
        }

        let radiance = params.light_color * params.light_intensity;
        let final_color = diffuse * radiance * n_dot_l * shading.exposure;
        return finalize_output_linear(final_color);
    }

    // 2. G-only: output Smith G as grayscale
    if params.g_only != 0u {
        // === DEBUG: G-ONLY BEGIN (WI-3) ===
        let n = normal;
        let l = light_dir;
        let v = view_dir;

        let NoL_dbg = saturate(dot(n, l));
        let NoV_dbg = saturate(dot(n, v));
        if (NoL_dbg <= 0.0 || NoV_dbg <= 0.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }

        let G_dbg = geometry_smith_correlated(NoV_dbg, NoL_dbg, params.roughness);
        // === DEBUG: G-ONLY END (WI-3) ===
        return vec4<f32>(G_dbg, G_dbg, G_dbg, 1.0);
    }

    // 2b. D-only: pure GGX NDF visualization (normalized, hemisphere masked)
    if params.debug_d != 0u {
        // === DEBUG: D-ONLY BEGIN (WI-3) ===
        {
            let n = normal;
            let l = light_dir;
            let v = view_dir;

            let NoL_dbg = saturate(dot(n, l));
            let NoV_dbg = saturate(dot(n, v));
            if (NoL_dbg <= 0.0 || NoV_dbg <= 0.0) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }

            var h_dbg = l + v;
            if all(h_dbg == vec3<f32>(0.0)) {
                h_dbg = n;
            } else {
                h_dbg = normalize(h_dbg);
            }
            let NoH_dbg = saturate(dot(n, h_dbg));

            let alpha_dbg = max(1e-4, params.roughness * params.roughness);
            let a2_dbg = alpha_dbg * alpha_dbg;
            let d_dbg = (NoH_dbg * NoH_dbg) * (a2_dbg - 1.0) + 1.0;
            let D_dbg = a2_dbg / (PI * d_dbg * d_dbg);

            let D_vis = saturate(D_dbg * 0.35);
            return vec4<f32>(D_vis, D_vis, D_vis, 1.0);
        }
        // === DEBUG: D-ONLY END (WI-3) ===
    }
    
    // 3. DFG-only: output normalized D*F*G/(4*nl*nv) energy core
    if params.dfg_only != 0u {
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        let half_vec = normalize(view_dir + light_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let v_dot_h = max(dot(view_dir, half_vec), 0.0);
        
        // Milestone 0: Use uniform-provided F0
        let f0 = params.f0;
        
        let D = distribution_ggx(n_dot_h, params.roughness);
        let F = fresnel_schlick(v_dot_h, f0);
        let G = geometry_smith_correlated(n_dot_v, n_dot_l, params.roughness);
        
        // M2.2: Compute specular BRDF term (D*F*G)/(4*nl*nv)
        let numerator = D * F * G;
        let denominator = spec_den(n_dot_l, n_dot_v);
        let specular_term = numerator / denominator;
        
        // M2.2: Normalize by D_max (since G_max=1 at nl=nv=1)
        // D_max = 1 / (PI * alpha^2)
        let alpha = clamp(params.roughness * params.roughness, 1e-4, 1.0);
        let alpha2 = alpha * alpha;
        let D_max = 1.0 / (PI * alpha2);
        
        // DG_norm = clamp(specular_term / D_max, 0, 1)
        let dfg_norm = clamp(specular_term / D_max, vec3<f32>(0.0), vec3<f32>(1.0));
        
        return finalize_output_linear(dfg_norm);
    }
    
    // 4. SPEC-only: output specular BRDF term only (Cook–Torrance)
    if params.spec_only != 0u {
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        let half_vec = normalize(view_dir + light_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let v_dot_h = max(dot(view_dir, half_vec), 0.0);
        let f0 = params.f0;
        let D = distribution_ggx(n_dot_h, params.roughness);
        let F = fresnel_schlick(v_dot_h, f0);
        let G = geometry_smith_correlated(n_dot_v, n_dot_l, params.roughness);
        let numerator = D * F * G;
        let denominator = spec_den(n_dot_l, n_dot_v);
        var specular = numerator / denominator;
        if params.debug_spec_no_nl != 0u {
            let rgb = apply_debug_stamp(input.uv, specular, vec3<f32>(0.0, 0.0, 1.0));
            return finalize_output_linear(rgb);
        }
        let radiance = params.light_color * params.light_intensity;
        let final_color = specular * radiance * n_dot_l * shading.exposure;
        let rgb = apply_debug_stamp(input.uv, final_color, vec3<f32>(0.0, 0.0, 1.0));
        return finalize_output_linear(rgb);
    }
    
    // 5. NDF-only debug mode: output normalized D for shape visualization
    if params.ndf_only != 0u {
        let half_vec = normalize(view_dir + light_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        
        // M2.1: Compute GGX NDF
        let D = distribution_ggx(n_dot_h, params.roughness);
        
        // M2.1: Roughness-invariant normalization
        // At peak (n_dot_h=1): D_peak = α²/π, so normalize by D_max = π/α²
        // D_norm = D * (π/α²) brings peak to 1.0 for all roughness values
        let alpha = clamp(params.roughness * params.roughness, 1e-4, 1.0);
        let alpha2 = alpha * alpha;
        let D_norm = clamp(D * PI / alpha2, 0.0, 1.0);
        
        return finalize_output_linear(vec3<f32>(D_norm));
    }

    // Energy debug: Principled mode visualizes compensation factor, GGX shows kS/kD
    if params.debug_energy != 0u {
        if shading.brdf == BRDF_DISNEY_PRINCIPLED {
            let dielectric_f0 = vec3<f32>(0.04);
            let f0 = mix(dielectric_f0, params.base_color, params.metallic);
            let comp = energy_comp_factor(f0, shading.roughness);
            return finalize_output_linear(vec3<f32>(comp));
        }
        let half_vec = normalize(view_dir + light_dir);
        let v_dot_h = max(dot(view_dir, half_vec), 0.0);
        let dielectric_f0 = vec3<f32>(0.04);
        let f0 = mix(dielectric_f0, params.base_color, params.metallic);
        let F = fresnel_schlick(v_dot_h, f0);
        let kS = F;
        let kD = (vec3<f32>(1.0) - kS) * (1.0 - params.metallic);
        let ks_r = clamp(kS.x, 0.0, 1.0);
        let kd_g = clamp(kD.x, 0.0, 1.0);
        let sum_b = clamp(ks_r + kd_g, 0.0, 1.0);
        return finalize_output_linear(vec3<f32>(ks_r, kd_g, sum_b));
    }
    
    // Full BRDF evaluation
    if n_dot_l <= 0.0 {
        // No lighting from this direction
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    // Dispatch to appropriate BRDF based on model index
    var brdf_color: vec3<f32>;
    
    if shading.brdf == BRDF_LAMBERT {
        brdf_color = brdf_lambert(params.base_color);
    } else if shading.brdf == BRDF_PHONG {
        brdf_color = brdf_phong(normal, view_dir, light_dir, params.base_color, shading.roughness);
    } else if shading.brdf == BRDF_COOK_TORRANCE_GGX {
        brdf_color = brdf_ggx(normal, view_dir, light_dir, params.base_color, shading.metallic, shading.roughness);
    } else if shading.brdf == BRDF_DISNEY_PRINCIPLED {
        // Milestone 3: Use proper Disney Principled BRDF
        brdf_color = brdf_disney(normal, view_dir, light_dir, params.base_color, shading.metallic, shading.roughness);
    } else {
        // Default to Lambert for unknown models
        brdf_color = brdf_lambert(params.base_color);
    }
    
    // Apply lighting (no tone mapping). exposure is carried but defaults to 1.0 in tests.
    let radiance = params.light_color * params.light_intensity;
    let final_color = brdf_color * radiance * n_dot_l * shading.exposure;

    if params.debug_kind != 0u {
        var debug_rgb = final_color;
        switch params.debug_kind {
            case 1u: {
                debug_rgb = vec3<f32>(debug_d);
            }
            case 2u: {
                debug_rgb = vec3<f32>(debug_g);
            }
            case 3u: {
                debug_rgb = debug_f;
            }
            default: {
                debug_rgb = final_color;
            }
        }
        return vec4<f32>(debug_rgb, 1.0);
    }

    // Angle sweep components: override output if requested
    if params.debug_angle_sweep != 0u {
        if params.debug_angle_component == 0u {
            // spec only
            let half_vec = normalize(view_dir + light_dir);
            let n_dot_h = max(dot(normal, half_vec), 0.0);
            let v_dot_h = max(dot(view_dir, half_vec), 0.0);
            let D = distribution_ggx(n_dot_h, shading.roughness);
            let F = fresnel_schlick(v_dot_h, vec3<f32>(0.04));
            let G = geometry_smith_correlated(n_dot_v, n_dot_l, shading.roughness);
            let specular = (D * F * G) / spec_den(n_dot_l, n_dot_v);
            return finalize_output_linear(specular);
        } else if params.debug_angle_component == 1u {
            // diffuse only
            let F = fresnel_schlick(n_dot_v, vec3<f32>(0.04));
            let kD = (vec3<f32>(1.0) - F) * (1.0 - shading.metallic);
            let diffuse = kD * params.base_color * INV_PI;
            return finalize_output_linear(diffuse);
        }
        // combined
        return finalize_output_linear(final_color);
    }

    return finalize_output_linear(final_color);
}
