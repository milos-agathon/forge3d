// DUPLA opt-in absolute-coordinate render and measurement entries.

struct CameraPair {
    camera_dd: DDVec3,
    padding: vec2<f32>,
    raw_f32: vec4<f32>,
}

struct Globals {
    view_proj: mat4x4<f32>,
    frame_count: u32,
    width: u32,
    height: u32,
    render_frame: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<DDVec3>;
@group(0) @binding(1) var<storage, read> cameras: array<CameraPair>;
@group(0) @binding(2) var<storage, read_write> measurements: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> globals: Globals;

fn local_f32(value: DDVec3) -> vec3<f32> {
    return vec3<f32>(value.x.hi + value.x.lo, value.y.hi + value.y.lo, value.z.hi + value.z.lo);
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_dd(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let position = positions[vertex_index];
    let camera_dd = cameras[globals.render_frame].camera_dd;
    let residual = dd_sub_vec3(position, camera_dd);
    let local = local_f32(residual);
    return VertexOut(globals.view_proj * vec4<f32>(local, 1.0), vec3<f32>(0.1, 0.85, 0.35));
}

@vertex
fn vs_raw_f32(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let position = positions[vertex_index];
    let raw_position = vec3<f32>(position.x.hi, position.y.hi, position.z.hi);
    let local = raw_position - cameras[globals.render_frame].raw_f32.xyz;
    return VertexOut(globals.view_proj * vec4<f32>(local, 1.0), vec3<f32>(0.9, 0.15, 0.1));
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}

@compute @workgroup_size(256)
fn measure_jitter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let frame = gid.x;
    if (frame >= globals.frame_count) { return; }
    let position = positions[0];
    let camera = cameras[frame];
    let residual = dd_sub_vec3(position, camera.camera_dd);
    let dd_local = local_f32(residual);
    let raw_local = vec3<f32>(position.x.hi, position.y.hi, position.z.hi) - camera.raw_f32.xyz;
    let dd_clip = globals.view_proj * vec4<f32>(dd_local, 1.0);
    let raw_clip = globals.view_proj * vec4<f32>(raw_local, 1.0);
    let dd_screen_y = (1.0 - dd_clip.y / dd_clip.w) * 0.5 * f32(globals.height);
    let raw_screen_y = (1.0 - raw_clip.y / raw_clip.w) * 0.5 * f32(globals.height);
    measurements[frame] = vec2<f32>(dd_screen_y, raw_screen_y);
}
