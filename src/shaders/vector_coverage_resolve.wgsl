// LIMES conflation-free resolve.
//
// Coverage is accumulated for every primitive in a layer before this pass.
// This kernel therefore composites each layer exactly ONCE per pixel. It never
// blends per triangle or per primitive, eliminating shared-edge hairline seams.
// Layers are traversed in fixed ascending order; later layers source-over
// earlier layers, matching existing vector submission order. Compositing stays
// in the pipeline's existing linear space. Output is premultiplied linear RGBA.

struct ResolveParams {
    extent_layers: vec4<u32>,
}

struct FloatWords {
    values: array<f32>,
}

struct Float4Words {
    values: array<vec4<f32>>,
}

@group(0) @binding(0) var<storage, read> coverage: FloatWords;
@group(0) @binding(1) var<storage, read> layer_colors: Float4Words;
@group(0) @binding(2) var<uniform> params: ResolveParams;
@group(0) @binding(3) var<storage, read_write> output_rgba: Float4Words;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params.extent_layers.x;
    let height = params.extent_layers.y;
    let layer_count = params.extent_layers.z;
    let pixel_count = params.extent_layers.w;
    if gid.x >= width || gid.y >= height {
        return;
    }
    let pixel = gid.y * width + gid.x;
    var accumulated = vec4<f32>(0.0);
    var layer = 0u;
    loop {
        if layer >= layer_count {
            break;
        }
        let color = layer_colors.values[layer];
        let layer_coverage = clamp(coverage.values[layer * pixel_count + pixel], 0.0, 1.0);
        let source_alpha = clamp(det_barrier(color.a * layer_coverage), 0.0, 1.0);
        let remaining = 1.0 - source_alpha;
        let destination_rgb = det_barrier3(accumulated.rgb * remaining);
        accumulated = vec4<f32>(
            det_fma3(color.rgb, vec3<f32>(source_alpha), destination_rgb),
            det_fma(accumulated.a, remaining, source_alpha)
        );
        layer += 1u;
    }
    output_rgba.values[pixel] = det_barrier4(accumulated);
}
