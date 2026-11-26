// Returns height, valid support, and whether this raster has a mask.
// Plain R32F supplies 0 in G. Masked RG32F supplies 0.5 (valid) or +1 (missing).
fn height_footprint(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, d: vec2<f32>, f: vec2<f32>) -> vec3<f32> {
    let flags = vec4<f32>(a.y, b.y, c.y, d.y);
    if (all(flags <= vec4<f32>(0.5))) {
        return vec3<f32>(det_mix(det_mix(a.x, b.x, f.x), det_mix(c.x, d.x, f.x), f.y), 1.0, a.y);
    }
    let taps = vec4<f32>((1.0-f.x)*(1.0-f.y), f.x*(1.0-f.y), (1.0-f.x)*f.y, f.x*f.y);
    let w = select(taps, vec4<f32>(0.0), flags > vec4<f32>(0.5));
    let support = det_dot4(w, vec4<f32>(1.0));
    // Unsupported geometry vertices still require a finite position. The
    // fragment and shadow passes discard the same unsupported footprint.
    if (support < 0.5) {
        let floor = min(min(a.x, b.x), min(c.x, d.x));
        return vec3<f32>(floor, support, 1.0);
    }
    return vec3<f32>(det_div(det_dot4(w, vec4<f32>(a.x,b.x,c.x,d.x)), max(support, 0.5)), support, 1.0);
}
