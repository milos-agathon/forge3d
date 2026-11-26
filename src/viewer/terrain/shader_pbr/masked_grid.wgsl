fn viewer_height_node(uv:vec2<f32>) -> vec2<i32> {
    let dims=vec2<i32>(textureDimensions(heightmap,0));
    return clamp(vec2<i32>(uv*vec2<f32>(dims)),vec2<i32>(0),dims-vec2<i32>(1));
}
fn viewer_height_axis(p:vec2<i32>, offset:vec2<i32>, c:f32) -> f32 {
    let last = vec2<i32>(textureDimensions(heightmap,0)) - vec2<i32>(1);
    let a = clamp(p-offset,vec2<i32>(0),last);
    let b = clamp(p+offset,vec2<i32>(0),last);
    let left=textureLoad(heightmap,a,0).rg;
    let right=textureLoad(heightmap,b,0).rg;
    let l=left.y<=0.5 && any(a!=p);
    let r=right.y<=0.5 && any(b!=p);
    if (l && r) { return (right.x-left.x)*0.5; }
    if (l) { return c-left.x; }
    if (r) { return right.x-c; }
    return 0.0;
}
fn viewer_masked_normal(p:vec2<i32>, center:f32, strength:f32) -> vec3<f32> {
    let last=vec2<i32>(textureDimensions(heightmap,0))-vec2<i32>(1);
    let grad=vec2<f32>(viewer_height_axis(p,vec2<i32>(1,0),center),viewer_height_axis(p,vec2<i32>(0,1),center));
    let slope=grad*strength*u.terrain_params.w*vec2<f32>(last)/u.render_span_xz;
    return normalize(vec3<f32>(-slope.x,1.0,-slope.y));
}
