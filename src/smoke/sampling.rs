pub fn sample_scalar(field: &[f32], dims: [usize; 3], p: [f32; 3]) -> f32 {
    let x = p[0].clamp(0.0, (dims[0] - 1) as f32);
    let y = p[1].clamp(0.0, (dims[1] - 1) as f32);
    let z = p[2].clamp(0.0, (dims[2] - 1) as f32);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(dims[0] - 1);
    let y1 = (y0 + 1).min(dims[1] - 1);
    let z1 = (z0 + 1).min(dims[2] - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let fz = z - z0 as f32;

    let c000 = field[index(dims, x0, y0, z0)];
    let c100 = field[index(dims, x1, y0, z0)];
    let c010 = field[index(dims, x0, y1, z0)];
    let c110 = field[index(dims, x1, y1, z0)];
    let c001 = field[index(dims, x0, y0, z1)];
    let c101 = field[index(dims, x1, y0, z1)];
    let c011 = field[index(dims, x0, y1, z1)];
    let c111 = field[index(dims, x1, y1, z1)];

    let c00 = lerp(c000, c100, fx);
    let c10 = lerp(c010, c110, fx);
    let c01 = lerp(c001, c101, fx);
    let c11 = lerp(c011, c111, fx);
    let c0 = lerp(c00, c10, fy);
    let c1 = lerp(c01, c11, fy);
    lerp(c0, c1, fz)
}

pub fn sample_vector(field: &[f32], dims: [usize; 3], p: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0; 3];
    for component in 0..3 {
        out[component] = sample_vector_component(field, dims, p, component);
    }
    out
}

pub fn sample_vector_component(
    field: &[f32],
    dims: [usize; 3],
    p: [f32; 3],
    component: usize,
) -> f32 {
    let x = p[0].clamp(0.0, (dims[0] - 1) as f32);
    let y = p[1].clamp(0.0, (dims[1] - 1) as f32);
    let z = p[2].clamp(0.0, (dims[2] - 1) as f32);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    let x1 = (x0 + 1).min(dims[0] - 1);
    let y1 = (y0 + 1).min(dims[1] - 1);
    let z1 = (z0 + 1).min(dims[2] - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let fz = z - z0 as f32;

    let read =
        |x: usize, y: usize, z: usize| -> f32 { field[index(dims, x, y, z) * 3 + component] };

    let c000 = read(x0, y0, z0);
    let c100 = read(x1, y0, z0);
    let c010 = read(x0, y1, z0);
    let c110 = read(x1, y1, z0);
    let c001 = read(x0, y0, z1);
    let c101 = read(x1, y0, z1);
    let c011 = read(x0, y1, z1);
    let c111 = read(x1, y1, z1);

    let c00 = lerp(c000, c100, fx);
    let c10 = lerp(c010, c110, fx);
    let c01 = lerp(c001, c101, fx);
    let c11 = lerp(c011, c111, fx);
    let c0 = lerp(c00, c10, fy);
    let c1 = lerp(c01, c11, fy);
    lerp(c0, c1, fz)
}

pub fn index(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    (z * dims[1] + y) * dims[0] + x
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1.0e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn hash01(mut value: u32) -> f32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}
