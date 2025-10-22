// Correlated Multi-Jittered (CMJ) sampling
// Provides better 2D stratification than both jittered and multi-jittered sampling
// Based on Kensler 2013: "Correlated Multi-Jittered Sampling"

// Permute function for CMJ
fn cmj_permute(i: u32, l: u32, p: u32) -> u32 {
    var w = l - 1u;
    w = w | (w >> 1u);
    w = w | (w >> 2u);
    w = w | (w >> 4u);
    w = w | (w >> 8u);
    w = w | (w >> 16u);
    
    var ii = i;
    loop {
        ii = ii ^ p;
        ii = ii * 0xe170893du;
        ii = ii ^ p;
        ii = ii ^ (ii >> 16u);
        ii = ii ^ p;
        ii = ii * 0x0929eb3fu;
        ii = ii ^ p;
        ii = ii ^ (ii >> 16u);
        ii = ii ^ (ii >> 16u);
        ii = ii * 0x796572b3u;
        ii = ii ^ (ii >> 16u);
        
        if (ii < l) {
            break;
        }
    }
    
    return (ii + p) % l;
}

// Generate CMJ 2D sample
fn cmj_sample_2d(s: u32, m: u32, n: u32, p: u32) -> vec2<f32> {
    let sx = cmj_permute(s % m, m, p * 0x51633e2du);
    let sy = cmj_permute(s / m, n, p * 0x68bc21ebu);
    
    let jx = cmj_permute(s, m * n, p * 0x02e5be93u);
    let jy = cmj_permute(s, m * n, p * 0x3d20adeau);
    
    let fx = f32(sx) + (f32(jx) / f32(m * n));
    let fy = f32(sy) + (f32(jy) / f32(m * n));
    
    return vec2<f32>(
        fx / f32(m),
        fy / f32(n)
    );
}

// Hash for per-pixel seed
fn cmj_hash(pixel: vec2<u32>, seed: u32) -> u32 {
    var h = pixel.x ^ (pixel.y << 16u) ^ seed;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    return h;
}

// Generate stratified CMJ sample for a pixel
// samples_per_dim: sqrt of total samples (e.g., 8 for 64 samples)
fn cmj_sample(pixel_coord: vec2<u32>, sample_index: u32, samples_per_dim: u32, seed: u32) -> vec2<f32> {
    let p = cmj_hash(pixel_coord, seed);
    return cmj_sample_2d(sample_index, samples_per_dim, samples_per_dim, p);
}
