// Sobol sequence generator using Joe-Kuo direction numbers
// Provides low-discrepancy 2D samples for better stratification than pseudo-random

// Sobol direction numbers (first 32 bits for dimensions 0 and 1)
// These are precomputed from Joe-Kuo tables
const SOBOL_DIRECTIONS_X: array<u32, 32> = array<u32, 32>(
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
    0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
    0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u,
    0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
    0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
    0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u
);

const SOBOL_DIRECTIONS_Y: array<u32, 32> = array<u32, 32>(
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u,
    0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
    0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u,
    0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
    0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u,
    0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
    0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u,
    0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu
);

// Generate Sobol sample for given index with Owen scrambling
// Manually unrolled to avoid dynamic array indexing (WGSL limitation)
fn sobol_2d(index: u32, scramble: vec2<u32>) -> vec2<f32> {
    var result_x = 0u;
    var result_y = 0u;
    var i = index;
    
    // Unrolled loop for first 16 bits (sufficient for most use cases)
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[0]; result_y ^= SOBOL_DIRECTIONS_Y[0]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[1]; result_y ^= SOBOL_DIRECTIONS_Y[1]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[2]; result_y ^= SOBOL_DIRECTIONS_Y[2]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[3]; result_y ^= SOBOL_DIRECTIONS_Y[3]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[4]; result_y ^= SOBOL_DIRECTIONS_Y[4]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[5]; result_y ^= SOBOL_DIRECTIONS_Y[5]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[6]; result_y ^= SOBOL_DIRECTIONS_Y[6]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[7]; result_y ^= SOBOL_DIRECTIONS_Y[7]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[8]; result_y ^= SOBOL_DIRECTIONS_Y[8]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[9]; result_y ^= SOBOL_DIRECTIONS_Y[9]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[10]; result_y ^= SOBOL_DIRECTIONS_Y[10]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[11]; result_y ^= SOBOL_DIRECTIONS_Y[11]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[12]; result_y ^= SOBOL_DIRECTIONS_Y[12]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[13]; result_y ^= SOBOL_DIRECTIONS_Y[13]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[14]; result_y ^= SOBOL_DIRECTIONS_Y[14]; } i = i >> 1u;
    if ((i & 1u) != 0u) { result_x ^= SOBOL_DIRECTIONS_X[15]; result_y ^= SOBOL_DIRECTIONS_Y[15]; }
    
    // Apply Owen scrambling (per-pixel randomization)
    result_x ^= scramble.x;
    result_y ^= scramble.y;
    
    // Convert to [0,1) float
    return vec2<f32>(
        f32(result_x) * 2.3283064365386963e-10, // 1.0 / 2^32
        f32(result_y) * 2.3283064365386963e-10
    );
}

// Hash function for generating per-pixel scramble
fn hash_scramble(pixel: vec2<u32>, seed: u32) -> vec2<u32> {
    var h = pixel.x ^ (pixel.y << 16u) ^ seed;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    
    var h2 = h ^ 0x9e3779b9u;
    h2 = (h2 ^ 61u) ^ (h2 >> 16u);
    h2 = h2 + (h2 << 3u);
    h2 = h2 ^ (h2 >> 4u);
    h2 = h2 * 0x27d4eb2du;
    h2 = h2 ^ (h2 >> 15u);
    
    return vec2<u32>(h, h2);
}

// Generate stratified Sobol sample for a pixel and sample index
fn sobol_sample(pixel_coord: vec2<u32>, sample_index: u32, dimension_offset: u32, seed: u32) -> vec2<f32> {
    let scramble = hash_scramble(pixel_coord + vec2<u32>(dimension_offset, 0u), seed);
    return sobol_2d(sample_index, scramble);
}
