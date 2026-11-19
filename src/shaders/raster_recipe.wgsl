struct Settings {
    width: u32,
    height: u32,
    edge_count: u32,
    right: u32,
    nodata_a: u32,
    nodata_b: u32,
}

const INVALID: u32 = 0x7fc00000u;
const ABS: u32 = 0x7fffffffu;
const SIGN: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> settings: Settings;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> cells: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> bins: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(7) var rgba: texture_storage_2d<rgba8unorm, write>;

fn supported(bits: u32, nodata: u32) -> bool {
    let magnitude = bits & ABS;
    // The least spacing at 2^-103 is the minimum normal (2^-126).
    // The upper bound is half of MAX, so both sum and difference stay finite.
    return bits != nodata &&
        !(magnitude == 0u && (nodata & ABS) == 0u) &&
        (magnitude == 0u || (magnitude >= 0x0c000000u && magnitude <= 0x7effffffu));
}

// Divide normal binary32 operands (or a zero numerator) with exact round to
// nearest, ties to even. WGSL floating division may differ by several ULPs,
// which would assign a different class to cells on an authored edge.
fn divide(n: u32, d: u32) -> u32 {
    let sign = (n ^ d) & SIGN;
    if ((n & ABS) == 0u) { return sign; }
    var exponent = i32((n >> 23u) & 255u) - i32((d >> 23u) & 255u);
    var remainder = (n & 0x007fffffu) | 0x00800000u;
    let divisor = (d & 0x007fffffu) | 0x00800000u;
    if (remainder < divisor) {
        remainder *= 2u;
        exponent -= 1;
    }
    if (exponent > 127) { return INVALID; }
    if (exponent < -150) { return sign; }
    if (exponent == -150) {
        // Exactly half the smallest subnormal ties to zero.
        return select(sign, sign | 1u, remainder != divisor);
    }
    let fraction_bits = u32(min(23, exponent + 149));
    remainder -= divisor;
    var quotient = 1u;
    for (var bit = 0u; bit < fraction_bits; bit += 1u) {
        remainder *= 2u;
        quotient *= 2u;
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient += 1u;
        }
    }
    if (remainder * 2u > divisor ||
        (remainder * 2u == divisor && (quotient & 1u) == 1u)) {
        quotient += 1u;
    }
    if (exponent < -126) { return sign | quotient; }
    // Addition carries into the exponent when the significand rounds up.
    let magnitude = (u32(exponent + 126) << 23u) + quotient;
    return select(sign | magnitude, INVALID, magnitude >= 0x7f800000u);
}

fn ordered(bits: u32) -> u32 {
    let canonical = select(bits, 0u, (bits & ABS) == 0u);
    return select(canonical ^ SIGN, ~canonical, (canonical & SIGN) != 0u);
}

@compute @workgroup_size(8, 8)
fn normalize(@builtin(global_invocation_id) pos: vec3<u32>) {
    if (pos.x >= settings.width || pos.y >= settings.height) { return; }
    let i = pos.y * settings.width + pos.x;
    let left = a[i];
    let right = b[i];
    var result = INVALID;
    if ((mask[i / 4u] >> ((i % 4u) * 8u) & 255u) != 0u &&
        supported(left, settings.nodata_a) && supported(right, settings.nodata_b)) {
        let lhs = bitcast<f32>(left);
        let rhs = bitcast<f32>(right);
        let denominator = bitcast<u32>(lhs + rhs);
        if ((denominator & ABS) != 0u) {
            let numerator = bitcast<u32>(lhs - rhs);
            // With a nonzero denominator, a zero difference is exact
            // cancellation and rounds to +0 before the division.
            result = divide(select(numerator, 0u, (numerator & ABS) == 0u), denominator);
        }
    }
    cells[i].x = result;
}

@compute @workgroup_size(8, 8)
fn classify(@builtin(global_invocation_id) pos: vec3<u32>) {
    if (pos.x >= settings.width || pos.y >= settings.height) { return; }
    let i = pos.y * settings.width + pos.x;
    let value = cells[i].x;
    var id = 0u;
    var color = vec4<f32>(0.0);
    if (value != INVALID) {
        let key = ordered(value);
        var lower = 0u;
        var upper = settings.edge_count;
        while (lower < upper) {
            let mid = lower + (upper - lower) / 2u;
            let edge = ordered(bins[mid].x);
            if (key > edge || (key == edge && settings.right == 0u)) {
                lower = mid + 1u;
            } else {
                upper = mid;
            }
        }
        id = lower + 1u;
        color = unpack4x8unorm(bins[lower].y);
    }
    cells[i].y = id;
    atomicAdd(&counts[id], 1u);
    textureStore(rgba, vec2<i32>(pos.xy), color);
}
