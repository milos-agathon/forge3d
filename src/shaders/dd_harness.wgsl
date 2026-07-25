// DUPLA GPU exactness and bound harness. determinism.wgsl and dd.wgsl are
// concatenated before this source; __DD_TWO_PROD_CALL__ is resolved first.

struct HarnessParams {
    offset: u32,
    count: u32,
    phase: u32,
    op: u32,
}

struct HarnessOutput {
    primary: DD,
    product: DD,
}

struct OperandPair {
    a: DD,
    b: DD,
}

@group(0) @binding(0) var<storage, read> params: HarnessParams;
@group(0) @binding(1) var<storage, read_write> results: array<HarnessOutput>;

fn mix32(value: u32) -> u32 {
    var x = value + 0x9e3779b9u;
    x = (x ^ (x >> 16u)) * 0x85ebca6bu;
    x = (x ^ (x >> 13u)) * 0xc2b2ae35u;
    return x ^ (x >> 16u);
}

fn normal_from_bits(state: u32, exponent: i32) -> f32 {
    let sign = (state & 1u) << 31u;
    let mantissa = (state >> 1u) & 0x007fffffu;
    let biased = u32(clamp(exponent + 127, 1, 254));
    return bitcast<f32>(sign | (biased << 23u) | mantissa);
}

fn generated_pair(index: u32) -> OperandPair {
    let s0 = mix32(index * 4u);
    let s1 = mix32(index * 4u + 1u);
    let s2 = mix32(index * 4u + 2u);
    let s3 = mix32(index * 4u + 3u);
    let exponent_a = i32(s0 % 121u) - 60;
    var exponent_b = i32(s1 % 121u) - 60;
    if (params.op == 1u) { exponent_b = -exponent_a; }
    if (params.op == 2u) { exponent_b = exponent_a + i32(s1 % 7u) - 3; }
    let a_hi = normal_from_bits(s0, exponent_a);
    var b_hi = normal_from_bits(s1, exponent_b);
    if (params.op == 0u && (index & 3u) == 0u) {
        b_hi = -a_hi;
    }
    let a_lo = normal_from_bits(s2, exponent_a - 25);
    var b_lo = normal_from_bits(s3, exponent_b - 25);
    if (params.op == 0u && (index & 3u) == 0u) {
        b_lo = -a_lo + normal_from_bits(s3, exponent_a - 48);
    }
    return OperandPair(quick_two_sum(a_hi, a_lo), quick_two_sum(b_hi, b_lo));
}

fn adversarial_pair(index: u32) -> OperandPair {
    let family = index & 7u;
    let salt = mix32(index);
    if (family == 0u) {
        return OperandPair(DD(1.0, bitcast<f32>(0x33800000u | (salt & 0x7fffffu))), DD(-1.0, bitcast<f32>(0x33000000u | (salt & 0x7fffffu))));
    }
    if (family == 1u) {
        return OperandPair(DD(bitcast<f32>(0x58800000u | (salt & 0x7fffffu)), 1.0), DD(bitcast<f32>(0x26800000u | ((salt >> 1u) & 0x7fffffu)), bitcast<f32>(0x1a000000u | (salt & 0x7fffffu))));
    }
    if (family == 2u) {
        return OperandPair(DD(bitcast<f32>(0x2b800000u | (salt & 0x7fffffu)), bitcast<f32>(0x1f000000u | (salt & 0x7fffffu))), DD(2.0, bitcast<f32>(0x33800000u)));
    }
    if (family == 3u) {
        return OperandPair(DD(bitcast<f32>(0x5e000000u | (salt & 0x7fffffu)), bitcast<f32>(0x51800000u | (salt & 0x7fffffu))), DD(bitcast<f32>(0x20000000u | ((salt >> 2u) & 0x7fffffu)), bitcast<f32>(0x13800000u | (salt & 0x7fffffu))));
    }
    if (family == 4u) {
        let hi = bitcast<f32>(0x3f800000u | (salt & 0x7fffffu));
        return OperandPair(DD(hi, bitcast<f32>(0x33000000u | ((salt >> 1u) & 0x7fffffu))), DD(-hi, bitcast<f32>(0x32800000u | ((salt >> 2u) & 0x7fffffu))));
    }
    if (family == 5u) {
        return OperandPair(DD(bitcast<f32>(0x58800000u | (salt & 0x7fffffu)), bitcast<f32>(0x4c000000u | ((salt >> 1u) & 0x7fffffu))), DD(bitcast<f32>(0x26800000u | ((salt >> 2u) & 0x7fffffu)), bitcast<f32>(0x1a000000u | ((salt >> 3u) & 0x7fffffu))));
    }
    if (family == 6u) {
        return OperandPair(DD(bitcast<f32>(0x3f000000u | (salt & 0x7fffffu)), bitcast<f32>(0x32800000u | ((salt >> 1u) & 0x7fffffu))), DD(bitcast<f32>(0x3f800000u | ((salt >> 2u) & 0x7fffffu)), -bitcast<f32>(0x33000000u | ((salt >> 3u) & 0x7fffffu))));
    }
    return OperandPair(DD(-bitcast<f32>(0x4f000000u | (salt & 0x7fffffu)), bitcast<f32>(0x42000000u | ((salt >> 1u) & 0x7fffffu))), DD(bitcast<f32>(0x2f000000u | ((salt >> 2u) & 0x7fffffu)), bitcast<f32>(0x23000000u | ((salt >> 3u) & 0x7fffffu))));
}

fn canary_pair(index: u32) -> vec2<u32> {
    switch index {
        case 0u: { return vec2<u32>(0x3f800000u, 0x33800000u); }
        case 1u: { return vec2<u32>(0x3f800000u, 0xbf800000u); }
        case 2u: { return vec2<u32>(0x007fffffu, 0x00000001u); }
        case 3u: { return vec2<u32>(0x7e801234u, 0x00804321u); }
        case 4u: { return vec2<u32>(0x00012345u, 0x5d123456u); }
        case 5u: { return vec2<u32>(0xfd554321u, 0x01800123u); }
        case 6u: { return vec2<u32>(0x3f800001u, 0x3f7fffffu); }
        case 7u: { return vec2<u32>(bitcast<u32>(12345.5), bitcast<u32>(-12344.75)); }
        case 8u: { return vec2<u32>(0x00000000u, 0x80000000u); }
        default: { return vec2<u32>(0x80000000u, 0x00000000u); }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= params.count) { return; }
    let index = params.offset + local;
    if (params.phase == 2u) {
        let pair = canary_pair(index);
        let a = bitcast<f32>(pair.x);
        let b = bitcast<f32>(pair.y);
        let sum = two_sum(a, b);
        let product = two_prod(a, b);
        results[local] = HarnessOutput(sum, product);
        return;
    }
    var operands = adversarial_pair(index);
    if (params.phase == 1u) { operands = generated_pair(index); }
    var value = dd_add(operands.a, operands.b);
    if (params.op == 1u) { value = dd_mul(operands.a, operands.b); }
    if (params.op == 2u) { value = dd_div(operands.a, operands.b); }
    if (params.op == 3u) { value = dd_sqrt(DD(abs(operands.a.hi), abs(operands.a.lo))); }
    results[local] = HarnessOutput(value, DD(0.0, 0.0));
}
