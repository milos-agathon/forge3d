// F3DZ v1 GPU decoder.
//
// One workgroup owns one independent page. Invocation 0 expands the two
// interleaved rANS states into byte scratch and parses variable-width escape
// tokens. The 64 invocations then reconstruct predictor values:
// - Lorenzo advances in row-wave (anti-diagonal) order. Cells on one diagonal
//   have no dependency on each other, so columns run in parallel.
// - plane, order-zero, and previous-LOD predictors are independent after token
//   parsing and run column-parallel.
//
// The workgroup array is the causal reconstruction context. RAW/NaN escape
// samples use INVALID_Q and therefore never contaminate predictor neighbors.

const DESC_STRIDE: u32 = 40u;
const TABLE_STRIDE: u32 = 4608u;
const SCALE_MASK: u32 = 4095u;
const SCALE_BITS: u32 = 12u;
const RANS_L: u32 = 8388608u;
const RAW_TOKEN: u32 = 4294967294u;
const NAN_TOKEN: u32 = 4294967295u;
const INVALID_Q: i32 = -2147483647i - 1i;

@group(0) @binding(0) var<storage, read> compressed: array<u32>;
@group(0) @binding(1) var<storage, read> descriptors: array<u32>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;
@group(0) @binding(3) var<storage, read_write> byte_scratch: array<u32>;
@group(0) @binding(4) var<storage, read_write> q_scratch: array<i32>;
@group(0) @binding(5) var<storage, read_write> value_bits: array<u32>;
@group(0) @binding(6) var<storage, read_write> output_bits: array<u32>;
@group(0) @binding(7) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(8) var atlas: texture_storage_2d<r32float, write>;

var<workgroup> work_q: array<i32, 4096>;

fn desc(page: u32, field: u32) -> u32 {
    return descriptors[page * DESC_STRIDE + field];
}

fn fail(page: u32, bit: u32) {
    atomicOr(&status[page], bit);
}

fn lane_byte(offset: u32, length: u32, position: ptr<function, u32>, page: u32) -> u32 {
    if (*position >= length) {
        fail(page, 1u);
        return 0u;
    }
    let value = compressed[offset + *position];
    *position = *position + 1u;
    return value;
}

fn decode_rans(page: u32, layer: u32) {
    let field = select(15u, 24u, layer == 1u);
    let decoded_len = desc(page, field);
    let state0_initial = desc(page, field + 1u);
    let state1_initial = desc(page, field + 2u);
    let lane0_offset = desc(page, field + 3u);
    let lane0_length = desc(page, field + 4u);
    let lane1_offset = desc(page, field + 5u);
    let lane1_length = desc(page, field + 6u);
    let table_offset = desc(page, field + 7u);
    let scratch_offset = desc(page, field + 8u);
    var states = vec2<u32>(state0_initial, state1_initial);
    var positions = vec2<u32>(0u, 0u);
    for (var index = 0u; index < decoded_len; index = index + 1u) {
        let lane = index & 1u;
        let state = select(states.x, states.y, lane == 1u);
        let slot = state & SCALE_MASK;
        let symbol = tables[table_offset + slot];
        let frequency = tables[table_offset + 4096u + symbol];
        let cumulative = tables[table_offset + 4352u + symbol];
        if (frequency == 0u) {
            fail(page, 2u);
            return;
        }
        var next = frequency * (state >> SCALE_BITS) + slot - cumulative;
        if (lane == 0u) {
            while (next < RANS_L) {
                next = (next << 8u) | lane_byte(lane0_offset, lane0_length, &positions.x, page);
            }
            states.x = next;
        } else {
            while (next < RANS_L) {
                next = (next << 8u) | lane_byte(lane1_offset, lane1_length, &positions.y, page);
            }
            states.y = next;
        }
        byte_scratch[scratch_offset + index] = symbol;
    }
    if (positions.x != lane0_length || positions.y != lane1_length ||
        states.x != RANS_L || states.y != RANS_L) {
        fail(page, 4u);
    }
}

fn word_at(byte_offset: u32, word: u32) -> u32 {
    let start = byte_offset + word * 4u;
    return byte_scratch[start] |
        (byte_scratch[start + 1u] << 8u) |
        (byte_scratch[start + 2u] << 16u) |
        (byte_scratch[start + 3u] << 24u);
}

fn is_nan_bits(bits: u32) -> bool {
    return (bits & 2139095040u) == 2139095040u && (bits & 8388607u) != 0u;
}

fn is_finite_bits(bits: u32) -> bool {
    return (bits & 2139095040u) != 2139095040u;
}

fn parse_tokens(page: u32, layer: u32) {
    let field = select(15u, 24u, layer == 1u);
    let decoded_len = desc(page, field);
    let byte_offset = desc(page, field + 8u);
    let sample_count = desc(page, 38u);
    let values_offset = desc(page, 35u);
    if ((decoded_len & 3u) != 0u) {
        fail(page, 8u);
        return;
    }
    let words = decoded_len / 4u;
    var cursor = 0u;
    var nan_count = 0u;
    for (var index = 0u; index < sample_count; index = index + 1u) {
        if (cursor >= words) {
            fail(page, 8u);
            return;
        }
        let token = word_at(byte_offset, cursor);
        cursor = cursor + 1u;
        if (token == RAW_TOKEN || token == NAN_TOKEN) {
            if (cursor >= words) {
                fail(page, 8u);
                return;
            }
            let bits = word_at(byte_offset, cursor);
            cursor = cursor + 1u;
            if ((token == RAW_TOKEN && !is_finite_bits(bits)) ||
                (token == NAN_TOKEN && !is_nan_bits(bits))) {
                fail(page, 16u);
            }
            if (token == NAN_TOKEN) {
                nan_count = nan_count + 1u;
            }
            work_q[index] = INVALID_Q;
            value_bits[values_offset + index] = bits;
        } else {
            work_q[index] = bitcast<i32>((token >> 1u) ^ (0u - (token & 1u)));
        }
    }
    if (cursor != words) {
        fail(page, 8u);
    }
    let final_layer = desc(page, 7u) == 1u || layer == 1u;
    if (final_layer && nan_count != desc(page, 39u)) {
        fail(page, 128u);
    }
}

fn add_checked(left: i32, right: i32, page: u32) -> i32 {
    let result = left + right;
    if (((left ^ result) & (right ^ result)) < 0i) {
        fail(page, 32u);
        return 0i;
    }
    return result;
}

fn add_overflows(left: i32, right: i32) -> bool {
    let result = left + right;
    return ((left ^ result) & (right ^ result)) < 0i;
}

fn sub_overflows(left: i32, right: i32) -> bool {
    let result = left - right;
    return ((left ^ right) & (left ^ result)) < 0i;
}

fn lorenzo_prediction(index: u32, width: u32) -> i32 {
    let x = index % width;
    let y = index / width;
    var left = INVALID_Q;
    var up = INVALID_Q;
    var upper_left = INVALID_Q;
    if (x > 0u) {
        left = work_q[index - 1u];
    }
    if (y > 0u) {
        up = work_q[index - width];
    }
    if (x > 0u && y > 0u) {
        upper_left = work_q[index - width - 1u];
    }
    if (left != INVALID_Q && up != INVALID_Q && upper_left != INVALID_Q) {
        // CPU's Lorenzo contract uses zero when the three-neighbor prediction
        // itself does not fit i32.
        if (add_overflows(left, up)) {
            return 0i;
        }
        let sum = left + up;
        if (sub_overflows(sum, upper_left)) {
            return 0i;
        }
        return sum - upper_left;
    }
    if (left != INVALID_Q) {
        return left;
    }
    if (up != INVALID_Q) {
        return up;
    }
    return 0i;
}

fn reconstruct_base(page: u32, lid: u32) {
    let width = desc(page, 0u);
    let height = desc(page, 1u);
    let count = desc(page, 38u);
    let predictor = desc(page, 8u);
    if (predictor == 0u) {
        // Anti-diagonal row waves preserve the exact causal Lorenzo order.
        for (var diagonal = 0u; diagonal < width + height - 1u; diagonal = diagonal + 1u) {
            let x = lid;
            if (x < width && diagonal >= x) {
                let y = diagonal - x;
                if (y < height) {
                    let index = y * width + x;
                    let residual = work_q[index];
                    if (residual != INVALID_Q) {
                        work_q[index] = add_checked(lorenzo_prediction(index, width), residual, page);
                    }
                }
            }
            workgroupBarrier();
        }
    } else {
        for (var index = lid; index < count; index = index + 64u) {
            let residual = work_q[index];
            if (residual != INVALID_Q) {
                var predicted = 0i;
                if (predictor == 1u) {
                    let x = i32(index % width);
                    let y = i32(index / width);
                    predicted = bitcast<i32>(desc(page, 12u)) * x +
                        bitcast<i32>(desc(page, 13u)) * y + bitcast<i32>(desc(page, 14u));
                }
                work_q[index] = add_checked(predicted, residual, page);
            }
        }
        workgroupBarrier();
    }
    let base_q_offset = desc(page, 33u);
    for (var index = lid; index < count; index = index + 64u) {
        q_scratch[base_q_offset + index] = work_q[index];
    }
    workgroupBarrier();
}

fn reconstruct_enhancement(page: u32, lid: u32) {
    let count = desc(page, 38u);
    let base_q_offset = desc(page, 33u);
    for (var index = lid; index < count; index = index + 64u) {
        let residual = work_q[index];
        if (residual != INVALID_Q) {
            let base = q_scratch[base_q_offset + index];
            var predicted = 0i;
            if (base != INVALID_Q) {
                if (base < -536870912i || base > 536870911i) {
                    fail(page, 64u);
                } else {
                    predicted = base * 4i;
                }
            }
            work_q[index] = add_checked(predicted, residual, page);
        }
    }
    workgroupBarrier();
}

fn write_result(page: u32, lid: u32, to_atlas: bool) {
    let width = desc(page, 0u);
    let count = desc(page, 38u);
    let global_width = desc(page, 2u);
    let global_x = desc(page, 3u);
    let global_y = desc(page, 4u);
    let atlas_x = desc(page, 5u);
    let atlas_y = desc(page, 6u);
    let value_offset = desc(page, 35u);
    let step = bitcast<f32>(select(desc(page, 11u), desc(page, 10u), desc(page, 7u) == 0u));
    for (var index = lid; index < count; index = index + 64u) {
        let x = index % width;
        let y = index / width;
        let q = work_q[index];
        let bits = select(value_bits[value_offset + index], bitcast<u32>(f32(q) * step), q != INVALID_Q);
        if (to_atlas) {
            textureStore(atlas, vec2<i32>(i32(atlas_x + global_x + x), i32(atlas_y + global_y + y)),
                vec4<f32>(bitcast<f32>(bits), 0.0, 0.0, 0.0));
        } else {
            output_bits[(global_y + y) * global_width + global_x + x] = bits;
        }
    }
}

fn decode_page(page: u32, lid: u32, to_atlas: bool) {
    if (lid == 0u) {
        decode_rans(page, 0u);
        parse_tokens(page, 0u);
    }
    workgroupBarrier();
    if (atomicLoad(&status[page]) != 0u) {
        return;
    }
    reconstruct_base(page, lid);
    if (atomicLoad(&status[page]) != 0u) {
        return;
    }
    if (desc(page, 7u) == 0u) {
        if (lid == 0u) {
            decode_rans(page, 1u);
            parse_tokens(page, 1u);
        }
        workgroupBarrier();
        if (atomicLoad(&status[page]) != 0u) {
            return;
        }
        reconstruct_enhancement(page, lid);
    }
    if (atomicLoad(&status[page]) != 0u) {
        return;
    }
    write_result(page, lid, to_atlas);
}

@compute @workgroup_size(64, 1, 1)
fn decode_to_buffer(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    decode_page(group_id.x, local_id.x, false);
}

@compute @workgroup_size(64, 1, 1)
fn decode_to_atlas(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    decode_page(group_id.x, local_id.x, true);
}
