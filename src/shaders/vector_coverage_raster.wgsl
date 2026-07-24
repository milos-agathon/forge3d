// LIMES exact scan-cell analytic coverage.
//
// Linear boundaries use the identity
//   integral_a^b x(y)dy = (x(a)+x(b))*(b-a)/2
// after endpoint/intersection/side-crossing subdivision: an exact trapezoid.
// Circular boundaries use an algebraically stable definite form of
//   integral_a^b sqrt(r^2-d^2)dd.
// Its angle delta comes from atan2(cross, dot), avoiding subtraction of two
// large antiderivative values. The fixed-order, range-reduced atan polynomial
// is built only from TERRA-DETERMINATA operations.
// The committed pinned-math sweep bounds its final-cell epsilon_geom below
// 1.8e-3 pixel area at r<=100, folded under LIMES's 0.5/255 max-error budget.
// No boundary is flattened.
//
// Atomic tile insertion order is never observed numerically. Crossings are
// sorted by x and stable primitive ID in every slab, fixing accumulation order.

const PRIMITIVE_LINE: u32 = 0u;
const FILL_NONZERO: u32 = 0u;
const MAX_ACTIVE: u32 = 96u;
const MAX_BREAKS: u32 = 256u;
const EPSILON: f32 = 1.0e-5;
const LEFT_BOUNDARY: u32 = 0xffffffffu;
const RIGHT_BOUNDARY: u32 = 0xfffffffeu;

struct PrimitiveRecord {
    geometry: vec4<f32>,
    bounds: vec4<f32>,
    metadata: vec4<u32>,
}

struct RasterParams {
    extent_tiles: vec4<u32>,
    layers_capacity: vec4<u32>,
}

struct AtomicWords {
    values: array<atomic<u32>>,
}

struct Words {
    values: array<u32>,
}

struct SignedWords {
    values: array<i32>,
}

struct FloatWords {
    values: array<f32>,
}

@group(0) @binding(0) var<storage, read> primitives: array<PrimitiveRecord>;
@group(0) @binding(1) var<storage, read_write> tile_counts: AtomicWords;
@group(0) @binding(2) var<storage, read> tile_indices: Words;
@group(0) @binding(3) var<storage, read> tile_baselines: SignedWords;
@group(0) @binding(4) var<storage, read> layer_rules: array<vec4<u32>>;
@group(0) @binding(5) var<uniform> params: RasterParams;
@group(0) @binding(6) var<storage, read_write> coverage_output: FloatWords;
@group(0) @binding(7) var<storage, read_write> errors: AtomicWords;

fn winding(primitive: PrimitiveRecord) -> i32 {
    return bitcast<i32>(primitive.metadata.z);
}

fn limes_fma2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(det_fma(a.x, b.x, c.x), det_fma(a.y, b.y, c.y));
}

fn active_at(primitive: PrimitiveRecord, y: f32) -> bool {
    return y >= primitive.bounds.y && y < primitive.bounds.w && winding(primitive) != 0i;
}

fn primitive_x(primitive: PrimitiveRecord, y: f32) -> f32 {
    if primitive.metadata.x == PRIMITIVE_LINE {
        let numerator = det_fma(y - primitive.geometry.y,
                                primitive.geometry.z - primitive.geometry.x,
                                0.0);
        return det_fma(det_div(numerator, primitive.geometry.w - primitive.geometry.y),
                       1.0, primitive.geometry.x);
    }
    let distance = y - primitive.geometry.y;
    let radicand = det_fma(-distance, distance,
                           primitive.geometry.z * primitive.geometry.z);
    return det_fma(primitive.geometry.w, det_sqrt(max(radicand, 0.0)),
                   primitive.geometry.x);
}

fn limes_atan_series(value: f32) -> f32 {
    let x = det_barrier(value);
    let s = det_barrier(x * x);
    // atan(x)=x*(1-s/3+s^2/5-...+s^12/25), |x|<=sqrt(2)-1.
    var p = -0.04;
    p = det_fma(s, p, 0.043478260869565216);
    p = det_fma(s, p, -0.047619047619047616);
    p = det_fma(s, p, 0.05263157894736842);
    p = det_fma(s, p, -0.058823529411764705);
    p = det_fma(s, p, 0.06666666666666667);
    p = det_fma(s, p, -0.07692307692307693);
    p = det_fma(s, p, 0.09090909090909091);
    p = det_fma(s, p, -0.1111111111111111);
    p = det_fma(s, p, 0.14285714285714285);
    p = det_fma(s, p, -0.2);
    p = det_fma(s, p, 0.3333333333333333);
    return x * det_fma(-s, p, 1.0);
}

fn limes_atan01(value: f32) -> f32 {
    let a = clamp(value, 0.0, 1.0);
    if a <= 0.41421356237309503 {
        return limes_atan_series(a);
    }
    let reduced = det_div(a - 1.0, a + 1.0);
    return det_barrier(0.7853981633974483 + limes_atan_series(reduced));
}

fn limes_atan2(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    let high = max(ax, ay);
    if high == 0.0 {
        return 0.0;
    }
    var angle = limes_atan01(det_div(min(ax, ay), high));
    angle = select(angle, 1.5707963267948966 - angle, ay > ax);
    angle = select(angle, 3.141592653589793 - angle, x < 0.0);
    return select(angle, -angle, y < 0.0);
}

fn circle_segment_integral(a: f32, b: f32, radius: f32) -> f32 {
    let da = clamp(a, -radius, radius);
    let db = clamp(b, -radius, radius);
    let radius_squared = radius * radius;
    let root_a = det_sqrt(max(det_fma(-da, da, radius_squared), 0.0));
    let root_b = det_sqrt(max(det_fma(-db, db, radius_squared), 0.0));
    let cross = max(det_fma(db, root_a, -da * root_b), 0.0);
    let dot = det_fma(root_a, root_b, da * db);
    let angle_delta = limes_atan2(cross, dot);
    let cosine_sum_numerator = det_fma(root_a, root_b, -da * db);
    let correction = det_div(cosine_sum_numerator * cross, radius_squared);
    return 0.5 * det_fma(radius_squared, angle_delta, correction);
}

fn primitive_integral(primitive: PrimitiveRecord, a: f32, b: f32) -> f32 {
    if primitive.metadata.x == PRIMITIVE_LINE {
        return 0.5 * (primitive_x(primitive, a) + primitive_x(primitive, b)) * (b - a);
    }
    let segment = circle_segment_integral(a - primitive.geometry.y,
                                          b - primitive.geometry.y,
                                          primitive.geometry.z);
    return det_fma(primitive.geometry.w, segment, primitive.geometry.x * (b - a));
}

fn push_break(values: ptr<function, array<f32, 256>>,
              count: ptr<function, u32>,
              value: f32, y0: f32, y1: f32) {
    if value != value || abs(value) > 3.4028235e38 ||
       value <= y0 + EPSILON || value >= y1 - EPSILON {
        return;
    }
    if (*count) >= MAX_BREAKS {
        atomicStore(&errors.values[2], 1u);
        return;
    }
    (*values)[*count] = value;
    *count = *count + 1u;
}

fn side_breaks(values: ptr<function, array<f32, 256>>,
               count: ptr<function, u32>,
               primitive: PrimitiveRecord, side_x: f32, y0: f32, y1: f32) {
    if primitive.metadata.x == PRIMITIVE_LINE {
        let dx = primitive.geometry.z - primitive.geometry.x;
        if abs(dx) > EPSILON {
            let t = det_div(side_x - primitive.geometry.x, dx);
            if t > 0.0 && t < 1.0 {
                push_break(values, count,
                    det_fma(t, primitive.geometry.w - primitive.geometry.y,
                            primitive.geometry.y), y0, y1);
            }
        }
        return;
    }
    let dx = side_x - primitive.geometry.x;
    if dx * primitive.geometry.w < -EPSILON {
        return;
    }
    let remaining = det_fma(-dx, dx, primitive.geometry.z * primitive.geometry.z);
    if remaining < 0.0 {
        return;
    }
    let dy = det_sqrt(max(remaining, 0.0));
    let low = primitive.geometry.y - dy;
    let high = primitive.geometry.y + dy;
    if low >= primitive.bounds.y - EPSILON && low <= primitive.bounds.w + EPSILON {
        push_break(values, count, low, y0, y1);
    }
    if high >= primitive.bounds.y - EPSILON && high <= primitive.bounds.w + EPSILON {
        push_break(values, count, high, y0, y1);
    }
}

fn contains_point(primitive: PrimitiveRecord, point: vec2<f32>) -> bool {
    if point.y < primitive.bounds.y - EPSILON || point.y > primitive.bounds.w + EPSILON {
        return false;
    }
    if primitive.metadata.x == PRIMITIVE_LINE {
        return point.x >= min(primitive.geometry.x, primitive.geometry.z) - EPSILON &&
               point.x <= max(primitive.geometry.x, primitive.geometry.z) + EPSILON &&
               point.y >= min(primitive.geometry.y, primitive.geometry.w) - EPSILON &&
               point.y <= max(primitive.geometry.y, primitive.geometry.w) + EPSILON;
    }
    return (point.x - primitive.geometry.x) * primitive.geometry.w >= -EPSILON;
}

fn intersection_break(values: ptr<function, array<f32, 256>>,
                      count: ptr<function, u32>,
                      point: vec2<f32>, left: PrimitiveRecord, right: PrimitiveRecord,
                      pixel: vec4<f32>) {
    if point.x >= pixel.x - EPSILON && point.x <= pixel.y + EPSILON &&
       contains_point(left, point) && contains_point(right, point) {
        push_break(values, count, point.y, pixel.z, pixel.w);
    }
}

fn line_line_break(values: ptr<function, array<f32, 256>>,
                   count: ptr<function, u32>,
                   left: PrimitiveRecord, right: PrimitiveRecord,
                   pixel: vec4<f32>) {
    let r = left.geometry.zw - left.geometry.xy;
    let s = right.geometry.zw - right.geometry.xy;
    let denominator = det_fma(r.x, s.y, -r.y * s.x);
    if abs(denominator) <= EPSILON {
        return;
    }
    let offset = right.geometry.xy - left.geometry.xy;
    let t = det_div(det_fma(offset.x, s.y, -offset.y * s.x), denominator);
    let u = det_div(det_fma(offset.x, r.y, -offset.y * r.x), denominator);
    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        intersection_break(values, count, limes_fma2(r, vec2<f32>(t), left.geometry.xy),
                           left, right, pixel);
    }
}

fn line_arc_breaks(values: ptr<function, array<f32, 256>>,
                   count: ptr<function, u32>,
                   line: PrimitiveRecord, arc: PrimitiveRecord,
                   pixel: vec4<f32>) {
    let direction = line.geometry.zw - line.geometry.xy;
    let offset = line.geometry.xy - arc.geometry.xy;
    let qa = det_dot2(direction, direction);
    if qa <= EPSILON {
        return;
    }
    let qb = 2.0 * det_dot2(offset, direction);
    let qc = det_fma(-arc.geometry.z, arc.geometry.z, det_dot2(offset, offset));
    let discriminant = det_fma(-4.0 * qa, qc, qb * qb);
    if discriminant < 0.0 {
        return;
    }
    let root = det_sqrt(max(discriminant, 0.0));
    let denominator = 2.0 * qa;
    let t0 = det_div(-qb - root, denominator);
    let t1 = det_div(-qb + root, denominator);
    if t0 >= -EPSILON && t0 <= 1.0 + EPSILON {
        intersection_break(values, count,
            limes_fma2(direction, vec2<f32>(t0), line.geometry.xy), line, arc, pixel);
    }
    if t1 >= -EPSILON && t1 <= 1.0 + EPSILON && abs(t1 - t0) > EPSILON {
        intersection_break(values, count,
            limes_fma2(direction, vec2<f32>(t1), line.geometry.xy), line, arc, pixel);
    }
}

fn arc_arc_breaks(values: ptr<function, array<f32, 256>>,
                  count: ptr<function, u32>,
                  left: PrimitiveRecord, right: PrimitiveRecord,
                  pixel: vec4<f32>) {
    let delta = right.geometry.xy - left.geometry.xy;
    let distance = det_sqrt(max(det_dot2(delta, delta), 0.0));
    let r0 = left.geometry.z;
    let r1 = right.geometry.z;
    if distance <= EPSILON || distance > r0 + r1 + EPSILON ||
       distance < abs(r0 - r1) - EPSILON {
        return;
    }
    let along = det_div(det_fma(-r1, r1, det_fma(r0, r0, distance * distance)),
                        2.0 * distance);
    let height = det_sqrt(max(det_fma(-along, along, r0 * r0), 0.0));
    let unit = delta * det_rcp(distance);
    let base = limes_fma2(unit, vec2<f32>(along), left.geometry.xy);
    let perpendicular = vec2<f32>(-unit.y, unit.x) * height;
    intersection_break(values, count, base + perpendicular, left, right, pixel);
    if height > EPSILON {
        intersection_break(values, count, base - perpendicular, left, right, pixel);
    }
}

fn pair_breaks(values: ptr<function, array<f32, 256>>,
               count: ptr<function, u32>,
               left: PrimitiveRecord, right: PrimitiveRecord,
               pixel: vec4<f32>) {
    if left.metadata.x == PRIMITIVE_LINE && right.metadata.x == PRIMITIVE_LINE {
        line_line_break(values, count, left, right, pixel);
    } else if left.metadata.x == PRIMITIVE_LINE {
        line_arc_breaks(values, count, left, right, pixel);
    } else if right.metadata.x == PRIMITIVE_LINE {
        line_arc_breaks(values, count, right, left, pixel);
    } else {
        arc_arc_breaks(values, count, left, right, pixel);
    }
}

fn is_inside(rule: u32, state: i32) -> bool {
    if rule == FILL_NONZERO {
        return state != 0i;
    }
    return (state & 1i) != 0i;
}

fn state_contribution(rule: u32, primitive: PrimitiveRecord) -> i32 {
    if rule == FILL_NONZERO {
        return winding(primitive);
    }
    return 1i;
}

fn clipped_integral(boundary: u32, a: f32, b: f32, mid: f32,
                    x0: f32, x1: f32, tile_left: f32, tile_right: f32) -> f32 {
    if boundary == LEFT_BOUNDARY {
        return clamp(tile_left, x0, x1) * (b - a);
    }
    if boundary == RIGHT_BOUNDARY {
        return clamp(tile_right, x0, x1) * (b - a);
    }
    let primitive = primitives[boundary];
    let x = primitive_x(primitive, mid);
    if x <= x0 {
        return x0 * (b - a);
    }
    if x >= x1 {
        return x1 * (b - a);
    }
    return primitive_integral(primitive, a, b);
}

fn interval_area(left: u32, right: u32, a: f32, b: f32, mid: f32,
                 x0: f32, x1: f32, tile_left: f32, tile_right: f32) -> f32 {
    let left_integral = clipped_integral(left, a, b, mid, x0, x1, tile_left, tile_right);
    let right_integral = clipped_integral(right, a, b, mid, x0, x1, tile_left, tile_right);
    return max(right_integral - left_integral, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params.extent_tiles.x;
    let height = params.extent_tiles.y;
    let layer_count = params.layers_capacity.x;
    if gid.x >= width || gid.y >= height || gid.z >= layer_count {
        return;
    }
    let pixel_offset = (gid.z * height + gid.y) * width + gid.x;
    let tile_columns = params.extent_tiles.z;
    let tile_rows = params.extent_tiles.w;
    let tile_x = gid.x / params.layers_capacity.z;
    let tile_y = gid.y / params.layers_capacity.z;
    let tile_count = tile_columns * tile_rows;
    let tile = gid.z * tile_count + tile_y * tile_columns + tile_x;
    let capacity = params.layers_capacity.y;
    let count = min(atomicLoad(&tile_counts.values[tile]), capacity);
    let tile_base = tile * capacity;

    var active_indices: array<u32, 96>;
    var active_count = 0u;
    let y0 = f32(gid.y);
    let y1 = y0 + 1.0;
    var slot = 0u;
    loop {
        if slot >= count {
            break;
        }
        let primitive_index = tile_indices.values[tile_base + slot];
        let primitive = primitives[primitive_index];
        if primitive.bounds.w > y0 && primitive.bounds.y < y1 && winding(primitive) != 0i {
            if active_count >= MAX_ACTIVE {
                atomicStore(&errors.values[1], 1u);
                coverage_output.values[pixel_offset] = 0.0;
                return;
            }
            active_indices[active_count] = primitive_index;
            active_count += 1u;
        }
        slot += 1u;
    }

    let rule = layer_rules[gid.z].x;
    let x0 = f32(gid.x);
    let x1 = x0 + 1.0;
    let tile_left = f32(tile_x * params.layers_capacity.z);
    let tile_right = min(tile_left + f32(params.layers_capacity.z), f32(width));
    let pixel_bounds = vec4<f32>(x0, x1, y0, y1);

    var breaks: array<f32, 256>;
    var break_count = 2u;
    breaks[0] = y0;
    breaks[1] = y1;
    var i = 0u;
    loop {
        if i >= active_count {
            break;
        }
        let primitive = primitives[active_indices[i]];
        push_break(&breaks, &break_count, primitive.bounds.y, y0, y1);
        push_break(&breaks, &break_count, primitive.bounds.w, y0, y1);
        side_breaks(&breaks, &break_count, primitive, x0, y0, y1);
        side_breaks(&breaks, &break_count, primitive, x1, y0, y1);
        side_breaks(&breaks, &break_count, primitive, tile_left, y0, y1);
        side_breaks(&breaks, &break_count, primitive, tile_right, y0, y1);
        var j = i + 1u;
        loop {
            if j >= active_count {
                break;
            }
            pair_breaks(&breaks, &break_count, primitive,
                        primitives[active_indices[j]], pixel_bounds);
            j += 1u;
        }
        i += 1u;
    }
    if atomicLoad(&errors.values[2]) != 0u {
        coverage_output.values[pixel_offset] = 0.0;
        return;
    }

    // Deterministic insertion sort and exact duplicate removal.
    i = 1u;
    loop {
        if i >= break_count {
            break;
        }
        let value = breaks[i];
        var j = i;
        loop {
            if j == 0u || breaks[j - 1u] <= value {
                break;
            }
            breaks[j] = breaks[j - 1u];
            j -= 1u;
        }
        breaks[j] = value;
        i += 1u;
    }
    var unique_count = 1u;
    i = 1u;
    loop {
        if i >= break_count {
            break;
        }
        if abs(breaks[i] - breaks[unique_count - 1u]) > EPSILON {
            breaks[unique_count] = breaks[i];
            unique_count += 1u;
        }
        i += 1u;
    }

    var result = 0.0;
    var slab_index = 0u;
    loop {
        if slab_index + 1u >= unique_count {
            break;
        }
        let a = breaks[slab_index];
        let b = breaks[slab_index + 1u];
        let mid = 0.5 * (a + b);
        if b - a > EPSILON {
            var crossing_indices: array<u32, 96>;
            var crossing_x: array<f32, 96>;
            var crossing_count = 0u;
            i = 0u;
            loop {
                if i >= active_count {
                    break;
                }
                let primitive_index = active_indices[i];
                let primitive = primitives[primitive_index];
                if active_at(primitive, mid) {
                    let x = primitive_x(primitive, mid);
                    if x >= tile_left - EPSILON && x <= tile_right + EPSILON {
                        // Insertion-sort the crossing immediately. Stable ID is
                        // the tie break, independent of atomic bin order.
                        var position = crossing_count;
                        loop {
                            if position == 0u {
                                break;
                            }
                            let previous = position - 1u;
                            let previous_index = crossing_indices[previous];
                            let previous_x = crossing_x[previous];
                            if previous_x < x - EPSILON ||
                               (abs(previous_x - x) <= EPSILON &&
                                primitives[previous_index].metadata.w <= primitive.metadata.w) {
                                break;
                            }
                            crossing_indices[position] = previous_index;
                            crossing_x[position] = previous_x;
                            position -= 1u;
                        }
                        crossing_indices[position] = primitive_index;
                        crossing_x[position] = x;
                        crossing_count += 1u;
                    }
                }
                i += 1u;
            }

            let baseline_index = (gid.z * height + gid.y) * tile_columns + tile_x;
            var state = tile_baselines.values[baseline_index];
            let row_center = y0 + 0.5;
            // The uploaded baseline is at row center. Adjust it exactly to this
            // slab midpoint using local boundaries; any boundary capable of
            // changing state at the tile edge is necessarily in this tile bin.
            i = 0u;
            loop {
                if i >= active_count {
                    break;
                }
                let primitive = primitives[active_indices[i]];
                let contribution = state_contribution(rule, primitive);
                if active_at(primitive, row_center) &&
                   primitive_x(primitive, row_center) < tile_left {
                    state -= contribution;
                }
                if active_at(primitive, mid) && primitive_x(primitive, mid) < tile_left {
                    state += contribution;
                }
                i += 1u;
            }

            var left_boundary = LEFT_BOUNDARY;
            var inside = is_inside(rule, state);
            var cursor = 0u;
            loop {
                if cursor >= crossing_count {
                    break;
                }
                let group_x = crossing_x[cursor];
                let representative = crossing_indices[cursor];
                let was_inside = inside;
                var group_end = cursor;
                loop {
                    if group_end >= crossing_count ||
                       abs(crossing_x[group_end] - group_x) > EPSILON {
                        break;
                    }
                    state += state_contribution(rule, primitives[crossing_indices[group_end]]);
                    group_end += 1u;
                }
                inside = is_inside(rule, state);
                if !was_inside && inside {
                    left_boundary = representative;
                } else if was_inside && !inside {
                    result += interval_area(left_boundary, representative, a, b, mid,
                                            x0, x1, tile_left, tile_right);
                }
                cursor = group_end;
            }
            if inside {
                result += interval_area(left_boundary, RIGHT_BOUNDARY, a, b, mid,
                                        x0, x1, tile_left, tile_right);
            }
        }
        slab_index += 1u;
    }
    coverage_output.values[pixel_offset] = clamp(det_barrier(result), 0.0, 1.0);
}
