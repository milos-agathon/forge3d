// src/shaders/radix_sort_pairs.wgsl
// GPU radix sort passes for key-value pairs (Morton code, primitive index)
// using 4-bit digits (16 buckets). Includes a fallback bitonic sorter for
// tiny arrays (<=256 elements) and multi-pass radix for larger arrays.
// RELEVANT FILES: src/accel/lbvh_gpu.rs

struct Uniforms {
    prim_count: u32,
    pass_shift: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> input_keys: array<u32>;
@group(1) @binding(1) var<storage, read> input_values: array<u32>;
@group(2) @binding(0) var<storage, read_write> output_keys: array<u32>;
@group(2) @binding(1) var<storage, read_write> output_values: array<u32>;
@group(3) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(3) @binding(1) var<storage, read_write> prefix_sums: array<u32>;

// Clear histogram and prefix buffers (single workgroup, 16 threads)
@compute @workgroup_size(16)
fn clear_hist(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    if i < 16u {
        atomicStore(&histogram[i], 0u);
        prefix_sums[i] = 0u;
    }
}

// Count digits into global histogram using atomics (many workgroups)
@compute @workgroup_size(256)
fn count_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= uniforms.prim_count { return; }
    let key = input_keys[index];
    let digit = (key >> uniforms.pass_shift) & 0xfu;
    atomicAdd(&histogram[digit], 1u);
}

// Compute exclusive prefix sums across 16 histogram buckets (single thread)
@compute @workgroup_size(1)
fn scan_pass() {
    var sum = 0u;
    let N = 16u;
    for (var i = 0u; i < N; i = i + 1u) {
        let c = atomicLoad(&histogram[i]);
        prefix_sums[i] = sum;
        sum = sum + c;
    }
}

// Scatter elements into output arrays using prefix_sums and per-bucket atomic offsets
@compute @workgroup_size(256)
fn scatter_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= uniforms.prim_count { return; }
    let key = input_keys[index];
    let val = input_values[index];
    let digit = (key >> uniforms.pass_shift) & 0xfu;
    let local_off = atomicAdd(&histogram[digit], 1u);
    let base = prefix_sums[digit];
    let pos = base + local_off;
    if pos < uniforms.prim_count {
        output_keys[pos] = key;
        output_values[pos] = val;
    }
}

// -----------------------------------------------------------------------------
// Functional small-array GPU sort using a single-workgroup bitonic sorter.
// This is sufficient for tiny scenes (<=256 elements). Larger scenes can fall
// back to CPU sort or be extended later.
// -----------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> uniforms2: Uniforms;
@group(1) @binding(0) var<storage, read> in_keys: array<u32>;
@group(1) @binding(1) var<storage, read> in_vals: array<u32>;
@group(2) @binding(0) var<storage, read_write> out_keys: array<u32>;
@group(2) @binding(1) var<storage, read_write> out_vals: array<u32>;

var<workgroup> wk: array<u32, 256>;
var<workgroup> wv: array<u32, 256>;

@compute @workgroup_size(256)
fn bitonic_sort(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let idx = lid.x;
    let base = wgid.x * 256u;
    let gidx = base + idx;

    // Load
    if gidx < uniforms2.prim_count {
        wk[idx] = in_keys[gidx];
        wv[idx] = in_vals[gidx];
    } else {
        wk[idx] = 0xffffffffu; // sentinel for out-of-range
        wv[idx] = 0u;
    }
    workgroupBarrier();

    // Bitonic sort within workgroup (256 elements)
    var k = 2u;
    loop {
        if k > 256u { break; }
        var j = k >> 1u;
        loop {
            if j == 0u { break; }
            let ixj = idx ^ j;
            if ixj > idx {
                let ascending = (idx & k) == 0u;
                // Compare-swap by key, then by value to break ties
                let key_i = wk[idx];
                let key_j = wk[ixj];
                let val_i = wv[idx];
                let val_j = wv[ixj];
                var swap = false;
                if ascending {
                    if (key_i > key_j) || ((key_i == key_j) && (val_i > val_j)) {
                        swap = true;
                    }
                } else {
                    if (key_i < key_j) || ((key_i == key_j) && (val_i < val_j)) {
                        swap = true;
                    }
                }
                if swap {
                    wk[idx] = key_j; wk[ixj] = key_i;
                    wv[idx] = val_j; wv[ixj] = val_i;
                }
            }
            workgroupBarrier();
            j = j >> 1u;
        }
        k = k << 1u;
    }

    // Store
    if gidx < uniforms2.prim_count {
        out_keys[gidx] = wk[idx];
        out_vals[gidx] = wv[idx];
    }
}