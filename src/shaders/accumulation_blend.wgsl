// M1: Accumulation AA blend shader
// Accumulates rendered samples into an HDR buffer and computes final average

struct AccumulationParams {
    sample_index: u32,      // Current sample (0-based)
    total_samples: u32,     // Total samples to accumulate
    width: u32,             // Image width
    height: u32,            // Image height
}

@group(0) @binding(0) var<uniform> params: AccumulationParams;
@group(0) @binding(1) var current_sample: texture_2d<f32>;
@group(0) @binding(2) var accumulation: texture_storage_2d<rgba32float, read_write>;

// Accumulate current sample into accumulation buffer
// Uses running average: acc = acc + (new - acc) / (n + 1)
// This is numerically more stable than sum / n for large n
@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let coords = vec2<i32>(i32(x), i32(y));
    
    // Load current sample (from rendered frame)
    let current = textureLoad(current_sample, coords, 0);
    
    // Load accumulated value
    let accumulated = textureLoad(accumulation, coords);
    
    // Running average formula: new_avg = old_avg + (sample - old_avg) / (n + 1)
    // where n is the number of samples already accumulated
    let n = f32(params.sample_index);
    let weight = 1.0 / (n + 1.0);
    
    // For first sample (index 0), this simplifies to: result = current
    // For subsequent samples, it correctly averages
    let result = accumulated + (current - accumulated) * weight;
    
    textureStore(accumulation, coords, result);
}

// Copy accumulation buffer to output (for final readback)
// This is a simple copy since accumulation already contains the average
@compute @workgroup_size(8, 8)
fn finalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let coords = vec2<i32>(i32(x), i32(y));
    
    // Load final accumulated average
    let result = textureLoad(accumulation, coords);
    
    // Store to output (accumulation buffer itself serves as output)
    textureStore(accumulation, coords, result);
}
