// Big buffer WGSL utilities for per-object data indexing (I7)
//
// Provides structs and helpers for accessing per-object data stored
// in a large storage buffer using dynamic offset or index addressing.

// Per-object data block (64 bytes, aligned for WGSL std140)
struct ObjectData {
    // Transform matrix (64 bytes = 4x vec4)
    transform: mat4x4<f32>,
    // Additional per-object parameters can be added here
    // Total size must remain 64 bytes for alignment
};

// Array of object data in storage buffer
struct BigBufferData {
    objects: array<ObjectData>,
};

// Bind group layout for big buffer
// @group(1) @binding(0) 
// var<storage, read> big_buffer_data: BigBufferData;

// Helper function to get object data by index
fn get_object_data(buffer: ptr<storage, BigBufferData, read>, index: u32) -> ObjectData {
    return buffer.objects[index];
}

// Helper function to get transform matrix by index  
fn get_object_transform(buffer: ptr<storage, BigBufferData, read>, index: u32) -> mat4x4<f32> {
    return buffer.objects[index].transform;
}

// Usage example in vertex shader:
// 
// @vertex
// fn vs_main(
//     @builtin(vertex_index) vertex_index: u32,
//     @builtin(instance_index) instance_index: u32,
//     @location(0) position: vec3<f32>
// ) -> @builtin(position) vec4<f32> {
//     let object_data = get_object_data(&big_buffer_data, instance_index);
//     let world_pos = object_data.transform * vec4<f32>(position, 1.0);
//     return globals.view_proj * world_pos;
// }

// Dynamic offset addressing (alternative to indexing)
// When using dynamic offsets, bind the buffer at a specific offset:
//
// render_pass.set_bind_group(1, &bind_group, &[dynamic_offset]);
//
// Where dynamic_offset = block.offset (from BigBufferBlock)

// Constants matching Rust side
const BIG_BUFFER_BLOCK_SIZE: u32 = 64u;