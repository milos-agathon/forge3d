"""
TBN (Tangent-Bitangent-Normal) vertex layout binding validation test

This test validates that the TBN vertex buffer layout binds correctly
without wgpu validation errors during render pipeline creation using Python.
"""

import pytest
import asyncio

try:
    import wgpu
    import wgpu.utils
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False


@pytest.mark.asyncio
async def test_tbn_vertex_layout_binding():
    """Test that TBN vertex layout with stride 56 binds without validation errors"""
    
    if not WGPU_AVAILABLE:
        pytest.skip("wgpu-py not available - install with: pip install wgpu")
    
    try:
        # Get headless device using wgpu.utils.get_default_device
        device = wgpu.utils.get_default_device()
        if device is None:
            pytest.skip("No wgpu device available for headless testing")
    
    except Exception as e:
        pytest.skip(f"Failed to initialize wgpu: {e}")
    
    # Note: wgpu-py doesn't expose push_error_scope/pop_error_scope
    # If there are validation errors, pipeline creation will fail with an exception
    
    # Define TBN vertex buffer layout with stride 56
    vertex_buffer_layout = {
        "array_stride": 56,  # Total size of TBN vertex
        "step_mode": wgpu.VertexStepMode.vertex,
        "attributes": [
            # Position: Float32x3 at location 0, offset 0
            {
                "offset": 0,
                "shader_location": 0,
                "format": wgpu.VertexFormat.float32x3,
            },
            # UV: Float32x2 at location 1, offset 12
            {
                "offset": 12,
                "shader_location": 1,
                "format": wgpu.VertexFormat.float32x2,
            },
            # Normal: Float32x3 at location 2, offset 20
            {
                "offset": 20,
                "shader_location": 2,
                "format": wgpu.VertexFormat.float32x3,
            },
            # Tangent: Float32x3 at location 3, offset 32
            {
                "offset": 32,
                "shader_location": 3,
                "format": wgpu.VertexFormat.float32x3,
            },
            # Bitangent: Float32x3 at location 4, offset 44
            {
                "offset": 44,
                "shader_location": 4,
                "format": wgpu.VertexFormat.float32x3,
            },
        ],
    }
    
    # Minimal WGSL shader consuming locations 0..4
    shader_source = """
        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) uv: vec2<f32>,
            @location(2) normal: vec3<f32>,
            @location(3) tangent: vec3<f32>,
            @location(4) bitangent: vec3<f32>,
        }

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec3<f32>,
        }

        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4<f32>(input.position, 1.0);
            
            // Use TBN vectors to create a simple color
            output.color = normalize(input.normal) * 0.5 + 0.5;
            
            return output;
        }

        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            return vec4<f32>(input.color, 1.0);
        }
    """
    
    shader_module = device.create_shader_module(code=shader_source)
    
    # Create render pipeline with rgba8unorm color target
    render_pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={
            "module": shader_module,
            "entry_point": "vs_main",
            "buffers": [vertex_buffer_layout],
        },
        fragment={
            "module": shader_module,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": wgpu.TextureFormat.rgba8unorm,
                    "blend": {
                        "color": {
                            "operation": wgpu.BlendOperation.add,
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.zero,
                        },
                        "alpha": {
                            "operation": wgpu.BlendOperation.add,
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.zero,
                        },
                    },
                    "write_mask": wgpu.ColorWrite.ALL,
                }
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.back,
        },
        depth_stencil=None,
        multisample={
            "count": 1,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        },
    )
    
    # If we reach here, pipeline creation succeeded without validation errors
    print("wgpu_validation=None")
    
    # Verify pipeline was created successfully
    assert render_pipeline is not None, "Failed to create render pipeline"