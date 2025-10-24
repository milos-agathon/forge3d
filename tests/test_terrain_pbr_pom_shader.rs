// Test for terrain PBR+POM shader compilation
// Verifies that the WGSL shader compiles without errors

#[cfg(test)]
mod tests {
    use wgpu;

    #[test]
    fn test_terrain_pbr_pom_shader_compiles() {
        // Create a minimal wgpu instance for shader compilation testing
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find a suitable GPU adapter");

        // Request device
        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create device");

        // Load the shader source
        let shader_source = include_str!("../src/shaders/terrain_pbr_pom.wgsl");

        // Attempt to compile the shader
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain PBR+POM Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // If we get here without panicking, the shader compiled successfully
        println!("Terrain PBR+POM shader compiled successfully!");

        // Verify the shader has the expected entry points
        // Note: wgpu doesn't expose entry point introspection, so we just verify compilation
        drop(shader_module);
    }

    #[test]
    fn test_shader_has_required_functions() {
        // Verify the shader source contains the required functions
        let shader_source = include_str!("../src/shaders/terrain_pbr_pom.wgsl");

        // Task 4.1: Normal calculation
        assert!(
            shader_source.contains("fn calculate_normal"),
            "Shader missing calculate_normal function (Task 4.1)"
        );

        // Task 4.2: Triplanar sampling
        assert!(
            shader_source.contains("fn sample_triplanar"),
            "Shader missing sample_triplanar function (Task 4.2)"
        );
        assert!(
            shader_source.contains("fn sample_triplanar_normal"),
            "Shader missing sample_triplanar_normal function (Task 4.2)"
        );

        // Task 4.3: Parallax Occlusion Mapping
        assert!(
            shader_source.contains("fn parallax_occlusion_mapping"),
            "Shader missing parallax_occlusion_mapping function (Task 4.3)"
        );
        assert!(
            shader_source.contains("fn pom_self_shadow"),
            "Shader missing pom_self_shadow function (Task 4.3)"
        );

        // Task 4.4: PBR BRDF
        assert!(
            shader_source.contains("fn distribution_ggx"),
            "Shader missing distribution_ggx function (Task 4.4)"
        );
        assert!(
            shader_source.contains("fn geometry_smith"),
            "Shader missing geometry_smith function (Task 4.4)"
        );
        assert!(
            shader_source.contains("fn fresnel_schlick"),
            "Shader missing fresnel_schlick function (Task 4.4)"
        );
        assert!(
            shader_source.contains("fn calculate_pbr_brdf"),
            "Shader missing calculate_pbr_brdf function (Task 4.4)"
        );

        // Entry points
        assert!(
            shader_source.contains("@vertex") && shader_source.contains("fn vs_main"),
            "Shader missing vertex shader entry point"
        );
        assert!(
            shader_source.contains("@fragment") && shader_source.contains("fn fs_main"),
            "Shader missing fragment shader entry point"
        );

        println!("All required shader functions present!");
    }

    #[test]
    fn test_shader_bind_groups() {
        // Verify the shader defines all required bind groups
        let shader_source = include_str!("../src/shaders/terrain_pbr_pom.wgsl");

        // Bind group 0: Globals
        assert!(
            shader_source.contains("@group(0)"),
            "Shader missing bind group 0 (globals)"
        );

        // Bind group 1: Height map
        assert!(
            shader_source.contains("@group(1)"),
            "Shader missing bind group 1 (height map)"
        );

        // Bind group 2: Colormap LUT
        assert!(
            shader_source.contains("@group(2)"),
            "Shader missing bind group 2 (colormap)"
        );

        // Bind group 3: Material textures
        assert!(
            shader_source.contains("@group(3)"),
            "Shader missing bind group 3 (materials)"
        );

        // Bind group 4: Triplanar & POM params
        assert!(
            shader_source.contains("@group(4)"),
            "Shader missing bind group 4 (params)"
        );

        // Bind group 5: IBL environment maps
        assert!(
            shader_source.contains("@group(5)"),
            "Shader missing bind group 5 (IBL)"
        );

        // Bind group 6: Shadow map
        assert!(
            shader_source.contains("@group(6)"),
            "Shader missing bind group 6 (shadows)"
        );

        println!("All required bind groups present!");
    }
}
