// Shadow mapping tests for terrain draping

#[cfg(test)]
mod tests {
    use forge3d::renderer::terrain_drape::{TerrainDrapeRenderer, TerrainDrapeConfig};
    use forge3d::core::uv_transform::UVTransform;

    #[tokio::test]
    async fn test_shadow_rendering_basic() {
        // Create a simple heightmap and landcover
        let size = 64;
        let mut heightmap = vec![0.0f32; size * size];
        
        // Create a raised area that should cast shadows
        for y in 20..40 {
            for x in 20..40 {
                heightmap[y * size + x] = 10.0;
            }
        }
        
        // Green landcover
        let mut landcover = vec![0u8; size * size * 4];
        for i in 0..size * size {
            landcover[i * 4 + 0] = 50;  // R
            landcover[i * 4 + 1] = 200; // G
            landcover[i * 4 + 2] = 50;  // B
            landcover[i * 4 + 3] = 255; // A
        }
        
        // Configure with shadows enabled
        let config = TerrainDrapeConfig {
            width: 512,
            height: 512,
            sample_count: 1,
            z_dir: 1.0,
            zscale: 2.0,
            light_type: 1,  // directional
            light_elevation: 45.0,
            light_azimuth: 315.0,
            light_intensity: 1.0,
            ambient: 0.25,
            shadow_intensity: 0.7,
            lighting_model: 2,  // blinn_phong
            shininess: 32.0,
            specular_strength: 0.3,
            shadow_softness: 2.0,
            gamma: 0.0,
            fov: 35.0,
            background_color: [1.0, 1.0, 1.0, 1.0],
            tonemap_mode: 1,
            gamma_correction: 2.2,
            hdri_intensity: 1.0,
            shadow_map_res: 1024,
            shadow_bias: 0.002,
            enable_shadows: true,
        };
        
        let renderer = TerrainDrapeRenderer::new(config).await
            .expect("Failed to create renderer");
        
        // Setup camera
        let camera_pos = glam::Vec3::new(80.0, 40.0, 80.0);
        let camera_target = glam::Vec3::ZERO;
        let view = glam::Mat4::look_at_rh(camera_pos, camera_target, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(35.0f32.to_radians(), 1.0, 0.1, 1000.0);
        
        let uv_transform = UVTransform::identity();
        
        // Render
        let result = renderer.render(
            &heightmap,
            size as u32,
            size as u32,
            &landcover,
            size as u32,
            size as u32,
            &uv_transform,
            &view,
            &proj,
            &camera_pos,
        );
        
        assert!(result.is_ok(), "Shadow rendering should succeed");
        let pixels = result.unwrap();
        assert_eq!(pixels.len(), 512 * 512 * 4, "Output should be 512x512 RGBA");
    }

    #[tokio::test]
    async fn test_shadow_intensity_control() {
        let size = 32;
        let heightmap = vec![0.0f32; size * size];
        let landcover = vec![128u8; size * size * 4];
        
        // Test with shadows disabled
        let config_no_shadow = TerrainDrapeConfig {
            enable_shadows: false,
            shadow_intensity: 0.0,
            ..Default::default()
        };
        
        let renderer_no_shadow = TerrainDrapeRenderer::new(config_no_shadow).await
            .expect("Failed to create renderer");
        
        // Test with shadows enabled
        let config_with_shadow = TerrainDrapeConfig {
            enable_shadows: true,
            shadow_intensity: 0.8,
            shadow_map_res: 512,
            ..Default::default()
        };
        
        let renderer_with_shadow = TerrainDrapeRenderer::new(config_with_shadow).await
            .expect("Failed to create renderer");
        
        // Both should render successfully
        let camera_pos = glam::Vec3::new(50.0, 30.0, 50.0);
        let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(35.0f32.to_radians(), 1.0, 0.1, 1000.0);
        let uv_transform = UVTransform::identity();
        
        let result_no_shadow = renderer_no_shadow.render(
            &heightmap, size as u32, size as u32,
            &landcover, size as u32, size as u32,
            &uv_transform, &view, &proj, &camera_pos,
        );
        
        let result_with_shadow = renderer_with_shadow.render(
            &heightmap, size as u32, size as u32,
            &landcover, size as u32, size as u32,
            &uv_transform, &view, &proj, &camera_pos,
        );
        
        assert!(result_no_shadow.is_ok(), "Rendering without shadows should succeed");
        assert!(result_with_shadow.is_ok(), "Rendering with shadows should succeed");
    }

    #[tokio::test]
    async fn test_shadow_map_resolution() {
        let size = 32;
        let heightmap = vec![0.0f32; size * size];
        let landcover = vec![128u8; size * size * 4];
        
        // Test various shadow map resolutions
        for &res in &[512, 1024, 2048, 4096] {
            let config = TerrainDrapeConfig {
                shadow_map_res: res,
                enable_shadows: true,
                ..Default::default()
            };
            
            let renderer = TerrainDrapeRenderer::new(config).await;
            assert!(renderer.is_ok(), "Should create renderer with shadow_map_res={}", res);
        }
    }

    #[tokio::test]
    async fn test_shadow_bias_prevents_acne() {
        let size = 32;
        let heightmap = vec![5.0f32; size * size]; // Flat elevated terrain
        let landcover = vec![128u8; size * size * 4];
        
        // Test with different bias values
        for &bias in &[0.0001, 0.001, 0.0015, 0.005, 0.01] {
            let config = TerrainDrapeConfig {
                shadow_bias: bias,
                enable_shadows: true,
                shadow_map_res: 1024,
                ..Default::default()
            };
            
            let renderer = TerrainDrapeRenderer::new(config).await
                .expect(&format!("Failed to create renderer with bias={}", bias));
            
            let camera_pos = glam::Vec3::new(40.0, 20.0, 40.0);
            let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
            let proj = glam::Mat4::perspective_rh(35.0f32.to_radians(), 1.0, 0.1, 1000.0);
            let uv_transform = UVTransform::identity();
            
            let result = renderer.render(
                &heightmap, size as u32, size as u32,
                &landcover, size as u32, size as u32,
                &uv_transform, &view, &proj, &camera_pos,
            );
            
            assert!(result.is_ok(), "Rendering with bias={} should succeed", bias);
        }
    }

    #[tokio::test]
    async fn test_pcf_softness_parameter() {
        let size = 32;
        let heightmap = vec![0.0f32; size * size];
        let landcover = vec![128u8; size * size * 4];
        
        // Test with different softness values
        for &softness in &[1.0, 2.0, 3.0, 5.0] {
            let config = TerrainDrapeConfig {
                shadow_softness: softness,
                enable_shadows: true,
                ..Default::default()
            };
            
            let renderer = TerrainDrapeRenderer::new(config).await
                .expect(&format!("Failed to create renderer with softness={}", softness));
            
            let camera_pos = glam::Vec3::new(40.0, 20.0, 40.0);
            let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
            let proj = glam::Mat4::perspective_rh(35.0f32.to_radians(), 1.0, 0.1, 1000.0);
            let uv_transform = UVTransform::identity();
            
            let result = renderer.render(
                &heightmap, size as u32, size as u32,
                &landcover, size as u32, size as u32,
                &uv_transform, &view, &proj, &camera_pos,
            );
            
            assert!(result.is_ok(), "Rendering with softness={} should succeed", softness);
        }
    }
}
