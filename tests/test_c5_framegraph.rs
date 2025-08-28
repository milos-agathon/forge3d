//! Tests for C5: Frame graph with transient aliasing
//!
//! Validates resource lifetime tracking, transient aliasing, and barrier generation.

use forge3d::core::framegraph_impl::{FrameGraph, ResourceDesc, ResourceType, PassType};
use wgpu::{TextureFormat, Extent3d, TextureUsages};

#[test]
fn test_c5_framegraph_basic_graph() {
    let mut graph = FrameGraph::new();
    
    // Create resources
    let scene_color = graph.add_resource(ResourceDesc {
        name: "scene_color".to_string(),
        resource_type: ResourceType::ColorAttachment,
        format: Some(TextureFormat::Rgba16Float),
        extent: Some(Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        }),
        size: None,
        usage: Some(TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING),
        can_alias: true,
    });
    
    let final_color = graph.add_resource(ResourceDesc {
        name: "final_color".to_string(),
        resource_type: ResourceType::ColorAttachment,
        format: Some(TextureFormat::Rgba8UnormSrgb),
        extent: Some(Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        }),
        size: None,
        usage: Some(TextureUsages::RENDER_ATTACHMENT),
        can_alias: false, // Final output can't be aliased
    });
    
    // Add scene pass
    let scene_pass = graph.add_pass("scene", PassType::Graphics, |builder| {
        builder.write(scene_color);
        Ok(())
    }).unwrap();
    
    // Add tonemap pass
    let tonemap_pass = graph.add_pass("tonemap", PassType::Graphics, |builder| {
        builder.read(scene_color).write(final_color);
        Ok(())
    }).unwrap();
    
    // Compile the graph
    graph.compile().unwrap();
    
    // Get execution plan
    let (passes, barriers) = graph.get_execution_plan().unwrap();
    
    // Verify pass order (scene should come before tonemap)
    assert_eq!(passes.len(), 2);
    assert_eq!(passes[0], scene_pass);
    assert_eq!(passes[1], tonemap_pass);
    
    // Check that we have barriers for the resource transition
    assert!(!barriers.is_empty(), "Should have barriers for resource transitions");
    
    let metrics = graph.metrics();
    assert_eq!(metrics.pass_count, 2);
    assert_eq!(metrics.resource_count, 2);
    assert_eq!(metrics.transient_count, 2); // Both resources are transient
}

#[test] 
fn test_c5_framegraph_aliasing() {
    let mut graph = FrameGraph::new();
    
    // Create two resources that don't overlap in lifetime
    let temp1 = graph.add_resource(ResourceDesc {
        name: "temp1".to_string(),
        resource_type: ResourceType::ColorAttachment,
        format: Some(TextureFormat::Rgba16Float),
        extent: Some(Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        }),
        size: None,
        usage: Some(TextureUsages::RENDER_ATTACHMENT),
        can_alias: true,
    });
    
    let temp2 = graph.add_resource(ResourceDesc {
        name: "temp2".to_string(),
        resource_type: ResourceType::ColorAttachment,
        format: Some(TextureFormat::Rgba16Float),
        extent: Some(Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        }),
        size: None,
        usage: Some(TextureUsages::RENDER_ATTACHMENT),
        can_alias: true,
    });
    
    let final_out = graph.add_resource(ResourceDesc {
        name: "final_out".to_string(),
        resource_type: ResourceType::ColorAttachment,
        format: Some(TextureFormat::Rgba8UnormSrgb),
        extent: Some(Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        }),
        size: None,
        usage: Some(TextureUsages::RENDER_ATTACHMENT),
        can_alias: false,
    });
    
    // Pass 1: Use temp1
    graph.add_pass("pass1", PassType::Graphics, |builder| {
        builder.write(temp1);
        Ok(())
    }).unwrap();
    
    // Pass 2: Use temp1, write temp2 (temp1 -> temp2)
    graph.add_pass("pass2", PassType::Graphics, |builder| {
        builder.read(temp1).write(temp2);
        Ok(())
    }).unwrap();
    
    // Pass 3: Use temp2, write final (temp2 -> final)
    graph.add_pass("pass3", PassType::Graphics, |builder| {
        builder.read(temp2).write(final_out);
        Ok(())
    }).unwrap();
    
    // Compile and check aliasing
    graph.compile().unwrap();
    
    let metrics = graph.metrics();
    assert_eq!(metrics.pass_count, 3);
    assert_eq!(metrics.resource_count, 3);
    
    // At least one resource should be aliased (temp1 and temp2 don't overlap after pass2)
    assert!(
        metrics.aliased_count > 0, 
        "Should have aliased at least one resource"
    );
    assert!(
        metrics.memory_saved_bytes > 0,
        "Should report memory savings from aliasing"
    );
}

#[test]
fn test_c5_framegraph_circular_dependency_detection() {
    let mut graph = FrameGraph::new();
    
    let resource_a = graph.add_resource(ResourceDesc {
        name: "resource_a".to_string(),
        resource_type: ResourceType::StorageBuffer,
        format: None,
        extent: None,
        size: Some(1024),
        usage: None,
        can_alias: false,
    });
    
    let resource_b = graph.add_resource(ResourceDesc {
        name: "resource_b".to_string(),
        resource_type: ResourceType::StorageBuffer,
        format: None,
        extent: None,
        size: Some(1024),
        usage: None,
        can_alias: false,
    });
    
    // Create circular dependency: pass1 writes A, reads B; pass2 writes B, reads A
    graph.add_pass("pass1", PassType::Compute, |builder| {
        builder.write(resource_a).read(resource_b);
        Ok(())
    }).unwrap();
    
    graph.add_pass("pass2", PassType::Compute, |builder| {
        builder.write(resource_b).read(resource_a);
        Ok(())
    }).unwrap();
    
    // Compile should succeed (dependency analysis happens later)
    graph.compile().unwrap();
    
    // But execution plan should detect the cycle
    let result = graph.get_execution_plan();
    assert!(
        result.is_err(),
        "Should detect circular dependency in execution plan"
    );
}

#[test]
fn test_c5_framegraph_barrier_generation() {
    let mut graph = FrameGraph::new();
    
    let texture = graph.add_resource(ResourceDesc {
        name: "shared_texture".to_string(),
        resource_type: ResourceType::ColorAttachment,
        format: Some(TextureFormat::Rgba8Unorm),
        extent: Some(Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        }),
        size: None,
        usage: Some(TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING),
        can_alias: false,
    });
    
    // Pass 1: Write to texture as render target
    graph.add_pass("render_pass", PassType::Graphics, |builder| {
        builder.write(texture);
        Ok(())
    }).unwrap();
    
    // Pass 2: Read texture for sampling
    graph.add_pass("sample_pass", PassType::Graphics, |builder| {
        builder.read(texture);
        Ok(())
    }).unwrap();
    
    graph.compile().unwrap();
    let (_passes, barriers) = graph.get_execution_plan().unwrap();
    
    // Should have at least one barrier for the texture transition
    assert!(
        barriers.len() > 0,
        "Should generate barriers for texture usage transition"
    );
    
    // Verify barrier is for our resource
    let has_texture_barrier = barriers.iter().any(|barrier| {
        barrier.resource == texture
    });
    assert!(has_texture_barrier, "Should have barrier for the shared texture");
}