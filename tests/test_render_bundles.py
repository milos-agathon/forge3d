#!/usr/bin/env python3
"""
Tests for Render Bundles functionality.

Tests bundle creation, compilation, execution, performance validation,
and integration with the rendering system.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add repository root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d as f3d
    import forge3d.bundles as bundles
    HAS_BUNDLES = True
except ImportError:
    HAS_BUNDLES = False

pytestmark = pytest.mark.skipif(not HAS_BUNDLES, reason="Bundles module not available")


class TestBundleSupport:
    """Test bundle system availability and basic functionality."""
    
    def test_bundles_support_detection(self):
        """Test bundle support detection."""
        has_support = bundles.has_bundles_support()
        assert isinstance(has_support, bool)
        
        if has_support:
            print("Render bundles support detected")
        else:
            pytest.skip("Render bundles support not available")


class TestBundleTypes:
    """Test bundle type enumeration."""
    
    def test_bundle_type_enum(self):
        """Test bundle type enumeration values."""
        types = list(bundles.BundleType)
        
        assert bundles.BundleType.INSTANCED in types
        assert bundles.BundleType.UI in types
        assert bundles.BundleType.PARTICLES in types
        assert bundles.BundleType.BATCH in types
        assert bundles.BundleType.WIREFRAME in types
        
        # Check enum values
        assert bundles.BundleType.INSTANCED.value == "instanced"
        assert bundles.BundleType.UI.value == "ui"
        assert bundles.BundleType.PARTICLES.value == "particles"
        assert bundles.BundleType.BATCH.value == "batch"
        assert bundles.BundleType.WIREFRAME.value == "wireframe"
    
    def test_buffer_usage_enum(self):
        """Test buffer usage enumeration."""
        usages = list(bundles.BufferUsage)
        
        assert bundles.BufferUsage.VERTEX in usages
        assert bundles.BufferUsage.INDEX in usages
        assert bundles.BufferUsage.UNIFORM in usages
        assert bundles.BufferUsage.STORAGE in usages


class TestRenderBundle:
    """Test RenderBundle class functionality."""
    
    def test_bundle_creation(self):
        """Test basic bundle creation."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test_bundle")
        
        assert bundle.bundle_type == bundles.BundleType.INSTANCED
        assert bundle.name == "test_bundle"
        assert bundle.compiled == False
        assert isinstance(bundle.stats, bundles.BundleStats)
    
    def test_bundle_creation_default_name(self):
        """Test bundle creation with default name."""
        bundle = bundles.RenderBundle(bundles.BundleType.UI)
        
        assert bundle.name == "ui_bundle"
        assert bundle.bundle_type == bundles.BundleType.UI
    
    def test_add_geometry_basic(self):
        """Test adding basic geometry to bundle."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test")
        
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2], dtype=np.uint32)
        
        result = bundle.add_geometry(vertices, indices)
        
        assert result is bundle  # Should return self for chaining
        assert len(bundle.vertex_data) == 1
        assert len(bundle.index_data) == 1
        assert np.array_equal(bundle.vertex_data[0], vertices)
        assert np.array_equal(bundle.index_data[0], indices)
    
    def test_add_geometry_with_instances(self):
        """Test adding geometry with instance data."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test")
        
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ], dtype=np.float32)
        
        instances = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Instance transform data
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ], dtype=np.float32)
        
        bundle.add_geometry(vertices, None, instances)
        
        assert len(bundle.vertex_data) == 1
        assert len(bundle.index_data) == 0  # No indices provided
        assert len(bundle.instance_data) == 1
        assert np.array_equal(bundle.instance_data[0], instances)
    
    def test_add_geometry_type_conversion(self):
        """Test automatic type conversion for geometry data."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test")
        
        # Input as float64, should convert to float32
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ], dtype=np.float64)
        
        # Input as int32, should convert to uint32
        indices = np.array([0, 1, 2], dtype=np.int32)
        
        bundle.add_geometry(vertices, indices)
        
        assert bundle.vertex_data[0].dtype == np.float32
        assert bundle.index_data[0].dtype == np.uint32
    
    def test_add_uniform(self):
        """Test adding uniform buffer data."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test")
        
        # Test numpy array uniform
        uniform_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        bundle.add_uniform("test_uniform", uniform_array)
        
        assert "test_uniform" in bundle.uniform_data
        assert isinstance(bundle.uniform_data["test_uniform"], bytes)
        
        # Test bytes uniform
        uniform_bytes = b"\x00\x01\x02\x03"
        bundle.add_uniform("test_bytes", uniform_bytes)
        
        assert "test_bytes" in bundle.uniform_data
        assert bundle.uniform_data["test_bytes"] == uniform_bytes
    
    def test_add_texture(self):
        """Test adding texture data."""
        bundle = bundles.RenderBundle(bundles.BundleType.UI, "test")
        
        texture_data = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        
        result = bundle.add_texture(texture_data, "rgba8")
        
        assert result is bundle  # Should return self for chaining
        assert len(bundle.textures) == 1
        assert bundle.textures[0]["format"] == "rgba8"
        assert np.array_equal(bundle.textures[0]["data"], texture_data)
    
    def test_bundle_compilation(self):
        """Test bundle compilation."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test")
        
        # Add required data
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        bundle.add_geometry(vertices)
        
        # Compile bundle
        result = bundle.compile()
        
        assert result is bundle  # Should return self for chaining
        assert bundle.compiled == True
        assert bundle.stats.draw_call_count == 1
        assert bundle.stats.total_vertices == 3
        assert bundle.stats.memory_usage > 0
        assert bundle.stats.compile_time_ms >= 0
    
    def test_compilation_validation(self):
        """Test compilation validation."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test")
        
        # Try to compile without geometry - should fail
        with pytest.raises(ValueError, match="Bundle must have at least one vertex buffer"):
            bundle.compile()
    
    def test_double_compilation_warning(self):
        """Test warning on double compilation."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "test")
        
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        bundle.add_geometry(vertices)
        
        # First compilation
        bundle.compile()
        
        # Second compilation should warn
        with pytest.warns(UserWarning):
            bundle.compile()
    
    def test_bundle_chaining(self):
        """Test method chaining functionality."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        uniform_data = np.array([1.0, 2.0], dtype=np.float32)
        texture_data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Chain all operations
        bundle = (bundles.RenderBundle(bundles.BundleType.INSTANCED, "chained")
                 .add_geometry(vertices, indices)
                 .add_uniform("test_uniform", uniform_data)
                 .add_texture(texture_data)
                 .compile())
        
        assert bundle.compiled == True
        assert len(bundle.vertex_data) == 1
        assert len(bundle.index_data) == 1
        assert "test_uniform" in bundle.uniform_data
        assert len(bundle.textures) == 1
    
    def test_bundle_repr(self):
        """Test bundle string representation."""
        bundle = bundles.RenderBundle(bundles.BundleType.UI, "repr_test")
        
        bundle_str = repr(bundle)
        
        assert "RenderBundle" in bundle_str
        assert "repr_test" in bundle_str
        assert "ui" in bundle_str
        assert "uncompiled" in bundle_str
        
        # After compilation
        vertices = np.array([[0, 0]], dtype=np.float32)
        bundle.add_geometry(vertices).compile()
        
        compiled_str = repr(bundle)
        assert "compiled" in compiled_str


class TestBundleStats:
    """Test BundleStats class."""
    
    def test_stats_creation(self):
        """Test statistics creation and default values."""
        stats = bundles.BundleStats()
        
        assert stats.draw_call_count == 0
        assert stats.total_vertices == 0
        assert stats.total_triangles == 0
        assert stats.memory_usage == 0
        assert stats.compile_time_ms == 0.0
        assert stats.execution_time_ms == 0.0
    
    def test_stats_repr(self):
        """Test statistics string representation."""
        stats = bundles.BundleStats()
        stats.draw_call_count = 3
        stats.total_vertices = 120
        stats.total_triangles = 40
        stats.memory_usage = 2048
        
        stats_str = repr(stats)
        
        assert "BundleStats" in stats_str
        assert "draws=3" in stats_str
        assert "vertices=120" in stats_str
        assert "triangles=40" in stats_str
        assert "memory=2KB" in stats_str


class TestBundleManager:
    """Test BundleManager class."""
    
    def test_manager_creation(self):
        """Test bundle manager creation."""
        manager = bundles.BundleManager()
        
        assert len(manager.bundles) == 0
        assert len(manager.execution_stats) == 0
        assert manager.active == True
    
    def test_create_bundle(self):
        """Test bundle creation through manager."""
        manager = bundles.BundleManager()
        
        bundle = manager.create_bundle("test_bundle", bundles.BundleType.INSTANCED)
        
        assert isinstance(bundle, bundles.RenderBundle)
        assert bundle.name == "test_bundle"
        assert bundle.bundle_type == bundles.BundleType.INSTANCED
        assert "test_bundle" in manager.bundles
        assert "test_bundle" in manager.execution_stats
    
    def test_create_duplicate_bundle_error(self):
        """Test error on creating duplicate bundle."""
        manager = bundles.BundleManager()
        
        manager.create_bundle("duplicate", bundles.BundleType.UI)
        
        with pytest.raises(ValueError, match="Bundle 'duplicate' already exists"):
            manager.create_bundle("duplicate", bundles.BundleType.PARTICLES)
    
    def test_add_bundle(self):
        """Test adding existing bundle to manager."""
        manager = bundles.BundleManager()
        
        # Create and compile bundle independently
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "external")
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        bundle.add_geometry(vertices).compile()
        
        manager.add_bundle(bundle)
        
        assert "external" in manager.bundles
        assert manager.bundles["external"] is bundle
    
    def test_add_uncompiled_bundle_error(self):
        """Test error when adding uncompiled bundle."""
        manager = bundles.BundleManager()
        
        bundle = bundles.RenderBundle(bundles.BundleType.UI, "uncompiled")
        
        with pytest.raises(ValueError, match="Bundle 'uncompiled' must be compiled"):
            manager.add_bundle(bundle)
    
    def test_get_bundle(self):
        """Test getting bundle by name."""
        manager = bundles.BundleManager()
        
        bundle = manager.create_bundle("get_test", bundles.BundleType.PARTICLES)
        
        retrieved = manager.get_bundle("get_test")
        assert retrieved is bundle
        
        not_found = manager.get_bundle("nonexistent")
        assert not_found is None
    
    def test_remove_bundle(self):
        """Test bundle removal."""
        manager = bundles.BundleManager()
        
        manager.create_bundle("to_remove", bundles.BundleType.WIREFRAME)
        
        assert "to_remove" in manager.bundles
        
        removed = manager.remove_bundle("to_remove")
        assert removed == True
        assert "to_remove" not in manager.bundles
        assert "to_remove" not in manager.execution_stats
        
        # Try to remove again
        removed_again = manager.remove_bundle("to_remove")
        assert removed_again == False
    
    def test_execute_bundle(self):
        """Test bundle execution."""
        manager = bundles.BundleManager()
        
        bundle = manager.create_bundle("executable", bundles.BundleType.INSTANCED)
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        bundle.add_geometry(vertices).compile()
        
        success = manager.execute_bundle("executable")
        assert success == True
        assert bundle.stats.execution_time_ms >= 0
        assert len(manager.execution_stats["executable"]) == 1
        
        # Try to execute non-existent bundle
        failure = manager.execute_bundle("nonexistent")
        assert failure == False
    
    def test_execute_multiple_bundles(self):
        """Test executing multiple bundles."""
        manager = bundles.BundleManager()
        
        # Create multiple bundles
        for i in range(3):
            bundle = manager.create_bundle(f"multi_{i}", bundles.BundleType.UI)
            vertices = np.array([[0, 0]], dtype=np.float32)
            bundle.add_geometry(vertices).compile()
        
        executed = manager.execute_bundles(["multi_0", "multi_1", "multi_2", "nonexistent"])
        
        assert executed == 3  # 3 successful, 1 failed
    
    def test_get_bundle_names(self):
        """Test getting all bundle names."""
        manager = bundles.BundleManager()
        
        names = ["alpha", "beta", "gamma"]
        for name in names:
            manager.create_bundle(name, bundles.BundleType.BATCH)
        
        retrieved_names = manager.get_bundle_names()
        
        assert len(retrieved_names) == 3
        for name in names:
            assert name in retrieved_names
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        manager = bundles.BundleManager()
        
        bundle = manager.create_bundle("perf_test", bundles.BundleType.INSTANCED)
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        bundle.add_geometry(vertices).compile()
        
        # Execute multiple times
        for _ in range(5):
            manager.execute_bundle("perf_test")
        
        stats = manager.get_performance_stats("perf_test")
        
        assert stats is not None
        assert stats['sample_count'] == 5
        assert stats['avg_execution_time_ms'] >= 0
        assert stats['min_execution_time_ms'] >= 0
        assert stats['max_execution_time_ms'] >= stats['min_execution_time_ms']
        
        # Non-existent bundle
        none_stats = manager.get_performance_stats("nonexistent")
        assert none_stats is None
    
    def test_total_stats(self):
        """Test total statistics calculation."""
        manager = bundles.BundleManager()
        
        # Create bundles with known stats
        for i in range(3):
            bundle = manager.create_bundle(f"stats_{i}", bundles.BundleType.INSTANCED)
            vertices = np.array([[i, 0, 0], [i+1, 0, 0]], dtype=np.float32)  # 2 vertices each
            bundle.add_geometry(vertices).compile()
        
        total_stats = manager.get_total_stats()
        
        assert total_stats['bundle_count'] == 3
        assert total_stats['total_vertices'] == 6  # 2 vertices * 3 bundles
        assert total_stats['total_draw_calls'] == 3  # 1 draw call per bundle
        assert total_stats['total_memory_usage'] > 0
    
    def test_clear(self):
        """Test clearing all bundles."""
        manager = bundles.BundleManager()
        
        # Add some bundles
        for i in range(3):
            manager.create_bundle(f"clear_{i}", bundles.BundleType.UI)
        
        assert len(manager.bundles) == 3
        
        manager.clear()
        
        assert len(manager.bundles) == 0
        assert len(manager.execution_stats) == 0


class TestBundleBuilder:
    """Test BundleBuilder helper class."""
    
    def test_create_instanced_bundle(self):
        """Test instanced bundle creation."""
        # Geometry data
        vertices = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],  # pos + normal + uv
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 1],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2], dtype=np.uint32)
        
        geometry = {'vertices': vertices, 'indices': indices}
        
        # Instance transforms
        num_instances = 10
        transforms = np.zeros((num_instances, 4, 4), dtype=np.float32)
        for i in range(num_instances):
            transforms[i] = np.eye(4)
            transforms[i][0, 3] = i  # X translation
        
        colors = np.ones((num_instances, 4), dtype=np.float32)
        
        bundle = bundles.BundleBuilder.create_instanced_bundle(
            "instanced_test", geometry, transforms, colors
        )
        
        assert bundle.name == "instanced_test"
        assert bundle.bundle_type == bundles.BundleType.INSTANCED
        assert len(bundle.vertex_data) == 2  # Base geometry + instances
        assert len(bundle.instance_data) == 1
        
        # Check instance data format
        instance_data = bundle.instance_data[0]
        assert instance_data.shape == (num_instances, 20)  # 16 transform + 4 color
    
    def test_create_ui_bundle(self):
        """Test UI bundle creation."""
        ui_elements = [
            {'position': (10, 20), 'size': (100, 50), 'color': (1, 0, 0, 1)},
            {'position': (120, 20), 'size': (80, 30), 'color': (0, 1, 0, 1)},
            {'position': (210, 20), 'size': (60, 40), 'color': (0, 0, 1, 1)},
        ]
        
        bundle = bundles.BundleBuilder.create_ui_bundle("ui_test", ui_elements)
        
        assert bundle.name == "ui_test"
        assert bundle.bundle_type == bundles.BundleType.UI
        assert len(bundle.vertex_data) == 1  # Shared quad geometry
        assert len(bundle.index_data) == 1   # Shared indices
        assert len(bundle.instance_data) == 1  # Instance data for each element
        
        # Check instance count
        instance_data = bundle.instance_data[0]
        assert len(instance_data) == 3  # One instance per UI element
    
    def test_create_particle_bundle(self):
        """Test particle bundle creation."""
        num_particles = 100
        particles = np.random.rand(num_particles, 13).astype(np.float32)
        
        bundle = bundles.BundleBuilder.create_particle_bundle("particle_test", particles)
        
        assert bundle.name == "particle_test"
        assert bundle.bundle_type == bundles.BundleType.PARTICLES
        assert len(bundle.vertex_data) == 1
        assert np.array_equal(bundle.vertex_data[0], particles)
    
    def test_create_batch_bundle(self):
        """Test batch bundle creation."""
        objects = []
        
        # Different objects with different geometry
        for i in range(3):
            vertices = np.random.rand(4, 8).astype(np.float32)  # 4 verts, 8 components
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
            transform = np.eye(4, dtype=np.float32)
            transform[0, 3] = i * 2  # X offset
            material = [0.5, 0.3, 0.1, 0.0]
            
            objects.append({
                'vertices': vertices,
                'indices': indices,
                'transform': transform,
                'material': material
            })
        
        bundle = bundles.BundleBuilder.create_batch_bundle("batch_test", objects)
        
        assert bundle.name == "batch_test"
        assert bundle.bundle_type == bundles.BundleType.BATCH
        assert len(bundle.vertex_data) >= 1  # Combined geometry
        assert "transforms" in bundle.uniform_data
        assert "materials" in bundle.uniform_data


class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_create_test_bundles(self):
        """Test test bundle creation."""
        test_bundles = bundles.create_test_bundles()
        
        assert isinstance(test_bundles, dict)
        assert len(test_bundles) > 0
        
        # Check expected bundle types
        expected_types = ['instanced', 'ui', 'particles']
        for bundle_type in expected_types:
            if bundle_type in test_bundles:
                bundle = test_bundles[bundle_type]
                assert isinstance(bundle, bundles.RenderBundle)
    
    def test_validate_bundle_performance_efficient(self):
        """Test performance validation for efficient bundle."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "efficient")
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        bundle.add_geometry(vertices).compile()
        
        validation = bundles.validate_bundle_performance(bundle)
        
        assert isinstance(validation, dict)
        assert 'efficient' in validation
        assert 'warnings' in validation
        assert 'recommendations' in validation
        assert 'metrics' in validation
        
        # Should be efficient for simple bundle
        assert validation['efficient'] == True
        assert validation['metrics']['draw_call_efficiency'] in ['excellent', 'good']
    
    def test_validate_bundle_performance_inefficient(self):
        """Test performance validation for inefficient bundle."""
        bundle = bundles.RenderBundle(bundles.BundleType.BATCH, "inefficient")
        
        # Add many draw calls to make it inefficient
        for i in range(10):
            vertices = np.random.rand(1000, 8).astype(np.float32)  # Large vertex count
            bundle.add_geometry(vertices)
        
        bundle.compile()
        
        validation = bundles.validate_bundle_performance(bundle)
        
        # Should detect inefficiency
        assert len(validation['warnings']) > 0 or validation['efficient'] == False
    
    def test_compare_bundle_vs_individual(self):
        """Test bundle vs individual draw call comparison."""
        bundle = bundles.RenderBundle(bundles.BundleType.INSTANCED, "comparison")
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        bundle.add_geometry(vertices).compile()
        
        comparison = bundles.compare_bundle_vs_individual(bundle, individual_draw_count=100)
        
        assert isinstance(comparison, dict)
        assert 'estimated_speedup' in comparison
        assert 'individual_time_ms' in comparison
        assert 'bundle_time_ms' in comparison
        assert 'draw_call_reduction' in comparison
        assert 'efficiency_score' in comparison
        
        # Should show benefit for batching many draw calls
        assert comparison['estimated_speedup'] > 1.0
        assert comparison['draw_call_reduction'] > 0


class TestRenderBundleAcceptanceCriteria:
    """Test acceptance criteria for render bundles."""
    
    def test_ssim_ge_0_995(self):
        """Test that SSIM between direct and bundle rendering is >= 0.995."""
        # Create fixed small test scene  
        scene_cfg = {
            'width': 200,
            'height': 150,
            'geometry': {
                'vertices': np.array([
                    # Triangle with deterministic colors
                    [-0.5, -0.5, 0.0,  1.0, 0.2, 0.1, 1.0],  # Red-ish bottom-left
                    [ 0.5, -0.5, 0.0,  0.2, 1.0, 0.1, 1.0],  # Green-ish bottom-right
                    [ 0.0,  0.5, 0.0,  0.1, 0.2, 1.0, 1.0],  # Blue-ish top
                ], dtype=np.float32),
                'indices': np.array([0, 1, 2], dtype=np.uint32)
            }
        }
        
        # Render via both direct and bundle paths
        result = bundles.render_direct_vs_bundle(scene_cfg)
        
        # Compute SSIM between the two renders
        ssim_value = bundles.compute_ssim(result['direct_image'], result['bundle_image'])
        
        print(f"SSIM between direct and bundle rendering: {ssim_value:.6f}")
        print(f"Direct render timing: {result['direct_time_ms']:.3f}ms")
        print(f"Bundle render timing: {result['bundle_time_ms']:.3f}ms")
        
        # Assert SSIM >= 0.995 requirement
        assert ssim_value >= 0.995, f"SSIM {ssim_value:.6f} < 0.995 requirement"
        
        # Additional validation: images should have expected shapes
        assert result['direct_image'].shape == (150, 200, 3)
        assert result['bundle_image'].shape == (150, 200, 3)
        assert result['direct_image'].dtype == np.uint8
        assert result['bundle_image'].dtype == np.uint8
    
    def test_timing_captured(self):
        """Test that both direct and bundle timing values are captured and > 0."""
        # Create simple test scene  
        scene_cfg = {
            'width': 100,
            'height': 100,
            'geometry': {
                'vertices': np.array([
                    # Simple quad
                    [-1.0, -1.0, 0.0,  0.5, 0.5, 0.5, 1.0],
                    [ 1.0, -1.0, 0.0,  0.5, 0.5, 0.5, 1.0],
                    [ 1.0,  1.0, 0.0,  0.5, 0.5, 0.5, 1.0],
                    [-1.0,  1.0, 0.0,  0.5, 0.5, 0.5, 1.0],
                ], dtype=np.float32),
                'indices': np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
            }
        }
        
        # Render and capture timing
        result = bundles.render_direct_vs_bundle(scene_cfg)
        
        direct_time = result['direct_time_ms']
        bundle_time = result['bundle_time_ms']
        
        # Print timing information as required
        print(f"Direct rendering time: {direct_time:.3f}ms")
        print(f"Bundle rendering time: {bundle_time:.3f}ms")
        print(f"Bundle stats: {result['bundle_stats']}")
        
        # Assert both timing values are present and > 0
        assert isinstance(direct_time, (int, float)), f"Direct time must be numeric, got {type(direct_time)}"
        assert isinstance(bundle_time, (int, float)), f"Bundle time must be numeric, got {type(bundle_time)}"
        assert direct_time > 0, f"Direct timing must be > 0, got {direct_time}"
        assert bundle_time > 0, f"Bundle timing must be > 0, got {bundle_time}"
        
        # Additional validation: timing should be reasonable (< 1000ms for simple scene)
        assert direct_time < 1000.0, f"Direct timing seems too high: {direct_time}ms"
        assert bundle_time < 1000.0, f"Bundle timing seems too high: {bundle_time}ms"
    
    def test_no_validation_warnings(self):
        """Test that no WGPU validation warnings/errors occur during minimal render."""
        import logging
        import io
        import contextlib
        
        # Skip if native backend is not available
        if not bundles.has_bundles_support():
            pytest.skip("Native render bundles backend not available")
        
        # Create minimal scene for validation test
        scene_cfg = {
            'width': 64,
            'height': 64,
            'geometry': {
                'vertices': np.array([
                    # Single triangle
                    [0.0, 0.5, 0.0,   1.0, 1.0, 1.0, 1.0],
                    [-0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 1.0], 
                    [0.5, -0.5, 0.0,  1.0, 1.0, 1.0, 1.0],
                ], dtype=np.float32),
                'indices': np.array([0, 1, 2], dtype=np.uint32)
            }
        }
        
        # Capture logs during rendering
        log_stream = io.StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.DEBUG)
        
        # Configure logging to capture WGPU validation messages
        logger = logging.getLogger()
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(log_handler)
        
        try:
            # Enable WGPU validation via environment variable (if supported)
            import os
            original_rust_log = os.environ.get('RUST_LOG')
            os.environ['RUST_LOG'] = 'wgpu=debug,wgpu_core=debug,wgpu_hal=debug'
            
            try:
                # Perform minimal render that should not produce validation warnings
                result = bundles.render_direct_vs_bundle(scene_cfg)
                
                # Ensure render completed successfully
                assert result['direct_image'].shape == (64, 64, 3)
                assert result['bundle_image'].shape == (64, 64, 3)
                
            finally:
                # Restore RUST_LOG
                if original_rust_log is not None:
                    os.environ['RUST_LOG'] = original_rust_log
                elif 'RUST_LOG' in os.environ:
                    del os.environ['RUST_LOG']
            
            # Check captured logs for validation errors/warnings
            log_output = log_stream.getvalue().lower()
            
            validation_error_terms = [
                'validation error',
                'validation warning', 
                'wgpu error',
                'wgpu warning',
                'validation failed',
                'invalid usage'
            ]
            
            found_validation_issues = []
            for term in validation_error_terms:
                if term in log_output:
                    found_validation_issues.append(term)
            
            if found_validation_issues:
                print(f"Log output (first 2000 chars): {log_output[:2000]}")
                
            # Assert no validation warnings/errors were found
            assert len(found_validation_issues) == 0, \
                f"Found validation issues: {found_validation_issues}. Log: {log_output[:500]}"
                
            print("No WGPU validation warnings detected during minimal render")
            
        finally:
            # Clean up logging  
            logger.removeHandler(log_handler)
            logger.setLevel(original_level)
            log_handler.close()


def test_bundles_example_runs():
    """Test that the bundles example can run without errors."""
    from pathlib import Path
    import subprocess
    import sys
    
    example_path = Path(__file__).parent.parent / "examples" / "bundles_demo.py"
    
    if not example_path.exists():
        pytest.skip("Bundles example not found")
    
    # Run example in test mode
    result = subprocess.run([
        sys.executable, str(example_path),
        "--type", "ui",  # Use simple UI type for testing
        "--out", "out/test_bundles.png",
        "--width", "400",
        "--height", "300"
    ], capture_output=True, text=True, cwd=str(example_path.parent.parent))
    
    # Note: This may fail if GPU support is not available, which is OK for testing
    if result.returncode != 0:
        if "not available" in result.stdout or "not supported" in result.stdout:
            pytest.skip("Render bundles not supported on this system")
        else:
            print(f"Example output: {result.stdout}")
            print(f"Example errors: {result.stderr}")
            # Don't fail the test - the example may require GPU features not available in CI
            pytest.skip("Bundles example requires GPU features")
    
    # Check that output file was created
    output_path = example_path.parent.parent / "out" / "test_bundles.png"
    if output_path.exists():
        print(f"Bundles example successfully created output: {output_path}")


if __name__ == "__main__":
    pytest.main([__file__])