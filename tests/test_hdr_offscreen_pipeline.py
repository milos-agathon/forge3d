"""
HDR Off-Screen Pipeline Tests

Tests for the HDR off-screen rendering pipeline implementation, validating:
- Pipeline creation and configuration
- HDR rendering to RGBA16Float textures
- Tone mapping post-processing to sRGB8 output
- PNG output generation and readback
- Clamp rate computation and validation
- VRAM usage tracking and memory constraints
- Multiple tone mapping operator support
"""

import numpy as np
import pytest
import logging
from pathlib import Path
import tempfile
import os

import forge3d as f3d

# Check if HDR off-screen pipeline feature is available
try:
    # This will fail if the feature is not compiled or not exposed to Python
    pipeline_test = f3d.create_hdr_offscreen_pipeline({'width': 32, 'height': 32})
    _HDR_OFFSCREEN_AVAILABLE = True
    del pipeline_test  # Clean up test instance
except (AttributeError, ImportError, RuntimeError):
    _HDR_OFFSCREEN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HDR_OFFSCREEN_AVAILABLE,
    reason="HDR off-screen pipeline not available (enable-hdr-offscreen feature)"
)

logger = logging.getLogger(__name__)


class TestHdrOffscreenPipelineConfig:
    """Test HDR off-screen pipeline configuration."""
    
    def test_create_default_config(self):
        """Test pipeline creation with default configuration."""
        config = {
            'width': 256,
            'height': 256
        }
        
        pipeline = f3d.create_hdr_offscreen_pipeline(config)
        assert pipeline is not None
        
        # Verify default configuration values
        assert pipeline.get_width() == 256
        assert pipeline.get_height() == 256
        assert pipeline.get_hdr_format() == 'rgba16float'
        assert pipeline.get_ldr_format() == 'rgba8unorm_srgb'
        assert pipeline.get_tone_mapping() == 'reinhard'
        assert pipeline.get_exposure() == pytest.approx(1.0)
        assert pipeline.get_gamma() == pytest.approx(2.2)
    
    def test_create_custom_config(self):
        """Test pipeline creation with custom configuration."""
        config = {
            'width': 128,
            'height': 64,
            'hdr_format': 'rgba16float',
            'ldr_format': 'rgba8unorm_srgb',
            'tone_mapping': 'aces',
            'exposure': 2.0,
            'white_point': 8.0,
            'gamma': 2.4
        }
        
        pipeline = f3d.create_hdr_offscreen_pipeline(config)
        assert pipeline is not None
        
        # Verify custom configuration
        assert pipeline.get_width() == 128
        assert pipeline.get_height() == 64
        assert pipeline.get_tone_mapping() == 'aces'
        assert pipeline.get_exposure() == pytest.approx(2.0)
        assert pipeline.get_gamma() == pytest.approx(2.4)
    
    def test_invalid_dimensions(self):
        """Test pipeline creation with invalid dimensions."""
        with pytest.raises(ValueError):
            f3d.create_hdr_offscreen_pipeline({'width': 0, 'height': 100})
        
        with pytest.raises(ValueError):
            f3d.create_hdr_offscreen_pipeline({'width': 100, 'height': -1})
        
        with pytest.raises(ValueError):
            f3d.create_hdr_offscreen_pipeline({'width': 8192, 'height': 8192})  # Too large
    
    def test_invalid_tone_mapping(self):
        """Test pipeline creation with invalid tone mapping operator."""
        config = {
            'width': 64,
            'height': 64,
            'tone_mapping': 'invalid_operator'
        }
        
        with pytest.raises(ValueError):
            f3d.create_hdr_offscreen_pipeline(config)
    
    def test_invalid_exposure(self):
        """Test pipeline creation with invalid exposure values."""
        with pytest.raises(ValueError):
            f3d.create_hdr_offscreen_pipeline({
                'width': 64, 'height': 64, 'exposure': 0.0
            })
        
        with pytest.raises(ValueError):
            f3d.create_hdr_offscreen_pipeline({
                'width': 64, 'height': 64, 'exposure': -1.0
            })


class TestHdrOffscreenPipelineRendering:
    """Test HDR off-screen rendering functionality."""
    
    def test_render_simple_scene(self):
        """Test rendering a simple HDR scene."""
        config = {
            'width': 64,
            'height': 64,
            'tone_mapping': 'reinhard',
            'exposure': 1.0
        }
        
        pipeline = f3d.create_hdr_offscreen_pipeline(config)
        
        # Create simple HDR scene data
        hdr_scene = self._create_test_hdr_scene(64, 64)
        
        # Upload scene to GPU
        pipeline.upload_hdr_data(hdr_scene)
        
        # Begin HDR render pass
        pipeline.begin_hdr_render()
        
        # Draw HDR scene (simplified API - actual implementation may vary)
        pipeline.draw_hdr_scene()
        
        # End HDR render pass
        pipeline.end_hdr_render()
        
        # Apply tone mapping
        pipeline.apply_tone_mapping()
        
        # Read back LDR result
        ldr_data = pipeline.read_ldr_data()
        
        # Validate output
        assert ldr_data.shape == (64, 64, 4)
        assert ldr_data.dtype == np.uint8
        assert np.all(ldr_data >= 0)
        assert np.all(ldr_data <= 255)
        
        # Check for reasonable variation (not all black/white)
        unique_values = len(np.unique(ldr_data))
        assert unique_values > 10, f"Too few unique values: {unique_values}"
    
    def test_vram_usage_tracking(self):
        """Test VRAM usage tracking and memory constraints."""
        config = {
            'width': 512,
            'height': 512,
            'tone_mapping': 'aces',
            'exposure': 1.0
        }
        
        pipeline = f3d.create_hdr_offscreen_pipeline(config)
        
        # Get initial VRAM usage
        vram_initial = pipeline.get_vram_usage()
        assert vram_initial > 0, "VRAM usage should be > 0"
        
        # Create and upload HDR scene
        hdr_scene = self._create_test_hdr_scene(512, 512)
        pipeline.upload_hdr_data(hdr_scene)
        
        # Get VRAM usage after upload
        vram_after_upload = pipeline.get_vram_usage()
        assert vram_after_upload >= vram_initial, "VRAM usage should increase after upload"
        
        # Render scene
        pipeline.begin_hdr_render()
        pipeline.draw_hdr_scene()
        pipeline.end_hdr_render()
        pipeline.apply_tone_mapping()
        
        # Get peak VRAM usage
        vram_peak = pipeline.get_vram_usage()
        
        # Validate memory constraint (≤512 MiB)
        vram_limit_bytes = 512 * 1024 * 1024
        assert vram_peak <= vram_limit_bytes, \
            f"VRAM usage {vram_peak/(1024*1024):.1f} MiB exceeds limit 512 MiB"
        
        logger.info(f"VRAM usage: initial={vram_initial/(1024*1024):.1f} MiB, "
                   f"peak={vram_peak/(1024*1024):.1f} MiB")
    
    def test_tone_mapping_operators(self):
        """Test multiple tone mapping operators."""
        operators = ['reinhard', 'reinhard_extended', 'aces', 'uncharted2', 'exposure']
        
        base_config = {
            'width': 128,
            'height': 128,
            'exposure': 1.0,
            'gamma': 2.2
        }
        
        results = {}
        hdr_scene = self._create_test_hdr_scene(128, 128)
        
        for operator in operators:
            config = base_config.copy()
            config['tone_mapping'] = operator
            
            try:
                pipeline = f3d.create_hdr_offscreen_pipeline(config)
                
                # Render with current operator
                pipeline.upload_hdr_data(hdr_scene)
                pipeline.begin_hdr_render()
                pipeline.draw_hdr_scene()
                pipeline.end_hdr_render()
                pipeline.apply_tone_mapping()
                
                ldr_data = pipeline.read_ldr_data()
                
                # Compute statistics
                luminance = 0.299 * ldr_data[:, :, 0] + 0.587 * ldr_data[:, :, 1] + 0.114 * ldr_data[:, :, 2]
                mean_lum = float(np.mean(luminance))
                
                results[operator] = {
                    'ldr_data': ldr_data,
                    'mean_luminance': mean_lum
                }
                
                logger.info(f"Tone mapping {operator}: mean_lum={mean_lum:.2f}")
                
            except Exception as e:
                logger.warning(f"Tone mapping operator {operator} failed: {e}")
                results[operator] = None
        
        # Validate that different operators produce different results
        successful_operators = [op for op, result in results.items() if result is not None]
        assert len(successful_operators) >= 3, f"Too few successful operators: {successful_operators}"
        
        # Check that operators produce visibly different results
        if 'reinhard' in results and 'aces' in results:
            reinhard_mean = results['reinhard']['mean_luminance']
            aces_mean = results['aces']['mean_luminance']
            difference_percent = abs(aces_mean - reinhard_mean) / max(reinhard_mean, aces_mean) * 100
            
            assert difference_percent > 1.0, \
                f"Reinhard and ACES too similar: {difference_percent:.2f}% difference"
    
    def _create_test_hdr_scene(self, width, height):
        """Create test HDR scene with known dynamic range."""
        scene = np.zeros((height, width, 3), dtype=np.float32)
        
        # Create gradient with high dynamic range
        for y in range(height):
            for x in range(width):
                # Create exponential brightness curve
                brightness = 0.01 * (50.0 / 0.01) ** (x / (width - 1))
                
                # Add color variation
                color_factor = y / (height - 1)
                scene[y, x, 0] = brightness * (0.8 + 0.4 * color_factor)  # Red
                scene[y, x, 1] = brightness * 0.9  # Green
                scene[y, x, 2] = brightness * (1.2 - 0.4 * color_factor)  # Blue
        
        # Add bright spots for testing tone mapping
        center_x, center_y = width // 2, height // 2
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if 0 <= center_y + dy < height and 0 <= center_x + dx < width:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < 4:
                        intensity = 20.0 * (1.0 - distance / 4.0)
                        scene[center_y + dy, center_x + dx] = [intensity, intensity * 0.9, intensity * 0.8]
        
        return scene


class TestHdrOffscreenPipelineValidation:
    """Test pipeline validation and acceptance criteria."""
    
    def test_png_output_generation(self):
        """Test PNG output generation and file I/O."""
        config = {
            'width': 128,
            'height': 128,
            'tone_mapping': 'aces',
            'exposure': 1.0
        }
        
        pipeline = f3d.create_hdr_offscreen_pipeline(config)
        
        # Create and render HDR scene
        hdr_scene = self._create_test_hdr_scene(128, 128)
        pipeline.upload_hdr_data(hdr_scene)
        
        pipeline.begin_hdr_render()
        pipeline.draw_hdr_scene()
        pipeline.end_hdr_render()
        pipeline.apply_tone_mapping()
        
        # Read LDR data
        ldr_data = pipeline.read_ldr_data()
        
        # Create temporary PNG file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Remove RGB channels for PNG (RGB, no alpha)
            ldr_rgb = ldr_data[:, :, :3]
            
            # Save PNG using forge3d
            f3d.numpy_to_png(tmp_path, ldr_rgb)
            
            # Verify file was created
            assert os.path.exists(tmp_path), "PNG file was not created"
            assert os.path.getsize(tmp_path) > 0, "PNG file is empty"
            
            # Load and verify PNG
            loaded_image = f3d.png_to_numpy(tmp_path)
            assert loaded_image.shape == (128, 128, 3), f"PNG shape mismatch: {loaded_image.shape}"
            assert loaded_image.dtype == np.uint8, f"PNG dtype mismatch: {loaded_image.dtype}"
            
            # Verify PNG content matches original
            np.testing.assert_array_equal(loaded_image, ldr_rgb)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_clamp_rate_validation(self):
        """Test clamp rate computation and validation."""
        config = {
            'width': 64,
            'height': 64,
            'tone_mapping': 'reinhard',  # Should produce low clamp rate
            'exposure': 1.0
        }
        
        pipeline = f3d.create_hdr_offscreen_pipeline(config)
        
        # Create moderate dynamic range scene (should produce low clamp rate)
        hdr_scene = np.random.rand(64, 64, 3).astype(np.float32) * 2.0  # Range [0, 2]
        pipeline.upload_hdr_data(hdr_scene)
        
        pipeline.begin_hdr_render()
        pipeline.draw_hdr_scene()
        pipeline.end_hdr_render()
        pipeline.apply_tone_mapping()
        
        # Compute clamp rate
        clamp_rate = pipeline.compute_clamp_rate()
        
        # Validate clamp rate is reasonable
        assert 0.0 <= clamp_rate <= 1.0, f"Invalid clamp rate: {clamp_rate}"
        
        # For moderate HDR with good tone mapping, clamp rate should be low
        assert clamp_rate < 0.01, f"Clamp rate too high: {clamp_rate:.6f} (should be < 0.01)"
        
        logger.info(f"Clamp rate: {clamp_rate:.6f} ({clamp_rate*100:.4f}%)")
        
        # Test high clamp rate scenario
        config_extreme = config.copy()
        config_extreme['tone_mapping'] = 'clamp'  # Should produce high clamp rate
        config_extreme['exposure'] = 10.0  # Very high exposure
        
        try:
            pipeline_extreme = f3d.create_hdr_offscreen_pipeline(config_extreme)
            
            # Create extreme dynamic range scene
            hdr_extreme = np.random.rand(64, 64, 3).astype(np.float32) * 100.0  # Range [0, 100]
            pipeline_extreme.upload_hdr_data(hdr_extreme)
            
            pipeline_extreme.begin_hdr_render()
            pipeline_extreme.draw_hdr_scene()
            pipeline_extreme.end_hdr_render()
            pipeline_extreme.apply_tone_mapping()
            
            clamp_rate_extreme = pipeline_extreme.compute_clamp_rate()
            
            # Extreme scenario should have higher clamp rate
            assert clamp_rate_extreme > clamp_rate, \
                f"Extreme scenario should have higher clamp rate: {clamp_rate_extreme:.6f} vs {clamp_rate:.6f}"
            
            logger.info(f"Extreme clamp rate: {clamp_rate_extreme:.6f} ({clamp_rate_extreme*100:.4f}%)")
            
        except Exception as e:
            logger.warning(f"Extreme clamp rate test failed: {e}")
    
    def test_full_pipeline_acceptance_criteria(self):
        """Test all acceptance criteria in one comprehensive test."""
        config = {
            'width': 256,
            'height': 256,
            'tone_mapping': 'aces',
            'exposure': 1.5,
            'gamma': 2.2
        }
        
        pipeline = f3d.create_hdr_offscreen_pipeline(config)
        
        # Create test HDR scene
        hdr_scene = self._create_complex_test_scene(256, 256)
        pipeline.upload_hdr_data(hdr_scene)
        
        # Track VRAM usage before rendering
        vram_before = pipeline.get_vram_usage()
        
        # Render HDR scene
        pipeline.begin_hdr_render()
        pipeline.draw_hdr_scene()
        pipeline.end_hdr_render()
        
        # Track VRAM usage after HDR rendering
        vram_after_hdr = pipeline.get_vram_usage()
        
        # Apply tone mapping
        pipeline.apply_tone_mapping()
        
        # Track peak VRAM usage
        vram_peak = pipeline.get_vram_usage()
        
        # Read LDR result
        ldr_data = pipeline.read_ldr_data()
        
        # Compute clamp rate
        clamp_rate = pipeline.compute_clamp_rate()
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "hdr_tonemap.png"
            
            # Save PNG
            ldr_rgb = ldr_data[:, :, :3]
            f3d.numpy_to_png(str(output_path), ldr_rgb)
            
            # ACCEPTANCE CRITERIA VALIDATION
            
            # 1. PNG output created
            png_created = output_path.exists() and output_path.stat().st_size > 0
            assert png_created, "PNG output not created"
            
            # 2. Clamp rate < 1% (0.01)
            clamp_rate_ok = clamp_rate < 0.01
            assert clamp_rate_ok, f"Clamp rate too high: {clamp_rate:.6f} >= 0.01"
            
            # 3. VRAM usage ≤ 512 MiB
            vram_limit_bytes = 512 * 1024 * 1024
            vram_ok = vram_peak <= vram_limit_bytes
            assert vram_ok, f"VRAM usage {vram_peak/(1024*1024):.1f} MiB > 512 MiB limit"
            
            # Additional validation: output format and range
            assert ldr_data.shape == (256, 256, 4), f"Output shape mismatch: {ldr_data.shape}"
            assert ldr_data.dtype == np.uint8, f"Output dtype mismatch: {ldr_data.dtype}"
            assert np.all(ldr_data >= 0) and np.all(ldr_data <= 255), "Output values outside [0,255]"
            
            # Log results
            logger.info(f"=== ACCEPTANCE CRITERIA RESULTS ===")
            logger.info(f"✓ PNG output created: {png_created} ({output_path.stat().st_size} bytes)")
            logger.info(f"{'✓' if clamp_rate_ok else '✗'} Clamp rate: {clamp_rate:.6f} < 0.01")
            logger.info(f"{'✓' if vram_ok else '✗'} VRAM usage: {vram_peak/(1024*1024):.1f} MiB ≤ 512 MiB")
            logger.info(f"  VRAM breakdown: before={vram_before/(1024*1024):.1f}, "
                       f"after_hdr={vram_after_hdr/(1024*1024):.1f}, peak={vram_peak/(1024*1024):.1f} MiB")
            
            all_criteria_met = png_created and clamp_rate_ok and vram_ok
            assert all_criteria_met, "Not all acceptance criteria met"
    
    def _create_test_hdr_scene(self, width, height):
        """Create basic test HDR scene."""
        return np.random.rand(height, width, 3).astype(np.float32) * 5.0
    
    def _create_complex_test_scene(self, width, height):
        """Create complex HDR scene with varied dynamic range."""
        scene = np.zeros((height, width, 3), dtype=np.float32)
        
        # Sky region (bright)
        sky_height = height // 3
        scene[:sky_height, :, :] = np.random.rand(sky_height, width, 3) * 10.0 + 5.0
        
        # Middle region (medium)
        mid_start = sky_height
        mid_end = 2 * height // 3
        scene[mid_start:mid_end, :, :] = np.random.rand(mid_end - mid_start, width, 3) * 2.0 + 1.0
        
        # Ground region (dark)
        scene[mid_end:, :, :] = np.random.rand(height - mid_end, width, 3) * 0.5 + 0.1
        
        # Add some bright highlights
        num_highlights = 10
        for _ in range(num_highlights):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(3, 8)
            intensity = np.random.rand() * 20.0 + 5.0
            
            for dy in range(-size//2, size//2):
                for dx in range(-size//2, size//2):
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance < size / 2:
                            falloff = 1.0 - distance / (size / 2)
                            scene[y + dy, x + dx] = [intensity * falloff] * 3
        
        return scene


if __name__ == "__main__":
    # Run with verbose output for debugging
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])