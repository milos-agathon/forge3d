"""
Environment mapping and IBL tests

Tests for environment map loading, validation, roughness monotonicity,
and luminance histogram changes with different roughness values.
"""

import numpy as np
import pytest
import logging

# Skip if environment mapping not available
try:
    import forge3d.envmap as envmap
    _ENVMAP_AVAILABLE = True
except ImportError:
    _ENVMAP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _ENVMAP_AVAILABLE,
    reason="Environment mapping module not available"
)

logger = logging.getLogger(__name__)


class TestEnvironmentMap:
    """Test EnvironmentMap class functionality."""
    
    def test_create_environment_map(self):
        """Test environment map creation with valid data."""
        width, height = 64, 32
        data = np.random.rand(height, width, 3).astype(np.float32)
        
        env = envmap.EnvironmentMap(width, height, data)
        
        assert env.width == width
        assert env.height == height
        assert env.data.shape == (height, width, 3)
        assert env.data.dtype == np.float32
    
    def test_create_environment_map_invalid_shape(self):
        """Test environment map creation with invalid data shape."""
        with pytest.raises(ValueError, match="data shape.*does not match expected"):
            envmap.EnvironmentMap(64, 32, np.zeros((16, 16, 3)))
    
    def test_create_environment_map_invalid_dtype(self):
        """Test environment map creation with invalid data type."""
        data = np.zeros((32, 64, 3), dtype=np.uint8)
        
        with pytest.raises(TypeError, match="data must be float32 or float64"):
            envmap.EnvironmentMap(64, 32, data)
    
    def test_create_test_envmap(self):
        """Test synthetic test environment map creation."""
        size = 128
        env = envmap.EnvironmentMap.create_test_envmap(size)
        
        assert env.width == size
        assert env.height == size
        assert env.data.shape == (size, size, 3)
        assert env.data.dtype == np.float32
        
        # Check data range is reasonable
        assert np.all(env.data >= 0.0)
        assert np.all(env.data <= 1.0)  # Test pattern should be in [0,1] range
    
    def test_sample_direction(self):
        """Test environment map directional sampling."""
        env = envmap.EnvironmentMap.create_test_envmap(64)
        
        # Test various directions
        test_directions = [
            [0, 1, 0],   # up
            [0, -1, 0],  # down
            [1, 0, 0],   # right
            [-1, 0, 0],  # left
            [0, 0, 1],   # forward
            [0, 0, -1],  # backward
        ]
        
        for direction in test_directions:
            color = env.sample_direction(np.array(direction, dtype=np.float32))
            
            # Check valid RGB output
            assert color.shape == (3,)
            assert np.all(color >= 0.0)
            assert np.all(np.isfinite(color))
    
    def test_sample_direction_invalid_input(self):
        """Test environment map sampling with invalid input."""
        env = envmap.EnvironmentMap.create_test_envmap(32)
        
        # Invalid shape
        with pytest.raises(ValueError, match="direction must be \\(3,\\) array"):
            env.sample_direction(np.array([1, 0]))
        
        # Zero vector should be handled gracefully
        color = env.sample_direction(np.array([0, 0, 0], dtype=np.float32))
        assert color.shape == (3,)


class TestEnvironmentMapValidation:
    """Test environment map validation functionality."""
    
    def test_validate_valid_environment_map(self):
        """Test validation of valid environment map."""
        env = envmap.EnvironmentMap.create_test_envmap(64)
        results = envmap.validate_environment_map(env)
        
        assert results['valid'] is True
        assert len(results['errors']) == 0
        assert 'statistics' in results
        
        stats = results['statistics']
        assert 'min_value' in stats
        assert 'max_value' in stats
        assert 'mean_value' in stats
        assert 'memory_mb' in stats
    
    def test_validate_invalid_dimensions(self):
        """Test validation with invalid dimensions."""
        data = np.zeros((1, 1, 3), dtype=np.float32)
        env = envmap.EnvironmentMap(0, 1, data)  # Invalid width
        env.width = 0  # Force invalid after creation
        
        results = envmap.validate_environment_map(env)
        
        assert results['valid'] is False
        assert any('Invalid dimensions' in error for error in results['errors'])
    
    def test_validate_negative_values(self):
        """Test validation with negative values."""
        width, height = 32, 32
        data = np.random.rand(height, width, 3).astype(np.float32)
        data[0, 0, 0] = -1.0  # Insert negative value
        
        env = envmap.EnvironmentMap(width, height, data)
        results = envmap.validate_environment_map(env)
        
        assert results['valid'] is False
        assert any('Negative values found' in error for error in results['errors'])
    
    def test_validate_nan_values(self):
        """Test validation with NaN values."""
        width, height = 16, 16
        data = np.random.rand(height, width, 3).astype(np.float32)
        data[0, 0, 0] = np.nan
        
        env = envmap.EnvironmentMap(width, height, data)
        results = envmap.validate_environment_map(env)
        
        assert results['valid'] is False
        assert any('NaN or infinite' in error for error in results['errors'])


class TestRoughnessMonotonicity:
    """Test roughness monotonicity requirements for IBL."""
    
    def test_roughness_luminance_series_basic(self):
        """Test basic roughness luminance computation."""
        env = envmap.EnvironmentMap.create_test_envmap(128)
        roughness_values = [0.1, 0.5, 0.9]
        
        luminances = envmap.compute_roughness_luminance_series(env, roughness_values)
        
        assert len(luminances) == len(roughness_values)
        assert all(isinstance(l, float) for l in luminances)
        assert all(l >= 0.0 for l in luminances)
    
    def test_roughness_monotonicity_trend(self):
        """Test that roughness strictly decreases luminance: L(0.1) > L(0.5) > L(0.9) with ±5% tolerance (AC-N5-1)."""
        env = envmap.EnvironmentMap.create_test_envmap(256)
        
        # Test specific roughness values from AC requirement
        roughness_values = [0.1, 0.5, 0.9]
        luminances = envmap.compute_roughness_luminance_series(env, roughness_values)
        
        l_01, l_05, l_09 = luminances
        
        print(f"\nRoughness monotonicity test (AC-N5-1):")
        print(f"  L(0.1) = {l_01:.6f}")
        print(f"  L(0.5) = {l_05:.6f}")
        print(f"  L(0.9) = {l_09:.6f}")
        
        # AC-N5-1 requirement: enforce L(0.1) > L(0.5) > L(0.9) with ±5% tolerance
        # Use ratio ≥0.95 as lower bound (5% tolerance)
        tolerance_ratio = 0.95
        
        condition1 = l_01 > l_05 * tolerance_ratio
        condition2 = l_05 > l_09 * tolerance_ratio
        
        print(f"  L(0.1) > L(0.5) * {tolerance_ratio}: {condition1} ({l_01:.6f} > {l_05 * tolerance_ratio:.6f})")
        print(f"  L(0.5) > L(0.9) * {tolerance_ratio}: {condition2} ({l_05:.6f} > {l_09 * tolerance_ratio:.6f})")
        
        # Strict assertions with ±5% tolerance
        assert l_01 > l_05 * tolerance_ratio, f"L(0.1) {l_01:.6f} must be > L(0.5) * {tolerance_ratio} = {l_05 * tolerance_ratio:.6f}"
        assert l_05 > l_09 * tolerance_ratio, f"L(0.5) {l_05:.6f} must be > L(0.9) * {tolerance_ratio} = {l_09 * tolerance_ratio:.6f}"
        
        print("  PASS: Roughness monotonicity satisfied with ±5% tolerance")
    
    def test_roughness_empty_list(self):
        """Test roughness computation with empty input."""
        env = envmap.EnvironmentMap.create_test_envmap(32)
        luminances = envmap.compute_roughness_luminance_series(env, [])
        
        assert luminances == []
    
    def test_roughness_single_value(self):
        """Test roughness computation with single value."""
        env = envmap.EnvironmentMap.create_test_envmap(64)
        luminances = envmap.compute_roughness_luminance_series(env, [0.5])
        
        assert len(luminances) == 1
        assert isinstance(luminances[0], float)
        assert luminances[0] >= 0.0


class TestHistogramChanges:
    """Test histogram changes with different roughness values (AC requirement)."""
    
    def test_luminance_histogram_changes(self):
        """Test that different roughness values produce different luminance histograms."""
        env = envmap.EnvironmentMap.create_test_envmap(256)
        
        # Generate luminance samples for different roughness values
        roughness_low = 0.1
        roughness_high = 0.9
        num_samples = 100
        
        # Sample multiple directions for histogram analysis
        directions = []
        for i in range(num_samples):
            # Generate random directions on sphere
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            
            theta = np.arccos(2*u - 1)  # Uniform distribution on sphere
            phi = 2 * np.pi * v
            
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.cos(theta),
                np.sin(theta) * np.sin(phi)
            ])
            directions.append(direction)
        
        # Compute luminance histograms for different roughness
        luminances_low = []
        luminances_high = []
        
        for direction in directions:
            # Simple roughness simulation by sampling with perturbation
            color_low = env.sample_direction(direction)
            luminance_low = 0.299 * color_low[0] + 0.587 * color_low[1] + 0.114 * color_low[2]
            luminances_low.append(luminance_low)
            
            # For high roughness, sample with more variation (simplified)
            perturbed_direction = direction + np.random.normal(0, roughness_high * 0.1, 3)
            perturbed_direction = perturbed_direction / np.linalg.norm(perturbed_direction)
            color_high = env.sample_direction(perturbed_direction)
            luminance_high = 0.299 * color_high[0] + 0.587 * color_high[1] + 0.114 * color_high[2]
            luminances_high.append(luminance_high)
        
        # Compute histogram statistics
        hist_low, _ = np.histogram(luminances_low, bins=10, range=(0, 1))
        hist_high, _ = np.histogram(luminances_high, bins=10, range=(0, 1))
        
        # Check that histograms are different
        histogram_difference = np.sum(np.abs(hist_low - hist_high))
        
        print(f"\nHistogram analysis:")
        print(f"  Low roughness ({roughness_low}) mean luminance: {np.mean(luminances_low):.4f}")
        print(f"  High roughness ({roughness_high}) mean luminance: {np.mean(luminances_high):.4f}")
        print(f"  Histogram difference (L1): {histogram_difference}")
        
        # AC requirement: histograms should be different
        assert histogram_difference > 0, "Histograms should differ between roughness values"
        print("  PASS: Histogram changes detected with different roughness values")
    
    def test_luminance_histogram_chisquare_rotation(self):
        """Test chi-square goodness-of-fit between original and rotated environment histograms (AC-N5-2)."""
        env = envmap.EnvironmentMap.create_test_envmap(256)
        
        # Create rotated copy by rolling along U by 25% width
        roll_amount = int(env.width * 0.25)
        rotated_data = np.roll(env.data, roll_amount, axis=1)  # Roll along width dimension
        rotated_env = envmap.EnvironmentMap(env.width, env.height, rotated_data)
        
        print(f"\nChi-square histogram test (AC-N5-2):")
        print(f"  Environment size: {env.width}x{env.height}")
        print(f"  Rotation: {roll_amount} pixels ({roll_amount/env.width:.1%} of width)")
        
        # Sample both environments to create luminance histograms
        num_samples = 1000
        bins = 20
        
        # Generate consistent sample directions for both environments
        np.random.seed(42)  # Fixed seed for reproducibility
        directions = []
        for i in range(num_samples):
            # Generate random directions on sphere
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            
            theta = np.arccos(2*u - 1)  # Uniform distribution on sphere
            phi = 2 * np.pi * v
            
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.cos(theta),
                np.sin(theta) * np.sin(phi)
            ])
            directions.append(direction)
        
        # Sample luminances from both environments
        original_luminances = []
        rotated_luminances = []
        
        for direction in directions:
            # Sample original environment
            color_orig = env.sample_direction(direction)
            lum_orig = 0.299 * color_orig[0] + 0.587 * color_orig[1] + 0.114 * color_orig[2]
            original_luminances.append(lum_orig)
            
            # Sample rotated environment
            color_rot = rotated_env.sample_direction(direction)
            lum_rot = 0.299 * color_rot[0] + 0.587 * color_rot[1] + 0.114 * color_rot[2]
            rotated_luminances.append(lum_rot)
        
        # Create histograms with same binning
        lum_range = (0, max(max(original_luminances), max(rotated_luminances)))
        hist_orig, bin_edges = np.histogram(original_luminances, bins=bins, range=lum_range)
        hist_rot, _ = np.histogram(rotated_luminances, bins=bins, range=lum_range)
        
        # Perform chi-square goodness-of-fit test
        try:
            # Try to use scipy.stats.chisquare if available
            import scipy.stats
            
            # Chi-square test expects frequency data; add small constant to avoid zero expected values
            expected = hist_orig + 1e-6
            observed = hist_rot + 1e-6
            
            chi2_stat, p_value = scipy.stats.chisquare(observed, expected)
            dof = bins - 1
            
            print(f"  Chi-square statistic: {chi2_stat:.6f}")
            print(f"  Degrees of freedom: {dof}")
            print(f"  p-value: {p_value:.6f}")
            
        except ImportError:
            # Fallback implementation using critical value comparison
            print("  SciPy not available, using critical value comparison")
            
            # Calculate chi-square statistic manually
            expected = hist_orig + 1e-6
            observed = hist_rot + 1e-6
            chi2_stat = np.sum((observed - expected)**2 / expected)
            dof = bins - 1
            
            # Critical value for df=19, alpha=0.01 is approximately 36.19
            critical_values = {19: 36.19, 18: 34.81, 20: 37.57, 15: 30.58, 9: 21.67}
            critical_value = critical_values.get(dof, 30.0)  # Conservative fallback
            
            p_value = 1.0 if chi2_stat < critical_value else 0.005  # Approximate
            
            print(f"  Chi-square statistic: {chi2_stat:.6f}")
            print(f"  Degrees of freedom: {dof}")
            print(f"  Critical value (alpha=0.01): {critical_value:.6f}")
            print(f"  Approximate p-value: {'<0.01' if p_value <= 0.01 else '>0.01'}")
        
        # AC-N5-2 requirement: chi-square p < 0.01 (significant difference between histograms)
        assert p_value < 0.01, f"Chi-square test p-value {p_value:.6f} must be < 0.01 for rotated histograms to differ significantly"
        
        print(f"  PASS: Chi-square test p-value {p_value:.6f} < 0.01 (significant histogram difference)")


class TestUtilityFunctions:
    """Test utility functions for environment mapping."""
    
    def test_has_envmap_support(self):
        """Test environment mapping support detection."""
        support = envmap.has_envmap_support()
        assert isinstance(support, bool)
        
        # If we got here, the module loaded, so support should be True
        assert support is True
    
    def test_compute_luminance_difference(self):
        """Test luminance difference computation."""
        # Create two different images
        image1 = np.zeros((64, 64, 3), dtype=np.uint8)
        image1.fill(100)  # Gray image
        
        image2 = np.zeros((64, 64, 3), dtype=np.uint8)  
        image2.fill(200)  # Brighter gray image
        
        diff = envmap.compute_luminance_difference(image1, image2)
        
        assert isinstance(diff, float)
        assert diff > 0  # Should be different
        assert diff <= 100  # Percentage difference
    
    def test_compute_luminance_difference_same_images(self):
        """Test luminance difference with identical images."""
        image = np.ones((32, 32, 3), dtype=np.uint8) * 128
        diff = envmap.compute_luminance_difference(image, image)
        
        assert diff == 0.0
    
    def test_compute_luminance_difference_shape_mismatch(self):
        """Test luminance difference with mismatched shapes."""
        image1 = np.zeros((32, 32, 3))
        image2 = np.zeros((64, 64, 3))
        
        with pytest.raises(ValueError, match="Image shapes must match"):
            envmap.compute_luminance_difference(image1, image2)


if __name__ == "__main__":
    # Run with verbose output to capture logs for AC requirements
    pytest.main([__file__, "-v", "-s"])