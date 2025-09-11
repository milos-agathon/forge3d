"""
GPU performance metrics and profiling for forge3d.

This module provides Python access to GPU timing data, profiling markers,
and performance statistics collected during rendering operations.

The GPU metrics system supports:
- Timestamp queries for measuring GPU time  
- RenderDoc/Nsight Graphics/RGP debug markers
- Pipeline statistics when available
- Minimal performance overhead when enabled

Usage:
    import forge3d.gpu_metrics as metrics
    
    # Enable GPU profiling for a renderer
    renderer.enable_gpu_profiling(config)
    
    # Render with timing
    renderer.render_rgba()
    
    # Get performance data
    timing_data = renderer.get_gpu_metrics()
    print(f"HDR pass: {timing_data['hdr_render_ms']:.2f} ms")
    print(f"Tonemap: {timing_data['hdr_tonemap_ms']:.2f} ms")
"""

from typing import Dict, Any, List, Optional, Union
import time
AVAILABLE_METRICS: Dict[str, str] = {
    'hdr_render': 'HDR render pass time (ms)',
    'hdr_tonemap': 'Tonemap pass time (ms)',
    'vector_indirect_culling': 'Vector indirect culling',
}

def get_available_metrics() -> Dict[str, str]:
    """Return available metric keys mapped to human-friendly descriptions."""
    return AVAILABLE_METRICS.copy()

class GpuTimingConfig:
    """Configuration for GPU timing and profiling."""
    
    def __init__(self,
                 enable_timestamps: bool = True,
                 enable_pipeline_stats: bool = False,
                 enable_debug_markers: bool = True,
                 label_prefix: str = "forge3d",
                 max_queries_per_frame: int = 32):
        """Initialize GPU timing configuration.
        
        Args:
            enable_timestamps: Enable timestamp queries (requires TIMESTAMP_QUERY feature)
            enable_pipeline_stats: Enable pipeline statistics (requires PIPELINE_STATISTICS_QUERY)  
            enable_debug_markers: Enable debug markers for external profilers
            label_prefix: Prefix for all GPU timing labels
            max_queries_per_frame: Maximum number of timing queries per frame
        """
        self.enable_timestamps = enable_timestamps
        self.enable_pipeline_stats = enable_pipeline_stats  
        self.enable_debug_markers = enable_debug_markers
        self.label_prefix = label_prefix
        self.max_queries_per_frame = max_queries_per_frame
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for native interface."""
        return {
            'enable_timestamps': self.enable_timestamps,
            'enable_pipeline_stats': self.enable_pipeline_stats,
            'enable_debug_markers': self.enable_debug_markers,
            'label_prefix': self.label_prefix,
            'max_queries_per_frame': self.max_queries_per_frame,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GpuTimingConfig':
        """Create from dictionary."""
        return cls(
            enable_timestamps=data.get('enable_timestamps', True),
            enable_pipeline_stats=data.get('enable_pipeline_stats', False),
            enable_debug_markers=data.get('enable_debug_markers', True),
            label_prefix=data.get('label_prefix', 'forge3d'),
            max_queries_per_frame=data.get('max_queries_per_frame', 32),
        )


class TimingResult:
    """Result from a GPU timing measurement."""
    
    def __init__(self,
                 name: str,
                 gpu_time_ms: float = 0.0,
                 timestamp_valid: bool = False,
                 pipeline_stats: Optional[Dict[str, int]] = None):
        """Initialize timing result.
        
        Args:
            name: Name/label of the timing scope
            gpu_time_ms: GPU time in milliseconds
            timestamp_valid: Whether the timestamp measurement is valid
            pipeline_stats: Optional pipeline statistics
        """
        self.name = name
        self.gpu_time_ms = gpu_time_ms
        self.timestamp_valid = timestamp_valid
        self.pipeline_stats = pipeline_stats or {}
    
    def __str__(self) -> str:
        status = "✓" if self.timestamp_valid else "✗"
        return f"{self.name}: {self.gpu_time_ms:.2f} ms {status}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'gpu_time_ms': self.gpu_time_ms,
            'timestamp_valid': self.timestamp_valid,
            'pipeline_stats': self.pipeline_stats,
        }


class GpuMetrics:
    """Container for GPU performance metrics and timing data."""
    
    def __init__(self):
        self.timing_results: List[TimingResult] = []
        self.frame_time_ms: float = 0.0
        self.total_gpu_time_ms: float = 0.0
        self.timestamp_overhead_ms: float = 0.0
        self.feature_support: Dict[str, bool] = {}
        self.device_info: Dict[str, Any] = {}
    
    def add_timing_result(self, result: TimingResult):
        """Add a timing result to this metrics collection."""
        self.timing_results.append(result)
        if result.timestamp_valid:
            self.total_gpu_time_ms += result.gpu_time_ms
    
    def get_timing_by_name(self, name: str) -> Optional[TimingResult]:
        """Get timing result by name."""
        for result in self.timing_results:
            if result.name == name:
                return result
        return None
    
    def get_timings_dict(self) -> Dict[str, float]:
        """Get all timings as a name -> milliseconds dictionary."""
        return {result.name: result.gpu_time_ms for result in self.timing_results}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        valid_timings = [r for r in self.timing_results if r.timestamp_valid]
        
        return {
            'frame_time_ms': self.frame_time_ms,
            'total_gpu_time_ms': self.total_gpu_time_ms,
            'timestamp_overhead_ms': self.timestamp_overhead_ms,
            'timing_count': len(self.timing_results),
            'valid_timing_count': len(valid_timings),
            'feature_support': self.feature_support.copy(),
            'device_info': self.device_info.copy(),
        }
    
    def __str__(self) -> str:
        lines = [
            f"GPU Metrics Summary:",
            f"  Frame time: {self.frame_time_ms:.2f} ms",
            f"  Total GPU time: {self.total_gpu_time_ms:.2f} ms", 
            f"  Timing results: {len(self.timing_results)}",
            ""
        ]
        
        for result in self.timing_results:
            lines.append(f"  {result}")
        
        if self.feature_support:
            lines.extend([
                "",
                "Feature Support:",
            ])
            for feature, supported in self.feature_support.items():
                status = "✓" if supported else "✗"
                lines.append(f"  {feature}: {status}")
        
        return "\n".join(lines)


def create_default_config() -> GpuTimingConfig:
    """Create a default GPU timing configuration.
    
    Returns:
        Default configuration suitable for most use cases.
    """
    return GpuTimingConfig(
        enable_timestamps=True,
        enable_pipeline_stats=False,  # Often not supported
        enable_debug_markers=True,
        label_prefix="forge3d",
        max_queries_per_frame=32,
    )


def create_minimal_config() -> GpuTimingConfig:
    """Create a minimal GPU timing configuration for maximum performance.
    
    Returns:
        Configuration with minimal overhead for production use.
    """
    return GpuTimingConfig(
        enable_timestamps=False,
        enable_pipeline_stats=False,
        enable_debug_markers=False,  # Disable all profiling
        label_prefix="forge3d",
        max_queries_per_frame=0,
    )


def create_debug_config() -> GpuTimingConfig:
    """Create a debug GPU timing configuration for detailed profiling.
    
    Returns:
        Configuration with all profiling features enabled.
    """
    return GpuTimingConfig(
        enable_timestamps=True,
        enable_pipeline_stats=True,  # May not be supported
        enable_debug_markers=True,
        label_prefix="forge3d",
        max_queries_per_frame=64,
    )


def estimate_timing_overhead(query_count: int, enable_markers: bool = True) -> float:
    """Estimate GPU timing overhead in milliseconds.
    
    Args:
        query_count: Number of timestamp queries
        enable_markers: Whether debug markers are enabled
        
    Returns:
        Estimated overhead in milliseconds
    """
    # Rough estimates based on typical GPU behavior
    timestamp_overhead_ms = query_count * 0.001  # ~1 microsecond per query
    marker_overhead_ms = query_count * 0.002 if enable_markers else 0.0  # ~2 microseconds per marker
    
    return timestamp_overhead_ms + marker_overhead_ms


def validate_config(config: GpuTimingConfig, device_features: Dict[str, bool]) -> List[str]:
    """Validate timing configuration against device capabilities.
    
    Args:
        config: GPU timing configuration to validate
        device_features: Dictionary of device feature support
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    if config.enable_timestamps and not device_features.get('timestamps', False):
        warnings.append("Timestamps requested but TIMESTAMP_QUERY feature not supported")
    
    if config.enable_pipeline_stats and not device_features.get('pipeline_stats', False):
        warnings.append("Pipeline stats requested but PIPELINE_STATISTICS_QUERY feature not supported")
    
    if config.max_queries_per_frame > 256:
        warnings.append("Very high query count may impact performance")
    
    return warnings


# Pre-defined timing scopes for common rendering passes
COMMON_TIMING_SCOPES = {
    'hdr_render': 'HDR rendering pass',
    'hdr_tonemap': 'HDR tone mapping',
    'hdr_offscreen_tonemap': 'HDR offscreen tone mapping', 
    'terrain_lod_update': 'Terrain LOD updates',
    'vector_indirect_culling': 'Vector indirect culling',
    'postfx_chain': 'Post-processing effects chain',
    'bloom_brightpass': 'Bloom bright pass',
    'bloom_blur_h': 'Bloom horizontal blur',
    'bloom_blur_v': 'Bloom vertical blur',
    'bloom_composite': 'Bloom composite',
}


def get_timing_scope_description(scope_name: str) -> str:
    """Get description for a timing scope name.
    
    Args:
        scope_name: Name of the timing scope
        
    Returns:
        Human-readable description of the scope
    """
    return COMMON_TIMING_SCOPES.get(scope_name, scope_name)
