"""
Colormap compression utilities for forge3d.

This module provides compressed texture functionality for colormaps,
including format detection, quality estimation, and device compatibility checking.
"""

import numpy as np
from typing import Dict, List, Optional, Any


def decode_png_rgba8(colormap_name: str) -> bytes:
    """
    Decode a colormap PNG to RGBA8 format.
    
    Args:
        colormap_name: Name of the colormap (e.g., 'viridis', 'magma', 'terrain')
        
    Returns:
        Raw RGBA8 bytes data
        
    Note:
        This is a stub implementation for testing compressed texture functionality.
        In a full implementation, this would decode actual PNG data.
    """
    # Stub implementation - return synthetic data based on colormap name
    # Typical colormap is 256x1 pixels, RGBA8 = 1024 bytes
    width, height = 256, 1
    size = width * height * 4  # RGBA8
    
    # Generate synthetic colormap data based on name
    if colormap_name == "viridis":
        # Purple to yellow gradient
        data = bytearray(size)
        for i in range(width):
            idx = i * 4
            t = i / (width - 1)
            data[idx] = int((1 - t) * 68 + t * 253)      # R
            data[idx + 1] = int(t * 231)                 # G  
            data[idx + 2] = int((1 - t) * 84 + t * 37)   # B
            data[idx + 3] = 255                          # A
    elif colormap_name == "magma":
        # Black to white through purple gradient
        data = bytearray(size)
        for i in range(width):
            idx = i * 4
            t = i / (width - 1)
            data[idx] = int(t * 252)                     # R
            data[idx + 1] = int(t * 253)                 # G
            data[idx + 2] = int(t * 191)                 # B
            data[idx + 3] = 255                          # A
    elif colormap_name == "terrain":
        # Brown to green to white gradient
        data = bytearray(size)
        for i in range(width):
            idx = i * 4
            t = i / (width - 1)
            if t < 0.5:
                # Brown to green
                data[idx] = int((1 - t*2) * 139 + t*2 * 34)      # R
                data[idx + 1] = int((1 - t*2) * 69 + t*2 * 139)  # G
                data[idx + 2] = int((1 - t*2) * 19 + t*2 * 34)   # B
            else:
                # Green to white  
                t2 = (t - 0.5) * 2
                data[idx] = int((1 - t2) * 34 + t2 * 255)        # R
                data[idx + 1] = int((1 - t2) * 139 + t2 * 255)   # G
                data[idx + 2] = int((1 - t2) * 34 + t2 * 255)    # B
            data[idx + 3] = 255                          # A
    else:
        # Default gradient
        data = bytearray(size)
        for i in range(width):
            idx = i * 4
            val = int(i * 255 / (width - 1))
            data[idx:idx+3] = [val, val, val]            # RGB
            data[idx + 3] = 255                          # A
    
    return bytes(data)


def get_colormap_compression_stats(colormap_name: str) -> str:
    """
    Get compression statistics and estimates for a colormap.
    
    Args:
        colormap_name: Name of the colormap
        
    Returns:
        String containing compression statistics and estimates
    """
    # Get original data size
    original_data = decode_png_rgba8(colormap_name)
    original_size = len(original_data)
    
    # Estimate compressed sizes for different formats
    bc1_size = original_size // 4  # 4:1 compression
    bc3_size = original_size // 2  # 2:1 compression  
    bc7_size = original_size // 2  # 2:1 compression (high quality)
    etc2_size = original_size // 3  # 3:1 compression
    
    stats = f"""Colormap '{colormap_name}' Compression Estimates:
Original Size: {original_size} bytes ({original_size / 1024:.1f} KB)

Compression Estimates:
- BC1 (DXT1): {bc1_size} bytes, 4.0:1 ratio, ~75% reduction
- BC3 (DXT5): {bc3_size} bytes, 2.0:1 ratio, ~50% reduction  
- BC7: {bc7_size} bytes, 2.0:1 ratio, ~50% reduction (high quality)
- ETC2: {etc2_size} bytes, 3.0:1 ratio, ~67% reduction

Quality Estimates (PSNR):
- BC1: ~35 dB (acceptable)
- BC3: ~40 dB (good)
- BC7: ~45 dB (excellent)
- ETC2: ~36 dB (acceptable)

Note: These are theoretical estimates based on typical compression ratios.
Actual results may vary depending on colormap content and compression settings.
"""
    
    return stats


def check_compressed_colormap_support(device: Optional[Any] = None) -> Dict[str, bool]:
    """
    Check compressed texture format support for colormaps on a device.
    
    Args:
        device: GPU device to check (None for no device available)
        
    Returns:
        Dictionary mapping format names to support status
    """
    if device is None:
        # No device available - return conservative estimates
        return {
            "BC1": False,
            "BC3": False, 
            "BC7": False,
            "ETC2": False,
            "ASTC": False,
        }
    
    # Stub implementation - in reality this would query the device capabilities
    # For testing purposes, assume some formats are supported
    return {
        "BC1": True,   # Widely supported
        "BC3": True,   # Widely supported
        "BC7": True,   # Modern devices
        "ETC2": False, # Mobile format
        "ASTC": False, # Mobile format
    }


def get_supported_compression_formats() -> List[str]:
    """
    Get list of supported compression formats.
    
    Returns:
        List of supported format names
    """
    # Stub implementation - return formats that are commonly supported
    return ["BC1", "BC3", "BC7"]


def estimate_compression_quality(format_name: str, source_data: bytes) -> float:
    """
    Estimate compression quality (PSNR) for a format.
    
    Args:
        format_name: Name of compression format
        source_data: Original image data
        
    Returns:
        Estimated PSNR in dB
    """
    # Quality estimates based on format characteristics
    quality_map = {
        "BC1": 35.0,   # Basic compression
        "BC3": 40.0,   # Good compression with alpha
        "BC7": 45.0,   # High quality
        "ETC2": 36.0,  # Mobile format
        "ASTC": 42.0,  # Adaptive format
    }
    
    return quality_map.get(format_name, 30.0)  # Default to minimum acceptable


def calculate_memory_savings(original_size: int, compressed_size: int) -> Dict[str, float]:
    """
    Calculate memory savings from compression.
    
    Args:
        original_size: Size of original data in bytes
        compressed_size: Size of compressed data in bytes
        
    Returns:
        Dictionary with savings metrics
    """
    ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    reduction_percent = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0.0
    
    return {
        "compression_ratio": ratio,
        "reduction_percent": reduction_percent,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "bytes_saved": original_size - compressed_size,
    }