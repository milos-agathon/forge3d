"""Color conversion utilities for forge3d viewers and renderers."""

from __future__ import annotations

from typing import List, Tuple


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple (0-255 range).
    
    Args:
        hex_color: Color in hex format, e.g. '#FF5500' or 'FF5500'
        
    Returns:
        Tuple of (R, G, B) values in 0-255 range
    """
    hex_color = hex_color.lstrip('#')
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> List[float]:
    """Convert hex color to RGBA list (0.0-1.0 range).
    
    Args:
        hex_color: Color in hex format, e.g. '#FF5500' or 'FF5500' or '#FF5500AA'
        alpha: Alpha value (0.0-1.0), used if hex doesn't include alpha
        
    Returns:
        List of [R, G, B, A] values in 0.0-1.0 range
        
    Raises:
        ValueError: if hex color format is invalid
    """
    hex_color = hex_color.lstrip('#')
    
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return [r, g, b, alpha]
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return [r, g, b, a]
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")


def rgb_to_normalized(rgb: Tuple[int, int, int]) -> List[float]:
    """Convert RGB (0-255) to normalized (0.0-1.0) values.
    
    Args:
        rgb: Tuple of (R, G, B) values in 0-255 range
        
    Returns:
        List of [R, G, B] values in 0.0-1.0 range
    """
    return [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]
