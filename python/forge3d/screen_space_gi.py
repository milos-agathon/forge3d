#!/usr/bin/env python3
"""P5: Screen-Space Effects (SSAO/GTAO, SSGI, SSR)

Provides GPU-accelerated screen-space techniques for ambient occlusion,
global illumination, and reflections.
"""

from typing import Optional, Dict, Any
import numpy as np

class ScreenSpaceGI:
    """Screen-space global illumination effects manager."""
    
    # Effect types
    SSAO = "ssao"
    SSGI = "ssgi"
    SSR = "ssr"
    
    def __init__(self, width: int = 1920, height: int = 1080):
        """Initialize screen-space GI system.
        
        Args:
            width: Render width
            height: Render height
        """
        self.width = width
        self.height = height
        self._enabled_effects = set()
        self._settings = {
            self.SSAO: {
                "radius": 0.5,
                "intensity": 1.0,
                "bias": 0.025,
                "num_samples": 16,
            },
            self.SSGI: {
                "radius": 1.0,
                "intensity": 0.5,
                "num_steps": 16,
                "step_size": 0.1,
            },
            self.SSR: {
                "max_steps": 32,
                "thickness": 0.1,
                "max_distance": 10.0,
                "intensity": 1.0,
            },
        }
    
    def enable_effect(self, effect: str, **kwargs) -> None:
        """Enable a screen-space effect.
        
        Args:
            effect: Effect type (ssao, ssgi, or ssr)
            **kwargs: Effect-specific parameters
        """
        if effect not in [self.SSAO, self.SSGI, self.SSR]:
            raise ValueError(f"Unknown effect: {effect}")
        
        self._enabled_effects.add(effect)
        
        # Update settings if provided
        if kwargs:
            self._settings[effect].update(kwargs)
    
    def disable_effect(self, effect: str) -> None:
        """Disable a screen-space effect.
        
        Args:
            effect: Effect type to disable
        """
        self._enabled_effects.discard(effect)
    
    def set_ssao_settings(self, radius: Optional[float] = None,
                         intensity: Optional[float] = None,
                         bias: Optional[float] = None,
                         num_samples: Optional[int] = None) -> None:
        """Configure SSAO/GTAO settings.
        
        Args:
            radius: Sample radius in view space (default: 0.5)
            intensity: Occlusion intensity multiplier (default: 1.0)
            bias: Normal bias to prevent self-occlusion (default: 0.025)
            num_samples: Number of samples per pixel (default: 16)
        """
        settings = self._settings[self.SSAO]
        if radius is not None:
            settings["radius"] = radius
        if intensity is not None:
            settings["intensity"] = intensity
        if bias is not None:
            settings["bias"] = bias
        if num_samples is not None:
            settings["num_samples"] = num_samples
    
    def set_ssgi_settings(self, radius: Optional[float] = None,
                         intensity: Optional[float] = None,
                         num_steps: Optional[int] = None,
                         step_size: Optional[float] = None) -> None:
        """Configure SSGI settings.
        
        Args:
            radius: Maximum ray distance (default: 1.0)
            intensity: Indirect light intensity (default: 0.5)
            num_steps: Ray marching steps (default: 16)
            step_size: Step size for ray marching (default: 0.1)
        """
        settings = self._settings[self.SSGI]
        if radius is not None:
            settings["radius"] = radius
        if intensity is not None:
            settings["intensity"] = intensity
        if num_steps is not None:
            settings["num_steps"] = num_steps
        if step_size is not None:
            settings["step_size"] = step_size
    
    def set_ssr_settings(self, max_steps: Optional[int] = None,
                        thickness: Optional[float] = None,
                        max_distance: Optional[float] = None,
                        intensity: Optional[float] = None) -> None:
        """Configure SSR settings.
        
        Args:
            max_steps: Maximum ray marching steps (default: 32)
            thickness: Ray intersection thickness (default: 0.1)
            max_distance: Maximum reflection distance (default: 10.0)
            intensity: Reflection intensity (default: 1.0)
        """
        settings = self._settings[self.SSR]
        if max_steps is not None:
            settings["max_steps"] = max_steps
        if thickness is not None:
            settings["thickness"] = thickness
        if max_distance is not None:
            settings["max_distance"] = max_distance
        if intensity is not None:
            settings["intensity"] = intensity
    
    def get_settings(self, effect: str) -> Dict[str, Any]:
        """Get current settings for an effect.
        
        Args:
            effect: Effect type
            
        Returns:
            Dictionary of current settings
        """
        return self._settings.get(effect, {}).copy()
    
    def is_enabled(self, effect: str) -> bool:
        """Check if an effect is enabled.
        
        Args:
            effect: Effect type
            
        Returns:
            True if enabled
        """
        return effect in self._enabled_effects
    
    def get_enabled_effects(self) -> list:
        """Get list of enabled effects.
        
        Returns:
            List of enabled effect names
        """
        return list(self._enabled_effects)


def parse_gi_args(args) -> ScreenSpaceGI:
    """Parse command-line arguments for GI effects.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        Configured ScreenSpaceGI instance
    """
    gi = ScreenSpaceGI()
    
    # Check which effect is requested
    if hasattr(args, 'gi') and args.gi:
        effect = args.gi.lower()
        
        if effect == 'ssao':
            # Parse SSAO options
            radius = getattr(args, 'ssao_radius', 0.5)
            intensity = getattr(args, 'ssao_intensity', 1.0)
            gi.enable_effect(ScreenSpaceGI.SSAO, 
                           radius=radius, 
                           intensity=intensity)
        
        elif effect == 'ssgi':
            # Parse SSGI options
            steps = getattr(args, 'ssgi_steps', 16)
            radius = getattr(args, 'ssgi_radius', 1.0)
            gi.enable_effect(ScreenSpaceGI.SSGI,
                           num_steps=steps,
                           radius=radius)
        
        elif effect == 'ssr':
            # Parse SSR options
            max_steps = getattr(args, 'ssr_max_steps', 32)
            thickness = getattr(args, 'ssr_thickness', 0.1)
            gi.enable_effect(ScreenSpaceGI.SSR,
                           max_steps=max_steps,
                           thickness=thickness)
    
    return gi


def add_gi_arguments(parser):
    """Add GI-related command-line arguments to an argparse parser.
    
    Args:
        parser: ArgumentParser instance
    """
    gi_group = parser.add_argument_group('Screen-Space GI Options')
    
    gi_group.add_argument(
        '--gi',
        type=str,
        choices=['ssao', 'ssgi', 'ssr'],
        help='Enable screen-space global illumination effect'
    )
    
    # SSAO options
    gi_group.add_argument(
        '--ssao-radius',
        type=float,
        default=0.5,
        help='SSAO sample radius (default: 0.5)'
    )
    gi_group.add_argument(
        '--ssao-intensity',
        type=float,
        default=1.0,
        help='SSAO occlusion intensity (default: 1.0)'
    )
    
    # SSGI options
    gi_group.add_argument(
        '--ssgi-steps',
        type=int,
        default=16,
        help='SSGI ray marching steps (default: 16)'
    )
    gi_group.add_argument(
        '--ssgi-radius',
        type=float,
        default=1.0,
        help='SSGI maximum ray distance (default: 1.0)'
    )
    
    # SSR options
    gi_group.add_argument(
        '--ssr-max-steps',
        type=int,
        default=32,
        help='SSR maximum ray marching steps (default: 32)'
    )
    gi_group.add_argument(
        '--ssr-thickness',
        type=float,
        default=0.1,
        help='SSR ray intersection thickness (default: 0.1)'
    )
