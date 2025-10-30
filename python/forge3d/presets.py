"""
High-level rendering presets for forge3d (P7)

This module provides convenient preset configurations that combine multiple
rendering features into cohesive, production-ready setups. Each preset returns
a configuration dictionary that can be merged into a Renderer's config or
passed directly during initialization.

Example usage:
    import forge3d
    from forge3d.presets import studio_pbr, outdoor_sun

    # Use preset directly
    renderer = forge3d.Renderer(800, 600, **studio_pbr())

    # Or merge preset with custom overrides
    config = outdoor_sun(turbidity=3.5, ground_albedo=0.3)
    renderer = forge3d.Renderer(800, 600, **config)

Available presets:
- studio_pbr(): Indoor studio lighting with IBL and soft shadows
- outdoor_sun(): Outdoor scene with physical sky and cascaded shadow maps
- toon_viz(): Stylized toon rendering with hard shadows
- minimal(): Minimal setup for fast previews
- high_quality(): Maximum quality settings for final renders
"""

from typing import Dict, Any, Optional


def studio_pbr(
    ibl_intensity: float = 1.0,
    light_intensity: float = 3.0,
    shadow_map_res: int = 2048,
    roughness: float = 0.5,
    metallic: float = 0.0,
    **overrides
) -> Dict[str, Any]:
    """
    Studio PBR lighting preset (P7)

    Indoor studio setup with:
    - Directional light (key light)
    - IBL for ambient and specular reflections
    - PCF shadows for soft shadow edges
    - Disney Principled BRDF for physically-based materials

    Perfect for: Product visualization, character rendering, studio setups

    Args:
        ibl_intensity: IBL contribution multiplier [0-2], default 1.0
        light_intensity: Main light intensity [0-10], default 3.0
        shadow_map_res: Shadow map resolution [512-4096], default 2048
        roughness: Default surface roughness [0-1], default 0.5
        metallic: Default metallic factor [0-1], default 0.0
        **overrides: Additional config overrides

    Returns:
        Configuration dictionary ready for Renderer(**config)

    Example:
        >>> from forge3d.presets import studio_pbr
        >>> renderer = forge3d.Renderer(800, 600, **studio_pbr())
        >>> # Or with overrides:
        >>> config = studio_pbr(ibl_intensity=1.5, roughness=0.3)
    """
    config = {
        # Lighting
        'lighting': {
            'lights': [
                {
                    'type': 'directional',
                    'direction': [0.5, -0.8, 0.3],  # 45° from top-right
                    'color': [1.0, 1.0, 1.0],
                    'intensity': light_intensity,
                }
            ],
        },

        # Material shading
        'shading': {
            'brdf': 'disney-principled',
            'roughness': roughness,
            'metallic': metallic,
            'clearcoat': 0.0,
            'sheen': 0.0,
        },

        # Shadows
        'shadows': {
            'enabled': True,
            'technique': 'PCF',
            'map_res': shadow_map_res,
            'bias': 0.001,
            'pcf_radius': 2.0,
        },

        # Global Illumination
        'gi': {
            'technique': 'IBL',
            'ibl_intensity': ibl_intensity,
            'ibl_rotation': 0.0,
        },

        # Atmosphere (minimal)
        'atmosphere': {
            'fog_density': 0.0,
            'exposure': 1.0,
            'sky_model': 'Off',
        },
    }

    # Merge overrides
    config.update(overrides)
    return config


def outdoor_sun(
    turbidity: float = 2.5,
    ground_albedo: float = 0.2,
    sun_elevation: float = 45.0,
    sun_azimuth: float = 135.0,
    cascades: int = 4,
    fog_density: float = 0.0,
    **overrides
) -> Dict[str, Any]:
    """
    Outdoor sun lighting preset (P7)

    Outdoor scene with:
    - Hosek-Wilkie physical sky model
    - Sun as directional light (synced with sky)
    - CSM (Cascaded Shadow Maps) for large scenes
    - Cook-Torrance GGX BRDF for realistic materials
    - Optional volumetric fog

    Perfect for: Terrain rendering, outdoor scenes, architectural visualization

    Args:
        turbidity: Atmospheric turbidity [1-10], default 2.5 (clear)
            1.0 = very clear, 2.5 = clear, 6.0 = hazy, 10.0 = very hazy
        ground_albedo: Ground reflectance [0-1], default 0.2
        sun_elevation: Sun elevation angle in degrees [0-90], default 45
        sun_azimuth: Sun azimuth angle in degrees [0-360], default 135
        cascades: Number of CSM cascades [1-4], default 4
        fog_density: Volumetric fog density [0-1], default 0.0 (no fog)
        **overrides: Additional config overrides

    Returns:
        Configuration dictionary ready for Renderer(**config)

    Example:
        >>> from forge3d.presets import outdoor_sun
        >>> # Sunrise scene
        >>> config = outdoor_sun(sun_elevation=10, turbidity=3.5)
        >>> renderer = forge3d.Renderer(1920, 1080, **config)
    """
    import math

    # Calculate sun direction from angles
    el_rad = math.radians(sun_elevation)
    az_rad = math.radians(sun_azimuth)
    sun_dir = [
        math.cos(el_rad) * math.sin(az_rad),
        math.sin(el_rad),
        math.cos(el_rad) * math.cos(az_rad),
    ]

    config = {
        # Lighting
        'lighting': {
            'lights': [
                {
                    'type': 'directional',
                    'direction': sun_dir,
                    'color': [1.0, 0.95, 0.9],  # Slightly warm sun
                    'intensity': 5.0,
                }
            ],
        },

        # Material shading
        'shading': {
            'brdf': 'cooktorrance-ggx',
            'roughness': 0.6,
            'metallic': 0.0,
        },

        # Shadows
        'shadows': {
            'enabled': True,
            'technique': 'CSM',
            'map_res': 2048,
            'cascades': cascades,
            'bias': 0.002,
            'normal_bias': 0.01,
        },

        # Global Illumination
        'gi': {
            'technique': 'None',  # Sky provides ambient
        },

        # Atmosphere
        'atmosphere': {
            'fog_density': fog_density,
            'exposure': 1.2,
            'sky_model': 'HosekWilkie',
        },

        # Sky settings (if available)
        'sky': {
            'model': 'hosek-wilkie',
            'turbidity': turbidity,
            'ground_albedo': ground_albedo,
            'sun_intensity': 20.0,
            'exposure': 1.0,
        },

        # Volumetric (if fog enabled)
        'volumetric': {
            'density': fog_density,
            'height_falloff': 0.1,
            'phase_g': 0.7,
            'use_shadows': True if fog_density > 0 else False,
        } if fog_density > 0 else None,
    }

    # Merge overrides
    config.update(overrides)
    return config


def toon_viz(
    outline_width: float = 2.0,
    shade_steps: int = 3,
    light_intensity: float = 2.0,
    **overrides
) -> Dict[str, Any]:
    """
    Toon/cel-shaded visualization preset (P7)

    Stylized non-photorealistic rendering with:
    - Toon BRDF for stepped shading
    - Hard shadows for crisp edges
    - No global illumination (pure local shading)
    - High contrast lighting

    Perfect for: Stylized graphics, technical illustrations, NPR rendering

    Args:
        outline_width: Edge outline width [0-5], default 2.0
        shade_steps: Number of discrete shade levels [2-5], default 3
        light_intensity: Main light intensity [0-10], default 2.0
        **overrides: Additional config overrides

    Returns:
        Configuration dictionary ready for Renderer(**config)

    Example:
        >>> from forge3d.presets import toon_viz
        >>> renderer = forge3d.Renderer(800, 600, **toon_viz(shade_steps=4))
    """
    config = {
        # Lighting
        'lighting': {
            'lights': [
                {
                    'type': 'directional',
                    'direction': [0.3, -0.6, 0.5],
                    'color': [1.0, 1.0, 1.0],
                    'intensity': light_intensity,
                }
            ],
        },

        # Material shading
        'shading': {
            'brdf': 'toon',
            'roughness': 0.5,  # Controls shade step positions
            'metallic': 0.0,
        },

        # Shadows
        'shadows': {
            'enabled': True,
            'technique': 'Hard',
            'map_res': 1024,
            'bias': 0.001,
        },

        # Global Illumination
        'gi': {
            'technique': 'None',
        },

        # Atmosphere
        'atmosphere': {
            'fog_density': 0.0,
            'exposure': 1.0,
            'sky_model': 'Off',
        },
    }

    # Merge overrides
    config.update(overrides)
    return config


def minimal(
    **overrides
) -> Dict[str, Any]:
    """
    Minimal lighting preset for fast previews (P7)

    Bare-bones setup with:
    - Single directional light
    - Lambert BRDF (fastest)
    - No shadows
    - No GI

    Perfect for: Quick previews, performance testing, debugging

    Args:
        **overrides: Additional config overrides

    Returns:
        Configuration dictionary ready for Renderer(**config)

    Example:
        >>> from forge3d.presets import minimal
        >>> renderer = forge3d.Renderer(640, 480, **minimal())
    """
    config = {
        'lighting': {
            'lights': [
                {
                    'type': 'directional',
                    'direction': [0.0, -1.0, 0.0],
                    'color': [1.0, 1.0, 1.0],
                    'intensity': 2.0,
                }
            ],
        },
        'shading': {
            'brdf': 'lambert',
            'roughness': 1.0,
            'metallic': 0.0,
        },
        'shadows': {
            'enabled': False,
        },
        'gi': {
            'technique': 'None',
        },
        'atmosphere': {
            'fog_density': 0.0,
            'exposure': 1.0,
            'sky_model': 'Off',
        },
    }

    config.update(overrides)
    return config


def high_quality(
    ibl_path: Optional[str] = None,
    shadow_map_res: int = 4096,
    **overrides
) -> Dict[str, Any]:
    """
    High-quality rendering preset for final renders (P7)

    Maximum quality settings with:
    - IBL with high-resolution environment maps
    - Disney Principled BRDF
    - PCSS soft shadows (high sample count)
    - SSAO for ambient occlusion detail
    - Optional SSR for reflections

    Perfect for: Final renders, marketing materials, hero shots

    Args:
        ibl_path: Path to HDR environment map, optional
        shadow_map_res: Shadow map resolution [2048-8192], default 4096
        **overrides: Additional config overrides

    Returns:
        Configuration dictionary ready for Renderer(**config)

    Example:
        >>> from forge3d.presets import high_quality
        >>> config = high_quality(ibl_path='assets/studio.hdr')
        >>> renderer = forge3d.Renderer(2048, 2048, **config)
    """
    config = {
        'lighting': {
            'lights': [
                {
                    'type': 'directional',
                    'direction': [0.5, -0.8, 0.3],
                    'color': [1.0, 1.0, 1.0],
                    'intensity': 3.0,
                }
            ],
        },
        'shading': {
            'brdf': 'disney-principled',
            'roughness': 0.4,
            'metallic': 0.0,
            'clearcoat': 0.1,
            'sheen': 0.0,
        },
        'shadows': {
            'enabled': True,
            'technique': 'PCSS',
            'map_res': shadow_map_res,
            'pcss_blocker_radius': 8.0,
            'pcss_filter_radius': 12.0,
            'light_size': 0.5,
            'moment_bias': 0.0005,
        },
        'gi': {
            'technique': 'IBL',
            'ibl_intensity': 1.2,
            'ibl_rotation': 0.0,
        },
        'atmosphere': {
            'fog_density': 0.0,
            'exposure': 1.0,
            'sky_model': 'Off',
        },
        # Screen-space effects
        'ssao': {
            'enabled': True,
            'radius': 0.5,
            'intensity': 1.0,
            'technique': 'GTAO',
            'sample_count': 32,
        },
    }

    if ibl_path:
        config['gi']['ibl_path'] = ibl_path

    config.update(overrides)
    return config


# Preset registry for programmatic access
PRESETS = {
    'studio_pbr': studio_pbr,
    'outdoor_sun': outdoor_sun,
    'toon_viz': toon_viz,
    'minimal': minimal,
    'high_quality': high_quality,
}


def list_presets() -> list:
    """
    List all available preset names

    Returns:
        List of preset name strings

    Example:
        >>> from forge3d.presets import list_presets
        >>> print(list_presets())
        ['studio_pbr', 'outdoor_sun', 'toon_viz', 'minimal', 'high_quality']
    """
    return list(PRESETS.keys())


def get_preset(name: str, **kwargs) -> Dict[str, Any]:
    """
    Get a preset configuration by name

    Args:
        name: Preset name (see list_presets())
        **kwargs: Preset-specific arguments and overrides

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If preset name is unknown

    Example:
        >>> from forge3d.presets import get_preset
        >>> config = get_preset('outdoor_sun', turbidity=3.5)
        >>> renderer = forge3d.Renderer(800, 600, **config)
    """
    if name not in PRESETS:
        available = ', '.join(list_presets())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return PRESETS[name](**kwargs)


def describe_preset(name: str) -> str:
    """
    Get human-readable description of a preset

    Args:
        name: Preset name

    Returns:
        Multiline description string

    Example:
        >>> from forge3d.presets import describe_preset
        >>> print(describe_preset('studio_pbr'))
    """
    if name not in PRESETS:
        available = ', '.join(list_presets())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    func = PRESETS[name]
    return func.__doc__ or f"No description available for preset '{name}'"
