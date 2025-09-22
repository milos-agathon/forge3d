"""
Post-processing effects API for forge3d.

This module provides high-level Python access to the post-processing compute pipeline,
allowing applications to enable, configure, and chain GPU-based post-processing effects.

The post-processing system provides:
- Effect chain management with automatic resource handling
- Compute-based effects with optimal GPU utilization
- Temporal effects with ping-pong buffer management
- Parameter control and real-time adjustment
- Integration with GPU timing for performance monitoring

Usage:
    import forge3d.postfx as postfx
    
    # Enable post-processing for a renderer
    postfx.enable_postfx_chain(renderer)
    
    # Enable specific effects
    postfx.enable("bloom", strength=0.6, threshold=1.0)
    postfx.enable("tonemap", exposure=1.2, gamma=2.2)
    
    # Configure effect parameters
    postfx.set_parameter("bloom", "strength", 0.8)
    
    # List available effects
    effects = postfx.list_available_effects()
    
    # Disable effects
    postfx.disable("bloom")
"""

from typing import Dict, Any, List, Optional, Union
import warnings


class PostFxConfig:
    """Configuration for a post-processing effect."""
    
    def __init__(self,
                 name: str,
                 enabled: bool = True,
                 parameters: Optional[Dict[str, float]] = None,
                 priority: int = 0,
                 temporal: bool = False):
        """Initialize post-processing effect configuration.
        
        Args:
            name: Effect name/identifier
            enabled: Whether the effect is enabled
            parameters: Effect-specific parameters
            priority: Priority for effect ordering (higher = later)
            temporal: Whether this effect needs temporal data
        """
        self.name = name
        self.enabled = enabled
        self.parameters = parameters or {}
        self.priority = priority
        self.temporal = temporal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for native interface."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'parameters': self.parameters.copy(),
            'priority': self.priority,
            'temporal': self.temporal,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostFxConfig':
        """Create from dictionary."""
        return cls(
            name=data.get('name', ''),
            enabled=data.get('enabled', True),
            parameters=data.get('parameters', {}),
            priority=data.get('priority', 0),
            temporal=data.get('temporal', False),
        )


class PostFxChainManager:
    """Manager for post-processing effect chains."""
    
    def __init__(self):
        self._enabled_effects: Dict[str, PostFxConfig] = {}
        self._available_effects: Dict[str, Dict[str, Any]] = {}
        self._chain_enabled: bool = True
        self._timing_stats: Dict[str, float] = {}
        self._initialize_available_effects()
    
    def _initialize_available_effects(self):
        """Initialize the catalog of available effects."""
        self._available_effects = {
            'tonemap': {
                'description': 'HDR tone mapping with exposure control',
                'parameters': {
                    'exposure': {'default': 1.0, 'min': 0.1, 'max': 10.0},
                    'gamma': {'default': 2.2, 'min': 1.0, 'max': 4.0},
                },
                'priority': 1000,  # Usually last
                'temporal': False,
            },
            'bloom': {
                'description': 'Bloom effect with bright-pass filtering',
                'parameters': {
                    'threshold': {'default': 1.0, 'min': 0.0, 'max': 5.0},
                    'strength': {'default': 0.5, 'min': 0.0, 'max': 2.0},
                    'radius': {'default': 1.0, 'min': 0.1, 'max': 3.0},
                },
                'priority': 800,
                'temporal': False,
            },
            'blur': {
                'description': 'Simple box blur effect',
                'parameters': {
                    'strength': {'default': 1.0, 'min': 0.0, 'max': 5.0},
                },
                'priority': 100,
                'temporal': False,
            },
            'ssao': {
                'description': 'Screen-space ambient occlusion with bilateral blur',
                'parameters': {
                    'radius': {'default': 1.0, 'min': 0.1, 'max': 5.0},
                    'intensity': {'default': 1.0, 'min': 0.0, 'max': 5.0},
                    'bias': {'default': 0.025, 'min': 0.0, 'max': 0.2},
                },
                'priority': 400,
                'temporal': False,
            },
            'sharpen': {
                'description': 'Unsharp mask sharpening',
                'parameters': {
                    'strength': {'default': 0.5, 'min': 0.0, 'max': 2.0},
                    'radius': {'default': 1.0, 'min': 0.1, 'max': 3.0},
                },
                'priority': 200,
                'temporal': False,
            },
            'fxaa': {
                'description': 'Fast Approximate Anti-Aliasing',
                'parameters': {
                    'quality': {'default': 1.0, 'min': 0.0, 'max': 1.0},
                },
                'priority': 900,
                'temporal': False,
            },
            'temporal_aa': {
                'description': 'Temporal anti-aliasing',
                'parameters': {
                    'blend_factor': {'default': 0.9, 'min': 0.1, 'max': 0.99},
                },
                'priority': 850,
                'temporal': True,
            },
        }
    
    def enable(self, name: str, **kwargs) -> bool:
        """Enable a post-processing effect with parameters.
        
        Args:
            name: Name of the effect to enable
            **kwargs: Effect-specific parameters
            
        Returns:
            True if effect was enabled successfully
        """
        if name not in self._available_effects:
            warnings.warn(f"Unknown post-processing effect: {name}")
            return False
        
        effect_info = self._available_effects[name]
        
        # Build parameters with defaults and user overrides
        parameters = {}
        for param_name, param_info in effect_info['parameters'].items():
            default_value = param_info['default']
            user_value = kwargs.get(param_name, default_value)
            
            # Validate parameter range
            if 'min' in param_info and user_value < param_info['min']:
                warnings.warn(f"Parameter {param_name}={user_value} below minimum {param_info['min']}")
                user_value = param_info['min']
            if 'max' in param_info and user_value > param_info['max']:
                warnings.warn(f"Parameter {param_name}={user_value} above maximum {param_info['max']}")
                user_value = param_info['max']
            
            parameters[param_name] = user_value
        
        config = PostFxConfig(
            name=name,
            enabled=True,
            parameters=parameters,
            priority=effect_info['priority'],
            temporal=effect_info['temporal']
        )
        
        self._enabled_effects[name] = config
        return True
    
    def disable(self, name: str) -> bool:
        """Disable a post-processing effect.
        
        Args:
            name: Name of the effect to disable
            
        Returns:
            True if effect was disabled successfully
        """
        if name in self._enabled_effects:
            del self._enabled_effects[name]
            return True
        return False
    
    def set_parameter(self, effect_name: str, param_name: str, value: float) -> bool:
        """Set parameter for an enabled effect.
        
        Args:
            effect_name: Name of the effect
            param_name: Name of the parameter
            value: New parameter value
            
        Returns:
            True if parameter was set successfully
        """
        if effect_name not in self._enabled_effects:
            return False
        
        # Validate parameter exists and range
        if effect_name in self._available_effects:
            effect_info = self._available_effects[effect_name]
            if param_name in effect_info['parameters']:
                param_info = effect_info['parameters'][param_name]
                
                if 'min' in param_info and value < param_info['min']:
                    value = param_info['min']
                if 'max' in param_info and value > param_info['max']:
                    value = param_info['max']
        
        self._enabled_effects[effect_name].parameters[param_name] = value
        return True
    
    def get_parameter(self, effect_name: str, param_name: str) -> Optional[float]:
        """Get parameter value for an enabled effect.
        
        Args:
            effect_name: Name of the effect
            param_name: Name of the parameter
            
        Returns:
            Parameter value or None if not found
        """
        if effect_name not in self._enabled_effects:
            return None
        
        return self._enabled_effects[effect_name].parameters.get(param_name)
    
    def list_enabled_effects(self) -> List[str]:
        """Get list of enabled effects in execution order.
        
        Returns:
            List of effect names sorted by priority
        """
        effects = [(name, config) for name, config in self._enabled_effects.items()]
        effects.sort(key=lambda x: x[1].priority)
        return [name for name, config in effects]
    
    def list_available_effects(self) -> List[str]:
        """Get list of all available effects.
        
        Returns:
            List of available effect names
        """
        return [name for name in self._available_effects.keys()]
    
    def get_effect_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about an available effect.
        
        Args:
            name: Effect name
            
        Returns:
            Effect information dictionary or None if not found
        """
        return self._available_effects.get(name)
    
    def set_chain_enabled(self, enabled: bool):
        """Enable or disable the entire post-processing chain.
        
        Args:
            enabled: Whether to enable the chain
        """
        self._chain_enabled = enabled
    
    def is_chain_enabled(self) -> bool:
        """Check if post-processing chain is enabled.
        
        Returns:
            True if chain is enabled
        """
        return self._chain_enabled
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics for effects.
        
        Returns:
            Dictionary mapping effect names to GPU time in milliseconds
        """
        return self._timing_stats.copy()
    
    def update_timing_stats(self, stats: Dict[str, float]):
        """Update timing statistics (called by native code).
        
        Args:
            stats: New timing statistics
        """
        self._timing_stats = stats.copy()
    
    def get_chain_config(self) -> Dict[str, Any]:
        """Get complete chain configuration for native interface.
        
        Returns:
            Chain configuration dictionary
        """
        return {
            'enabled': self._chain_enabled,
            'effects': [config.to_dict() for config in self._enabled_effects.values()],
            'execution_order': self.list_enabled_effects(),
        }


# Global post-processing chain manager
_postfx_manager = PostFxChainManager()

# Public API functions

def enable_ssao(*, radius: float = 1.0, intensity: float = 1.0, bias: float = 0.025, scene=None) -> bool:
    """Enable SSAO and optionally configure a scene instance."""
    if scene is not None:
        if hasattr(scene, 'set_ssao_parameters'):
            try:
                scene.set_ssao_parameters(radius, intensity, bias)
            except Exception:  # pragma: no cover - defensive fallback
                warnings.warn('Scene.set_ssao_parameters failed', stacklevel=2)
        if hasattr(scene, 'set_ssao_enabled'):
            try:
                scene.set_ssao_enabled(True)
            except Exception:  # pragma: no cover - defensive fallback
                warnings.warn('Scene.set_ssao_enabled failed', stacklevel=2)
    return enable('ssao', radius=radius, intensity=intensity, bias=bias)

def disable_ssao(scene=None) -> bool:
    """Disable SSAO and optionally update a scene instance."""
    if scene is not None and hasattr(scene, 'set_ssao_enabled'):
        try:
            scene.set_ssao_enabled(False)
        except Exception:  # pragma: no cover - defensive fallback
            warnings.warn('Scene.set_ssao_enabled failed', stacklevel=2)
    return disable('ssao')

def enable(name: str, **kwargs) -> bool:
    """Enable a post-processing effect with parameters.
    
    Args:
        name: Name of the effect to enable
        **kwargs: Effect-specific parameters
        
    Returns:
        True if effect was enabled successfully
        
    Examples:
        >>> postfx.enable("bloom", threshold=1.2, strength=0.8)
        >>> postfx.enable("tonemap", exposure=1.5, gamma=2.2)
    """
    return _postfx_manager.enable(name, **kwargs)


def disable(name: str) -> bool:
    """Disable a post-processing effect.
    
    Args:
        name: Name of the effect to disable
        
    Returns:
        True if effect was disabled successfully
    """
    return _postfx_manager.disable(name)


def set_parameter(effect_name: str, param_name: str, value: float) -> bool:
    """Set parameter for an enabled effect.
    
    Args:
        effect_name: Name of the effect
        param_name: Name of the parameter  
        value: New parameter value
        
    Returns:
        True if parameter was set successfully
        
    Examples:
        >>> postfx.set_parameter("bloom", "strength", 1.2)
        >>> postfx.set_parameter("tonemap", "exposure", 0.8)
    """
    return _postfx_manager.set_parameter(effect_name, param_name, value)


def get_parameter(effect_name: str, param_name: str) -> Optional[float]:
    """Get parameter value for an enabled effect.
    
    Args:
        effect_name: Name of the effect
        param_name: Name of the parameter
        
    Returns:
        Parameter value or None if not found
    """
    return _postfx_manager.get_parameter(effect_name, param_name)



def list_enabled_effects() -> List[str]:
    """Alias for list() to improve readability while SSAO scaffolding lands."""
    return list()

def list() -> List[str]:
    """Get list of enabled effects in execution order.
    
    Returns:
        List of effect names sorted by priority
    """
    return _postfx_manager.list_enabled_effects()


def list_available() -> List[str]:
    """Get list of all available effects.
    
    Returns:
        List of available effect names
    """
    return _postfx_manager.list_available_effects()


def get_effect_info(name: str) -> Optional[Dict[str, Any]]:
    """Get information about an available effect.
    
    Args:
        name: Effect name
        
    Returns:
        Effect information dictionary or None if not found
    """
    return _postfx_manager.get_effect_info(name)


def set_chain_enabled(enabled: bool):
    """Enable or disable the entire post-processing chain.
    
    Args:
        enabled: Whether to enable the chain
    """
    _postfx_manager.set_chain_enabled(enabled)


def is_chain_enabled() -> bool:
    """Check if post-processing chain is enabled.
    
    Returns:
        True if chain is enabled
    """
    return _postfx_manager.is_chain_enabled()


def get_timing_stats() -> Dict[str, float]:
    """Get timing statistics for effects.
    
    Returns:
        Dictionary mapping effect names to GPU time in milliseconds
    """
    return _postfx_manager.get_timing_stats()


def create_preset(name: str, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a post-processing preset.
    
    Args:
        name: Preset name
        effects: List of effect configurations
        
    Returns:
        Preset configuration dictionary
    """
    return {
        'name': name,
        'effects': effects,
    }


# Built-in presets
PRESETS = {
    'cinematic': [
        {'name': 'bloom', 'parameters': {'threshold': 1.2, 'strength': 0.6}},
        {'name': 'tonemap', 'parameters': {'exposure': 1.1, 'gamma': 2.2}},
    ],
    'sharp': [
        {'name': 'sharpen', 'parameters': {'strength': 0.8}},
        {'name': 'fxaa', 'parameters': {'quality': 1.0}},
        {'name': 'tonemap', 'parameters': {'exposure': 1.0, 'gamma': 2.2}},
    ],
    'performance': [
        {'name': 'fxaa', 'parameters': {'quality': 0.5}},
        {'name': 'tonemap', 'parameters': {'exposure': 1.0, 'gamma': 2.2}},
    ],
    'quality': [
        {'name': 'temporal_aa', 'parameters': {'blend_factor': 0.9}},
        {'name': 'bloom', 'parameters': {'threshold': 1.0, 'strength': 0.4}},
        {'name': 'tonemap', 'parameters': {'exposure': 1.0, 'gamma': 2.2}},
    ],
}


def apply_preset(preset_name: str) -> bool:
    """Apply a built-in post-processing preset.
    
    Args:
        preset_name: Name of the preset to apply
        
    Returns:
        True if preset was applied successfully
    """
    if preset_name not in PRESETS:
        return False
    
    # Clear existing effects
    for effect_name in _postfx_manager.list_enabled_effects():
        disable(effect_name)
    
    # Apply preset effects
    preset = PRESETS[preset_name]
    for effect_config in preset:
        name = effect_config['name']
        parameters = effect_config.get('parameters', {})
        enable(name, **parameters)
    
    return True


def list_presets() -> List[str]:
    """Get list of available presets.
    
    Returns:
        List of preset names
    """
    return [name for name in PRESETS.keys()]


# Renderer-level PostFX toggle (Python-side helper)
_RENDERER_POSTFX_STATE: Dict[int, bool] = {}


def set_renderer_postfx_enabled(renderer: Any, enabled: bool) -> None:
    """Enable/disable PostFX chain for a renderer (Python-side helper).

    If the native method exists (Renderer.set_postfx_enabled), it is used.
    Otherwise a Python attribute is set for higher-level tools to inspect.
    """
    if hasattr(renderer, "set_postfx_enabled"):
        renderer.set_postfx_enabled(bool(enabled))
    else:
        _RENDERER_POSTFX_STATE[id(renderer)] = bool(enabled)


def is_renderer_postfx_enabled(renderer: Any) -> bool:
    """Check if PostFX is enabled on a renderer (Python-side helper)."""
    if hasattr(renderer, "is_postfx_enabled"):
        return bool(renderer.is_postfx_enabled())
    return bool(_RENDERER_POSTFX_STATE.get(id(renderer), False))
