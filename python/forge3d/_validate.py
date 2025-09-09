# T01-BEGIN:validate
from __future__ import annotations
from pathlib import Path
from typing import Tuple

_MAX_DIM = 8192  # conservative guardrail for headless targets

def _as_int(name: str, v) -> int:
    try:
        i = int(v)
    except Exception as e:
        raise ValueError(f"{name} must be an integer, got {type(v).__name__}") from e
    return i

def size_wh(width, height) -> Tuple[int, int]:
    w = _as_int("width", width)
    h = _as_int("height", height)
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be > 0")
    if w > _MAX_DIM or h > _MAX_DIM:
        raise ValueError(f"width/height must be <= {_MAX_DIM}")
    return w, h

def grid(n) -> int:
    g = _as_int("grid", n)
    if g < 2:
        raise ValueError("grid must be >= 2")
    if g > 4096:
        raise ValueError("grid must be <= 4096")
    return g

def png_path(p: str | Path) -> str:
    s = str(p)
    if not s.lower().endswith(".png"):
        raise ValueError("path must end with .png")
    parent = Path(s).resolve().parent
    if not parent.exists():
        raise ValueError(f"directory does not exist: {parent}")
    return s
# T01-END:validate

# Zero-copy validation helpers
import numpy as np
from typing import Dict, List, Any

def ptr(arr: np.ndarray) -> int:
    """Get the underlying data pointer from a NumPy array."""
    return arr.ctypes.data

def is_c_contiguous(arr: np.ndarray) -> bool:
    """Check if a NumPy array is C-contiguous."""
    return arr.flags['C_CONTIGUOUS']

def validate_zero_copy_path(arr: np.ndarray, context: str = "") -> Dict[str, Any]:
    """Validate that an array is suitable for zero-copy operations."""
    issues = []
    
    if not is_c_contiguous(arr):
        issues.append("Array is not C-contiguous")
    
    if arr.data.nbytes == 0:
        issues.append("Array has no data")
    
    data_ptr = ptr(arr)
    if data_ptr == 0:
        issues.append("Array has null data pointer")
    
    return {
        'compatible': len(issues) == 0,
        'issues': issues,
        'context': context,
        'dtype': str(arr.dtype),
        'shape': arr.shape,
        'strides': arr.strides,
        'flags': {
            'c_contiguous': arr.flags['C_CONTIGUOUS'],
            'f_contiguous': arr.flags['F_CONTIGUOUS'],
            'owndata': arr.flags['OWNDATA'],
            'writeable': arr.flags['WRITEABLE']
        },
        'data_ptr': data_ptr
    }

def check_zero_copy_compatibility(arr: np.ndarray, context: str = "") -> Dict[str, Any]:
    """Check zero-copy compatibility and return detailed analysis."""
    return validate_zero_copy_path(arr, context)


# ============================================================================
# COMPREHENSIVE INPUT VALIDATION - R12 IMPLEMENTATION
# ============================================================================

from typing import Optional, Literal, Sequence

# Validation constants
MAX_ARRAY_DIM = 8192  # Maximum array dimension (aligned with _MAX_DIM)
VALID_DTYPES_FLOAT = [np.float32, np.float64]
VALID_DTYPES_UINT8 = [np.uint8]
VALID_DTYPES_NUMERIC = VALID_DTYPES_FLOAT + [np.int32, np.int64, np.uint32, np.uint64, np.uint8, np.uint16]

# Common shape validation patterns
SHAPE_2D = "2D"  # (H, W)
SHAPE_3D_RGB = "3D_RGB"  # (H, W, 3)
SHAPE_3D_RGBA = "3D_RGBA"  # (H, W, 4)
SHAPE_HEIGHT_FIELD = "HEIGHT_FIELD"  # (H, W) float32/float64


def validate_array(
    arr: np.ndarray,
    name: str,
    dtype: Optional[Sequence[type]] = None,
    shape: Optional[str] = None,
    min_size: int = 1,
    max_size: Optional[int] = None,
    require_contiguous: bool = True,
    context: str = ""
) -> np.ndarray:
    """
    Comprehensive array validation with detailed error messages.
    
    Args:
        arr: NumPy array to validate
        name: Parameter name for error messages
        dtype: Allowed dtypes (None = any numeric)
        shape: Expected shape pattern (see SHAPE_* constants)
        min_size: Minimum total array size
        max_size: Maximum total array size (None = no limit)
        require_contiguous: Require C-contiguous memory layout
        context: Additional context for error messages
        
    Returns:
        Validated array (possibly converted dtype)
        
    Raises:
        TypeError: Invalid array type
        ValueError: Invalid shape, dtype, or memory layout
    """
    # Type validation
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} must be numpy.ndarray, got {type(arr).__name__}. "
            f"Context: {context}"
        )
    
    # Size validation
    if arr.size < min_size:
        raise ValueError(
            f"{name} array too small: {arr.size} elements < {min_size} minimum. "
            f"Shape: {arr.shape}. Context: {context}"
        )
    
    if max_size and arr.size > max_size:
        raise ValueError(
            f"{name} array too large: {arr.size} elements > {max_size} maximum. "
            f"Shape: {arr.shape}. Context: {context}"
        )
    
    # Memory layout validation
    if require_contiguous and not arr.flags['C_CONTIGUOUS']:
        raise ValueError(
            f"{name} must be C-contiguous (row-major). "
            f"Use np.ascontiguousarray() to fix. Context: {context}"
        )
    
    # Shape pattern validation
    if shape:
        _validate_shape_pattern(arr, name, shape, context)
    
    # Dtype validation
    if dtype:
        if arr.dtype not in dtype:
            valid_dtypes = [str(dt).replace("class ", "").replace("<", "").replace(">", "") for dt in dtype]
            raise ValueError(
                f"{name} has invalid dtype: {arr.dtype}. "
                f"Expected one of: {valid_dtypes}. Context: {context}"
            )
    else:
        # Default: require numeric type
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"{name} must have numeric dtype, got {arr.dtype}. "
                f"Context: {context}"
            )
    
    return arr


def _validate_shape_pattern(arr: np.ndarray, name: str, pattern: str, context: str) -> None:
    """Validate array shape matches expected pattern."""
    shape = arr.shape
    ndim = arr.ndim
    
    if pattern == SHAPE_2D:
        if ndim != 2:
            raise ValueError(
                f"{name} must be 2D array (H, W), got {ndim}D shape {shape}. "
                f"Context: {context}"
            )
        h, w = shape
        if h <= 0 or w <= 0:
            raise ValueError(
                f"{name} dimensions must be > 0, got {shape}. Context: {context}"
            )
        if h > MAX_ARRAY_DIM or w > MAX_ARRAY_DIM:
            raise ValueError(
                f"{name} dimensions exceed limit: {shape} > {MAX_ARRAY_DIM}. "
                f"Context: {context}"
            )
    
    elif pattern == SHAPE_3D_RGB:
        if ndim != 3 or shape[2] != 3:
            raise ValueError(
                f"{name} must be 3D RGB array (H, W, 3), got shape {shape}. "
                f"Context: {context}"
            )
        h, w, c = shape
        if h <= 0 or w <= 0:
            raise ValueError(
                f"{name} spatial dimensions must be > 0, got (H={h}, W={w}). "
                f"Context: {context}"
            )
            
    elif pattern == SHAPE_3D_RGBA:
        if ndim != 3 or shape[2] != 4:
            raise ValueError(
                f"{name} must be 3D RGBA array (H, W, 4), got shape {shape}. "
                f"Context: {context}"
            )
        h, w, c = shape
        if h <= 0 or w <= 0:
            raise ValueError(
                f"{name} spatial dimensions must be > 0, got (H={h}, W={w}). "
                f"Context: {context}"
            )
            
    elif pattern == SHAPE_HEIGHT_FIELD:
        if ndim != 2:
            raise ValueError(
                f"{name} height field must be 2D array (H, W), got {ndim}D shape {shape}. "
                f"Context: {context}"
            )
        h, w = shape
        if h < 2 or w < 2:
            raise ValueError(
                f"{name} height field too small: {shape}. Minimum 2x2. "
                f"Context: {context}"
            )
        if h > MAX_ARRAY_DIM or w > MAX_ARRAY_DIM:
            raise ValueError(
                f"{name} height field too large: {shape} > {MAX_ARRAY_DIM}. "
                f"Context: {context}"
            )
    
    else:
        raise ValueError(f"Unknown shape pattern: {pattern}")


def validate_numeric_parameter(
    value: Any,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    expected_type: type = float,
    context: str = ""
) -> Union[float, int]:
    """
    Validate numeric parameter with range checking.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        expected_type: Expected return type (float or int)
        context: Additional context for error messages
        
    Returns:
        Validated value converted to expected_type
        
    Raises:
        TypeError: Invalid type
        ValueError: Out of range
    """
    # Type conversion and validation
    try:
        if expected_type == float:
            converted = float(value)
        elif expected_type == int:
            converted = int(value)
        else:
            raise ValueError(f"Unsupported expected_type: {expected_type}")
    except (TypeError, ValueError, OverflowError) as e:
        raise TypeError(
            f"{name} must be convertible to {expected_type.__name__}, "
            f"got {type(value).__name__}: {value}. Context: {context}"
        ) from e
    
    # Range validation
    if min_val is not None and converted < min_val:
        raise ValueError(
            f"{name} too small: {converted} < {min_val}. Context: {context}"
        )
    
    if max_val is not None and converted > max_val:
        raise ValueError(
            f"{name} too large: {converted} > {max_val}. Context: {context}"
        )
    
    return converted


def validate_color_tuple(
    color: Any,
    name: str,
    num_channels: int = 3,
    value_range: Tuple[float, float] = (0.0, 1.0),
    context: str = ""
) -> Tuple[float, ...]:
    """
    Validate color tuple with range checking.
    
    Args:
        color: Color tuple to validate
        name: Parameter name for error messages
        num_channels: Expected number of channels (3=RGB, 4=RGBA)
        value_range: Valid range for each channel
        context: Additional context for error messages
        
    Returns:
        Validated color tuple
        
    Raises:
        TypeError: Invalid type
        ValueError: Invalid length or values out of range
    """
    # Type validation
    if not isinstance(color, (tuple, list, np.ndarray)):
        raise TypeError(
            f"{name} must be tuple, list, or array, got {type(color).__name__}. "
            f"Context: {context}"
        )
    
    # Length validation
    if len(color) != num_channels:
        raise ValueError(
            f"{name} must have {num_channels} channels, got {len(color)}. "
            f"Expected format: {'RGB' if num_channels == 3 else 'RGBA'}. "
            f"Context: {context}"
        )
    
    # Convert and validate each channel
    validated_channels = []
    min_val, max_val = value_range
    
    for i, channel in enumerate(color):
        try:
            ch_val = float(channel)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"{name}[{i}] must be numeric, got {type(channel).__name__}: {channel}. "
                f"Context: {context}"
            ) from e
        
        if ch_val < min_val or ch_val > max_val:
            raise ValueError(
                f"{name}[{i}] out of range: {ch_val} not in [{min_val}, {max_val}]. "
                f"Context: {context}"
            )
        
        validated_channels.append(ch_val)
    
    return tuple(validated_channels)


def validate_string_choice(
    value: Any,
    name: str,
    choices: List[str],
    case_sensitive: bool = True,
    context: str = ""
) -> str:
    """
    Validate string parameter against allowed choices.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        choices: List of valid choices
        case_sensitive: Whether comparison is case-sensitive
        context: Additional context for error messages
        
    Returns:
        Validated string (original case preserved)
        
    Raises:
        TypeError: Invalid type
        ValueError: Invalid choice
    """
    # Type validation
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be string, got {type(value).__name__}: {value}. "
            f"Context: {context}"
        )
    
    # Choice validation
    if case_sensitive:
        if value not in choices:
            raise ValueError(
                f"{name} invalid choice: '{value}'. "
                f"Valid choices: {choices}. Context: {context}"
            )
    else:
        value_lower = value.lower()
        choices_lower = [c.lower() for c in choices]
        if value_lower not in choices_lower:
            raise ValueError(
                f"{name} invalid choice: '{value}'. "
                f"Valid choices (case-insensitive): {choices}. Context: {context}"
            )
    
    return value
