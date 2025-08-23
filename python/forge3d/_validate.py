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