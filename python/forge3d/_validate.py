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

# Zero-copy validation helpers for NumPy interop
import numpy as np
from typing import Union, Optional


def ptr(np_array: np.ndarray) -> int:
    """
    Get the underlying data pointer from a NumPy array.
    
    Args:
        np_array: NumPy array to extract pointer from
        
    Returns:
        int: Memory address of the array's data buffer
        
    Examples:
        >>> arr = np.ones((3, 4), dtype=np.float32)
        >>> p = ptr(arr)
        >>> isinstance(p, int) and p != 0
        True
    """
    if not isinstance(np_array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(np_array)}")
    
    return np_array.ctypes.data


def is_c_contiguous(np_array: np.ndarray) -> bool:
    """
    Check if a NumPy array is C-contiguous (row-major layout).
    
    Args:
        np_array: NumPy array to check
        
    Returns:
        bool: True if array is C-contiguous, False otherwise
        
    Examples:
        >>> arr = np.ones((3, 4), dtype=np.float32)
        >>> is_c_contiguous(arr)
        True
        >>> is_c_contiguous(arr.T)  # Transpose is not C-contiguous
        False
    """
    if not isinstance(np_array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(np_array)}")
    
    return np_array.flags['C_CONTIGUOUS']


def validate_zero_copy_path(
    name: str, 
    want_ptr: int, 
    got_ptr: int, 
    tolerance_bytes: int = 0
) -> None:
    """
    Validate that zero-copy pathway was used by comparing memory pointers.
    
    Args:
        name: Descriptive name for the operation being validated
        want_ptr: Expected memory pointer (reference)
        got_ptr: Actual memory pointer (obtained)
        tolerance_bytes: Allowed byte offset between pointers (default: 0 for exact match)
        
    Raises:
        RuntimeError: If pointers don't match within tolerance, indicating a copy occurred
        
    Examples:
        >>> arr1 = np.ones(10, dtype=np.float32)
        >>> arr2 = arr1  # Same backing store
        >>> validate_zero_copy_path("test", ptr(arr1), ptr(arr2))  # Should pass
        
        >>> arr3 = arr1.copy()  # Different backing store
        >>> try:
        ...     validate_zero_copy_path("test", ptr(arr1), ptr(arr3))
        ... except RuntimeError as e:
        ...     "zero-copy validation failed" in str(e)
        True
    """
    if not isinstance(want_ptr, int) or not isinstance(got_ptr, int):
        raise TypeError("Pointers must be integers")
    
    if want_ptr == 0 or got_ptr == 0:
        raise ValueError("Invalid pointer: pointers cannot be zero")
    
    offset = abs(got_ptr - want_ptr)
    
    if offset > tolerance_bytes:
        raise RuntimeError(
            f"Zero-copy validation failed for '{name}': "
            f"expected pointer 0x{want_ptr:x}, got 0x{got_ptr:x} "
            f"(offset: {offset} bytes, tolerance: {tolerance_bytes} bytes). "
            f"This indicates an unexpected memory copy occurred. "
            f"Check that input arrays are C-contiguous and dtype-compatible."
        )


def check_zero_copy_compatibility(
    np_array: np.ndarray, 
    name: str = "array"
) -> dict:
    """
    Check if a NumPy array is compatible with zero-copy operations.
    
    Args:
        np_array: NumPy array to analyze
        name: Descriptive name for the array
        
    Returns:
        dict: Analysis results with keys:
            - 'compatible': bool, whether array is zero-copy compatible
            - 'c_contiguous': bool, whether array is C-contiguous  
            - 'data_ptr': int, memory pointer to data buffer
            - 'dtype': str, data type name
            - 'shape': tuple, array shape
            - 'strides': tuple, memory strides
            - 'issues': list of str, potential compatibility issues
    """
    if not isinstance(np_array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(np_array)}")
    
    issues = []
    c_contiguous = is_c_contiguous(np_array)
    data_ptr = ptr(np_array)
    
    # Check for common compatibility issues
    if not c_contiguous:
        issues.append("Array is not C-contiguous (row-major layout required)")
    
    if data_ptr == 0:
        issues.append("Array has invalid data pointer")
    
    if np_array.size == 0:
        issues.append("Array is empty")
        
    # Check for supported dtypes (common ones for graphics/numerical work)
    supported_dtypes = {
        np.float32, np.float64, np.uint8, np.uint16, np.uint32, 
        np.int8, np.int16, np.int32
    }
    if np_array.dtype.type not in supported_dtypes:
        issues.append(f"Array dtype {np_array.dtype} may not be supported")
    
    return {
        'compatible': len(issues) == 0,
        'c_contiguous': c_contiguous,
        'data_ptr': data_ptr,
        'dtype': str(np_array.dtype),
        'shape': np_array.shape,
        'strides': np_array.strides,
        'issues': issues
    }