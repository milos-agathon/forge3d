# python/forge3d/style_expressions.py
"""Mapbox Style Spec expression evaluation.

Implements evaluation of data-driven expressions including:
- `interpolate`: Linear/exponential interpolation between stops
- `step`: Stepped/discrete values at breakpoints
- `match`: Pattern matching on property values
- `get`: Property value lookup
- `coalesce`: First non-null value
- Math operators: +, -, *, /, %, ^
- Comparison: <, <=, >, >=
- Logic: all, any, !, case
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalContext:
    """Expression evaluation context."""
    properties: dict[str, Any] = field(default_factory=dict)
    zoom: float = 10.0
    geometry_type: Optional[str] = None


def evaluate(expr: Any, ctx: EvalContext) -> Any:
    """Evaluate a Mapbox expression.
    
    Args:
        expr: Expression value (literal or array expression).
        ctx: Evaluation context with properties and zoom.
    
    Returns:
        Evaluated result, or None if evaluation fails.
    """
    if expr is None:
        return None
    if isinstance(expr, (bool, int, float, str)):
        return expr
    if isinstance(expr, list):
        return _evaluate_array(expr, ctx)
    if isinstance(expr, dict):
        return expr
    return None


def _evaluate_array(arr: list, ctx: EvalContext) -> Any:
    """Evaluate an array-based expression."""
    if not arr:
        return None
    
    op = arr[0]
    if not isinstance(op, str):
        return None
    
    # Property access
    if op == "get":
        return _eval_get(arr, ctx)
    if op == "has":
        return _eval_has(arr, ctx)
    if op == "at":
        return _eval_at(arr, ctx)
    if op == "length":
        return _eval_length(arr, ctx)
    
    # Interpolation
    if op in ("interpolate", "interpolate-hcl", "interpolate-lab"):
        return _eval_interpolate(arr, ctx)
    if op == "step":
        return _eval_step(arr, ctx)
    
    # Pattern matching
    if op == "match":
        return _eval_match(arr, ctx)
    if op == "case":
        return _eval_case(arr, ctx)
    if op == "coalesce":
        return _eval_coalesce(arr, ctx)
    
    # Comparison
    if op == "==":
        return _eval_eq(arr, ctx)
    if op == "!=":
        return _eval_neq(arr, ctx)
    if op == "<":
        return _eval_lt(arr, ctx)
    if op == "<=":
        return _eval_lte(arr, ctx)
    if op == ">":
        return _eval_gt(arr, ctx)
    if op == ">=":
        return _eval_gte(arr, ctx)
    
    # Logic
    if op == "all":
        return _eval_all(arr, ctx)
    if op == "any":
        return _eval_any(arr, ctx)
    if op == "!":
        return _eval_not(arr, ctx)
    
    # Math
    if op == "+":
        return _eval_add(arr, ctx)
    if op == "-":
        return _eval_sub(arr, ctx)
    if op == "*":
        return _eval_mul(arr, ctx)
    if op == "/":
        return _eval_div(arr, ctx)
    if op == "%":
        return _eval_mod(arr, ctx)
    if op == "^":
        return _eval_pow(arr, ctx)
    if op == "abs":
        return _eval_abs(arr, ctx)
    if op == "ceil":
        return _eval_ceil(arr, ctx)
    if op == "floor":
        return _eval_floor(arr, ctx)
    if op == "round":
        return _eval_round(arr, ctx)
    if op == "min":
        return _eval_min(arr, ctx)
    if op == "max":
        return _eval_max(arr, ctx)
    if op == "ln":
        return _eval_ln(arr, ctx)
    if op == "log10":
        return _eval_log10(arr, ctx)
    if op == "log2":
        return _eval_log2(arr, ctx)
    if op == "sin":
        return _eval_sin(arr, ctx)
    if op == "cos":
        return _eval_cos(arr, ctx)
    if op == "tan":
        return _eval_tan(arr, ctx)
    if op == "sqrt":
        return _eval_sqrt(arr, ctx)
    
    # String
    if op == "concat":
        return _eval_concat(arr, ctx)
    if op == "downcase":
        return _eval_downcase(arr, ctx)
    if op == "upcase":
        return _eval_upcase(arr, ctx)
    
    # Type conversion
    if op == "to-number":
        return _eval_to_number(arr, ctx)
    if op == "to-string":
        return _eval_to_string(arr, ctx)
    if op == "to-boolean":
        return _eval_to_boolean(arr, ctx)
    if op == "typeof":
        return _eval_typeof(arr, ctx)
    
    # Color
    if op == "rgb":
        return _eval_rgb(arr, ctx)
    if op == "rgba":
        return _eval_rgba(arr, ctx)
    
    # Special
    if op == "zoom":
        return ctx.zoom
    if op == "geometry-type":
        return ctx.geometry_type
    if op == "literal":
        return arr[1] if len(arr) > 1 else None
    
    return None


# Property access
def _eval_get(arr: list, ctx: EvalContext) -> Any:
    if len(arr) < 2:
        return None
    key = arr[1]
    return ctx.properties.get(key)


def _eval_has(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 2:
        return False
    key = arr[1]
    return key in ctx.properties


def _eval_at(arr: list, ctx: EvalContext) -> Any:
    if len(arr) < 3:
        return None
    index = evaluate(arr[1], ctx)
    array = evaluate(arr[2], ctx)
    if isinstance(index, (int, float)) and isinstance(array, list):
        idx = int(index)
        if 0 <= idx < len(array):
            return array[idx]
    return None


def _eval_length(arr: list, ctx: EvalContext) -> Optional[int]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    if isinstance(val, (str, list)):
        return len(val)
    return None


# Interpolation
def _eval_interpolate(arr: list, ctx: EvalContext) -> Any:
    if len(arr) < 5:
        return None
    
    interp_type = arr[1]
    input_expr = arr[2]
    input_val = evaluate(input_expr, ctx)
    
    if not isinstance(input_val, (int, float)):
        return None
    
    # Parse interpolation type
    is_exponential = False
    base = 1.0
    if isinstance(interp_type, list) and interp_type:
        if interp_type[0] == "exponential" and len(interp_type) > 1:
            is_exponential = True
            base = float(interp_type[1])
    
    # Parse stops
    stops = []
    for i in range(3, len(arr) - 1, 2):
        stop = arr[i]
        val = arr[i + 1]
        if isinstance(stop, (int, float)):
            stops.append((float(stop), val))
    
    if not stops:
        return None
    
    # Find surrounding stops
    if input_val <= stops[0][0]:
        return evaluate(stops[0][1], ctx)
    if input_val >= stops[-1][0]:
        return evaluate(stops[-1][1], ctx)
    
    for i in range(len(stops) - 1):
        stop_low, val_low = stops[i]
        stop_high, val_high = stops[i + 1]
        
        if stop_low <= input_val <= stop_high:
            range_size = stop_high - stop_low
            if range_size == 0:
                t = 0.0
            elif is_exponential and base != 1.0:
                t = (base ** (input_val - stop_low) - 1) / (base ** range_size - 1)
            else:
                t = (input_val - stop_low) / range_size
            
            return _interpolate_values(
                evaluate(val_low, ctx),
                evaluate(val_high, ctx),
                t
            )
    
    return None


def _interpolate_values(a: Any, b: Any, t: float) -> Any:
    """Interpolate between two values."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_interpolate_values(va, vb, t) for va, vb in zip(a, b)]
    
    # Can't interpolate, return closest
    return a if t < 0.5 else b


def _eval_step(arr: list, ctx: EvalContext) -> Any:
    if len(arr) < 4:
        return None
    
    input_val = evaluate(arr[1], ctx)
    default = arr[2]
    
    if not isinstance(input_val, (int, float)):
        return evaluate(default, ctx)
    
    # Parse stops
    result = default
    for i in range(3, len(arr) - 1, 2):
        stop = arr[i]
        val = arr[i + 1]
        if isinstance(stop, (int, float)) and input_val >= stop:
            result = val
        else:
            break
    
    return evaluate(result, ctx)


def _eval_match(arr: list, ctx: EvalContext) -> Any:
    if len(arr) < 4:
        return None
    
    input_val = evaluate(arr[1], ctx)
    pairs = arr[2:-1]
    default = arr[-1]
    
    for i in range(0, len(pairs) - 1, 2):
        label = pairs[i]
        output = pairs[i + 1]
        
        # Label can be single value or array
        if isinstance(label, list):
            if input_val in label:
                return evaluate(output, ctx)
        else:
            if input_val == label:
                return evaluate(output, ctx)
    
    return evaluate(default, ctx)


def _eval_case(arr: list, ctx: EvalContext) -> Any:
    if len(arr) < 3:
        return None
    
    pairs = arr[1:-1]
    default = arr[-1]
    
    for i in range(0, len(pairs) - 1, 2):
        condition = evaluate(pairs[i], ctx)
        if condition:
            return evaluate(pairs[i + 1], ctx)
    
    return evaluate(default, ctx)


def _eval_coalesce(arr: list, ctx: EvalContext) -> Any:
    for expr in arr[1:]:
        val = evaluate(expr, ctx)
        if val is not None:
            return val
    return None


# Comparison
def _eval_eq(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 3:
        return False
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    return a == b


def _eval_neq(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 3:
        return True
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    return a != b


def _eval_lt(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 3:
        return False
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a < b
    return False


def _eval_lte(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 3:
        return False
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a <= b
    return False


def _eval_gt(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 3:
        return False
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a > b
    return False


def _eval_gte(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 3:
        return False
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a >= b
    return False


# Logic
def _eval_all(arr: list, ctx: EvalContext) -> bool:
    return all(evaluate(expr, ctx) for expr in arr[1:])


def _eval_any(arr: list, ctx: EvalContext) -> bool:
    return any(evaluate(expr, ctx) for expr in arr[1:])


def _eval_not(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 2:
        return True
    return not evaluate(arr[1], ctx)


# Math
def _eval_add(arr: list, ctx: EvalContext) -> Optional[float]:
    total = 0.0
    for expr in arr[1:]:
        val = evaluate(expr, ctx)
        if isinstance(val, (int, float)):
            total += val
        else:
            return None
    return total


def _eval_sub(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) == 2:
        val = evaluate(arr[1], ctx)
        return -val if isinstance(val, (int, float)) else None
    if len(arr) < 3:
        return None
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    return None


def _eval_mul(arr: list, ctx: EvalContext) -> Optional[float]:
    product = 1.0
    for expr in arr[1:]:
        val = evaluate(expr, ctx)
        if isinstance(val, (int, float)):
            product *= val
        else:
            return None
    return product


def _eval_div(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 3:
        return None
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
        return a / b
    return None


def _eval_mod(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 3:
        return None
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
        return a % b
    return None


def _eval_pow(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 3:
        return None
    a = evaluate(arr[1], ctx)
    b = evaluate(arr[2], ctx)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a ** b
    return None


def _eval_abs(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return abs(val) if isinstance(val, (int, float)) else None


def _eval_ceil(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.ceil(val) if isinstance(val, (int, float)) else None


def _eval_floor(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.floor(val) if isinstance(val, (int, float)) else None


def _eval_round(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return round(val) if isinstance(val, (int, float)) else None


def _eval_min(arr: list, ctx: EvalContext) -> Optional[float]:
    vals = []
    for expr in arr[1:]:
        val = evaluate(expr, ctx)
        if isinstance(val, (int, float)):
            vals.append(val)
    return min(vals) if vals else None


def _eval_max(arr: list, ctx: EvalContext) -> Optional[float]:
    vals = []
    for expr in arr[1:]:
        val = evaluate(expr, ctx)
        if isinstance(val, (int, float)):
            vals.append(val)
    return max(vals) if vals else None


def _eval_ln(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.log(val) if isinstance(val, (int, float)) and val > 0 else None


def _eval_log10(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.log10(val) if isinstance(val, (int, float)) and val > 0 else None


def _eval_log2(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.log2(val) if isinstance(val, (int, float)) and val > 0 else None


def _eval_sin(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.sin(val) if isinstance(val, (int, float)) else None


def _eval_cos(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.cos(val) if isinstance(val, (int, float)) else None


def _eval_tan(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.tan(val) if isinstance(val, (int, float)) else None


def _eval_sqrt(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return math.sqrt(val) if isinstance(val, (int, float)) and val >= 0 else None


# String
def _eval_concat(arr: list, ctx: EvalContext) -> str:
    parts = []
    for expr in arr[1:]:
        val = evaluate(expr, ctx)
        parts.append(str(val) if val is not None else "")
    return "".join(parts)


def _eval_downcase(arr: list, ctx: EvalContext) -> Optional[str]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return val.lower() if isinstance(val, str) else None


def _eval_upcase(arr: list, ctx: EvalContext) -> Optional[str]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    return val.upper() if isinstance(val, str) else None


# Type conversion
def _eval_to_number(arr: list, ctx: EvalContext) -> Optional[float]:
    if len(arr) < 2:
        return None
    val = evaluate(arr[1], ctx)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    return None


def _eval_to_string(arr: list, ctx: EvalContext) -> str:
    if len(arr) < 2:
        return ""
    val = evaluate(arr[1], ctx)
    return str(val) if val is not None else ""


def _eval_to_boolean(arr: list, ctx: EvalContext) -> bool:
    if len(arr) < 2:
        return False
    val = evaluate(arr[1], ctx)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return len(val) > 0
    return val is not None


def _eval_typeof(arr: list, ctx: EvalContext) -> str:
    if len(arr) < 2:
        return "null"
    val = evaluate(arr[1], ctx)
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "number"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "array"
    if isinstance(val, dict):
        return "object"
    return "null"


# Color
def _eval_rgb(arr: list, ctx: EvalContext) -> Optional[tuple]:
    if len(arr) < 4:
        return None
    r = evaluate(arr[1], ctx)
    g = evaluate(arr[2], ctx)
    b = evaluate(arr[3], ctx)
    if all(isinstance(v, (int, float)) for v in [r, g, b]):
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)
    return None


def _eval_rgba(arr: list, ctx: EvalContext) -> Optional[tuple]:
    if len(arr) < 5:
        return None
    r = evaluate(arr[1], ctx)
    g = evaluate(arr[2], ctx)
    b = evaluate(arr[3], ctx)
    a = evaluate(arr[4], ctx)
    if all(isinstance(v, (int, float)) for v in [r, g, b, a]):
        return (r / 255.0, g / 255.0, b / 255.0, a)
    return None


def evaluate_color(expr: Any, ctx: EvalContext) -> Optional[tuple[float, float, float, float]]:
    """Evaluate expression and return as RGBA color tuple."""
    from forge3d.style import parse_color
    
    result = evaluate(expr, ctx)
    
    if isinstance(result, str):
        return parse_color(result)
    
    if isinstance(result, (list, tuple)) and len(result) >= 3:
        r = float(result[0])
        g = float(result[1])
        b = float(result[2])
        a = float(result[3]) if len(result) > 3 else 1.0
        return (r, g, b, a)
    
    return None


def evaluate_number(expr: Any, ctx: EvalContext) -> Optional[float]:
    """Evaluate expression and return as number."""
    result = evaluate(expr, ctx)
    if isinstance(result, (int, float)):
        return float(result)
    return None
