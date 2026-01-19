//! Mapbox Style Spec expression evaluation.
//!
//! Implements evaluation of data-driven expressions including:
//! - `interpolate`: Linear/exponential interpolation between stops
//! - `step`: Stepped/discrete values at breakpoints
//! - `match`: Pattern matching on property values
//! - `get`: Property value lookup
//! - `coalesce`: First non-null value
//! - Math operators: `+`, `-`, `*`, `/`, `%`, `^`
//! - Comparison: `<`, `<=`, `>`, `>=`
//! - Logic: `all`, `any`, `!`, `case`

use serde_json::Value;

/// Expression evaluation context containing feature properties and zoom level.
#[derive(Debug, Clone)]
pub struct EvalContext<'a> {
    /// Feature properties map.
    pub properties: &'a serde_json::Map<String, Value>,
    /// Current zoom level.
    pub zoom: f64,
    /// Geometry type (optional).
    pub geometry_type: Option<&'a str>,
}

impl<'a> EvalContext<'a> {
    pub fn new(properties: &'a serde_json::Map<String, Value>, zoom: f64) -> Self {
        Self {
            properties,
            zoom,
            geometry_type: None,
        }
    }

    pub fn with_geometry_type(mut self, geom_type: &'a str) -> Self {
        self.geometry_type = Some(geom_type);
        self
    }
}

/// Evaluate an expression and return a typed result.
pub fn evaluate_expression(expr: &Value, ctx: &EvalContext) -> Option<Value> {
    match expr {
        Value::Null => Some(Value::Null),
        Value::Bool(b) => Some(Value::Bool(*b)),
        Value::Number(n) => Some(Value::Number(n.clone())),
        Value::String(s) => Some(Value::String(s.clone())),
        Value::Array(arr) => evaluate_array_expression(arr, ctx),
        Value::Object(_) => Some(expr.clone()), // Objects pass through
    }
}

/// Evaluate an array-based expression.
fn evaluate_array_expression(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.is_empty() {
        return None;
    }

    let op = arr[0].as_str()?;

    match op {
        // Property access
        "get" => eval_get(arr, ctx),
        "has" => eval_has(arr, ctx),
        "at" => eval_at(arr, ctx),
        "length" => eval_length(arr, ctx),

        // Interpolation
        "interpolate" | "interpolate-hcl" | "interpolate-lab" => eval_interpolate(arr, ctx),
        "step" => eval_step(arr, ctx),

        // Pattern matching
        "match" => eval_match(arr, ctx),
        "case" => eval_case(arr, ctx),
        "coalesce" => eval_coalesce(arr, ctx),

        // Comparison
        "==" => eval_eq(arr, ctx),
        "!=" => eval_neq(arr, ctx),
        "<" => eval_lt(arr, ctx),
        "<=" => eval_lte(arr, ctx),
        ">" => eval_gt(arr, ctx),
        ">=" => eval_gte(arr, ctx),

        // Logic
        "all" => eval_all(arr, ctx),
        "any" => eval_any(arr, ctx),
        "!" => eval_not(arr, ctx),

        // Math
        "+" => eval_add(arr, ctx),
        "-" => eval_sub(arr, ctx),
        "*" => eval_mul(arr, ctx),
        "/" => eval_div(arr, ctx),
        "%" => eval_mod(arr, ctx),
        "^" => eval_pow(arr, ctx),
        "abs" => eval_abs(arr, ctx),
        "ceil" => eval_ceil(arr, ctx),
        "floor" => eval_floor(arr, ctx),
        "round" => eval_round(arr, ctx),
        "min" => eval_min(arr, ctx),
        "max" => eval_max(arr, ctx),
        "ln" => eval_ln(arr, ctx),
        "log10" => eval_log10(arr, ctx),
        "log2" => eval_log2(arr, ctx),
        "sin" => eval_sin(arr, ctx),
        "cos" => eval_cos(arr, ctx),
        "tan" => eval_tan(arr, ctx),
        "sqrt" => eval_sqrt(arr, ctx),

        // String
        "concat" => eval_concat(arr, ctx),
        "downcase" => eval_downcase(arr, ctx),
        "upcase" => eval_upcase(arr, ctx),

        // Type
        "to-number" => eval_to_number(arr, ctx),
        "to-string" => eval_to_string(arr, ctx),
        "to-boolean" => eval_to_boolean(arr, ctx),
        "to-color" => eval_to_color(arr, ctx),
        "typeof" => eval_typeof(arr, ctx),

        // Color
        "rgb" => eval_rgb(arr, ctx),
        "rgba" => eval_rgba(arr, ctx),

        // Special
        "zoom" => Some(Value::Number(serde_json::Number::from_f64(ctx.zoom)?)),
        "geometry-type" => ctx.geometry_type.map(|s| Value::String(s.to_string())),
        "literal" => arr.get(1).cloned(),

        _ => None, // Unknown operator
    }
}

// Property access
fn eval_get(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let key = arr.get(1)?.as_str()?;
    ctx.properties.get(key).cloned()
}

fn eval_has(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let key = arr.get(1)?.as_str()?;
    Some(Value::Bool(ctx.properties.contains_key(key)))
}

fn eval_at(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let index = evaluate_expression(arr.get(1)?, ctx)?.as_u64()? as usize;
    let array = evaluate_expression(arr.get(2)?, ctx)?;
    array.as_array()?.get(index).cloned()
}

fn eval_length(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = evaluate_expression(arr.get(1)?, ctx)?;
    let len = match &val {
        Value::String(s) => s.len(),
        Value::Array(a) => a.len(),
        _ => return None,
    };
    Some(Value::Number(serde_json::Number::from(len as u64)))
}

// Interpolation
fn eval_interpolate(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    // ["interpolate", ["linear"] | ["exponential", base], ["zoom"], stop1, val1, stop2, val2, ...]
    if arr.len() < 5 {
        return None;
    }

    let interp_type = arr.get(1)?;
    let input_expr = arr.get(2)?;
    let input = evaluate_expression(input_expr, ctx)?.as_f64()?;

    // Parse interpolation type
    let (is_exponential, base) = if let Some(interp_arr) = interp_type.as_array() {
        match interp_arr.first()?.as_str()? {
            "linear" => (false, 1.0),
            "exponential" => {
                let b = interp_arr.get(1)?.as_f64().unwrap_or(1.0);
                (true, b)
            }
            "cubic-bezier" => (false, 1.0), // Fallback to linear
            _ => (false, 1.0),
        }
    } else {
        (false, 1.0)
    };

    // Parse stops (pairs of input, output)
    let stops: Vec<(f64, Value)> = arr[3..]
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                let stop = chunk[0].as_f64()?;
                Some((stop, chunk[1].clone()))
            } else {
                None
            }
        })
        .collect();

    if stops.is_empty() {
        return None;
    }

    // Find surrounding stops
    if input <= stops[0].0 {
        return Some(stops[0].1.clone());
    }
    if input >= stops.last()?.0 {
        return Some(stops.last()?.1.clone());
    }

    for i in 0..stops.len() - 1 {
        let (stop_low, val_low) = &stops[i];
        let (stop_high, val_high) = &stops[i + 1];

        if input >= *stop_low && input <= *stop_high {
            let t = if is_exponential && base != 1.0 {
                // Exponential interpolation
                let range = stop_high - stop_low;
                if range == 0.0 {
                    0.0
                } else {
                    (base.powf(input - stop_low) - 1.0) / (base.powf(range) - 1.0)
                }
            } else {
                // Linear interpolation
                (input - stop_low) / (stop_high - stop_low)
            };

            return interpolate_values(val_low, val_high, t);
        }
    }

    None
}

fn interpolate_values(a: &Value, b: &Value, t: f64) -> Option<Value> {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            let va = na.as_f64()?;
            let vb = nb.as_f64()?;
            let result = va + (vb - va) * t;
            Some(Value::Number(serde_json::Number::from_f64(result)?))
        }
        (Value::Array(aa), Value::Array(ab)) if aa.len() == ab.len() => {
            // Interpolate arrays element-wise (e.g., colors)
            let result: Option<Vec<Value>> = aa
                .iter()
                .zip(ab.iter())
                .map(|(ea, eb)| interpolate_values(ea, eb, t))
                .collect();
            result.map(Value::Array)
        }
        (Value::String(sa), Value::String(sb)) => {
            // Try to parse as colors and interpolate
            if let (Some(ca), Some(cb)) = (parse_color_to_array(sa), parse_color_to_array(sb)) {
                let result: Vec<Value> = ca
                    .iter()
                    .zip(cb.iter())
                    .map(|(a, b)| {
                        let v = a + (b - a) * t as f32;
                        Value::Number(serde_json::Number::from_f64(v as f64).unwrap())
                    })
                    .collect();
                Some(Value::Array(result))
            } else {
                // Can't interpolate strings
                if t < 0.5 { Some(a.clone()) } else { Some(b.clone()) }
            }
        }
        _ => {
            // Can't interpolate, return closest
            if t < 0.5 { Some(a.clone()) } else { Some(b.clone()) }
        }
    }
}

fn eval_step(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    // ["step", input, default, stop1, val1, stop2, val2, ...]
    if arr.len() < 4 {
        return None;
    }

    let input = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let default = arr.get(2)?;

    // Parse stops
    let stops: Vec<(f64, &Value)> = arr[3..]
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                let stop = chunk[0].as_f64()?;
                Some((stop, &chunk[1]))
            } else {
                None
            }
        })
        .collect();

    // Find the applicable stop (largest stop <= input)
    let mut result = default;
    for (stop, val) in &stops {
        if input >= *stop {
            result = *val;
        } else {
            break;
        }
    }

    Some(result.clone())
}

fn eval_match(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    // ["match", input, label1, output1, label2, output2, ..., default]
    if arr.len() < 4 {
        return None;
    }

    let input = evaluate_expression(arr.get(1)?, ctx)?;
    let pairs = &arr[2..arr.len() - 1];
    let default = arr.last()?;

    for chunk in pairs.chunks(2) {
        if chunk.len() != 2 {
            continue;
        }

        let label = &chunk[0];
        let output = &chunk[1];

        // Label can be a single value or array of values
        let matches = if let Some(labels) = label.as_array() {
            labels.iter().any(|l| values_equal(&input, l))
        } else {
            values_equal(&input, label)
        };

        if matches {
            return evaluate_expression(output, ctx);
        }
    }

    evaluate_expression(default, ctx)
}

fn eval_case(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    // ["case", cond1, output1, cond2, output2, ..., default]
    if arr.len() < 3 {
        return None;
    }

    let pairs = &arr[1..arr.len() - 1];
    let default = arr.last()?;

    for chunk in pairs.chunks(2) {
        if chunk.len() != 2 {
            continue;
        }

        let condition = evaluate_expression(&chunk[0], ctx)?;
        if condition.as_bool().unwrap_or(false) {
            return evaluate_expression(&chunk[1], ctx);
        }
    }

    evaluate_expression(default, ctx)
}

fn eval_coalesce(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    for expr in &arr[1..] {
        if let Some(val) = evaluate_expression(expr, ctx) {
            if !val.is_null() {
                return Some(val);
            }
        }
    }
    None
}

// Comparison operators
fn eval_eq(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?;
    let b = evaluate_expression(arr.get(2)?, ctx)?;
    Some(Value::Bool(values_equal(&a, &b)))
}

fn eval_neq(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?;
    let b = evaluate_expression(arr.get(2)?, ctx)?;
    Some(Value::Bool(!values_equal(&a, &b)))
}

fn eval_lt(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Bool(a < b))
}

fn eval_lte(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Bool(a <= b))
}

fn eval_gt(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Bool(a > b))
}

fn eval_gte(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Bool(a >= b))
}

// Logic operators
fn eval_all(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    for expr in &arr[1..] {
        let val = evaluate_expression(expr, ctx)?;
        if !val.as_bool().unwrap_or(false) {
            return Some(Value::Bool(false));
        }
    }
    Some(Value::Bool(true))
}

fn eval_any(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    for expr in &arr[1..] {
        let val = evaluate_expression(expr, ctx)?;
        if val.as_bool().unwrap_or(false) {
            return Some(Value::Bool(true));
        }
    }
    Some(Value::Bool(false))
}

fn eval_not(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = evaluate_expression(arr.get(1)?, ctx)?;
    Some(Value::Bool(!val.as_bool().unwrap_or(false)))
}

// Math operators
fn eval_add(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let mut sum = 0.0;
    for expr in &arr[1..] {
        sum += evaluate_expression(expr, ctx)?.as_f64()?;
    }
    Some(Value::Number(serde_json::Number::from_f64(sum)?))
}

fn eval_sub(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.len() == 2 {
        // Unary negation
        let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
        return Some(Value::Number(serde_json::Number::from_f64(-a)?));
    }
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a - b)?))
}

fn eval_mul(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let mut product = 1.0;
    for expr in &arr[1..] {
        product *= evaluate_expression(expr, ctx)?.as_f64()?;
    }
    Some(Value::Number(serde_json::Number::from_f64(product)?))
}

fn eval_div(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    if b == 0.0 {
        return None;
    }
    Some(Value::Number(serde_json::Number::from_f64(a / b)?))
}

fn eval_mod(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    if b == 0.0 {
        return None;
    }
    Some(Value::Number(serde_json::Number::from_f64(a % b)?))
}

fn eval_pow(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.powf(b))?))
}

fn eval_abs(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.abs())?))
}

fn eval_ceil(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.ceil())?))
}

fn eval_floor(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.floor())?))
}

fn eval_round(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.round())?))
}

fn eval_min(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let mut min = f64::MAX;
    for expr in &arr[1..] {
        let v = evaluate_expression(expr, ctx)?.as_f64()?;
        if v < min {
            min = v;
        }
    }
    Some(Value::Number(serde_json::Number::from_f64(min)?))
}

fn eval_max(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let mut max = f64::MIN;
    for expr in &arr[1..] {
        let v = evaluate_expression(expr, ctx)?.as_f64()?;
        if v > max {
            max = v;
        }
    }
    Some(Value::Number(serde_json::Number::from_f64(max)?))
}

fn eval_ln(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.ln())?))
}

fn eval_log10(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.log10())?))
}

fn eval_log2(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.log2())?))
}

fn eval_sin(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.sin())?))
}

fn eval_cos(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.cos())?))
}

fn eval_tan(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.tan())?))
}

fn eval_sqrt(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(a.sqrt())?))
}

// String operators
fn eval_concat(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let mut result = String::new();
    for expr in &arr[1..] {
        let val = evaluate_expression(expr, ctx)?;
        result.push_str(&value_to_string(&val));
    }
    Some(Value::String(result))
}

fn eval_downcase(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let s = evaluate_expression(arr.get(1)?, ctx)?.as_str()?.to_lowercase();
    Some(Value::String(s))
}

fn eval_upcase(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let s = evaluate_expression(arr.get(1)?, ctx)?.as_str()?.to_uppercase();
    Some(Value::String(s))
}

// Type conversion
fn eval_to_number(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = evaluate_expression(arr.get(1)?, ctx)?;
    let num = match &val {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.parse().ok()?,
        Value::Bool(b) => if *b { 1.0 } else { 0.0 },
        _ => return None,
    };
    Some(Value::Number(serde_json::Number::from_f64(num)?))
}

fn eval_to_string(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = evaluate_expression(arr.get(1)?, ctx)?;
    Some(Value::String(value_to_string(&val)))
}

fn eval_to_boolean(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = evaluate_expression(arr.get(1)?, ctx)?;
    let b = match &val {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|v| v != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        _ => true,
    };
    Some(Value::Bool(b))
}

fn eval_to_color(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = evaluate_expression(arr.get(1)?, ctx)?;
    if let Some(s) = val.as_str() {
        if let Some(rgba) = parse_color_to_array(s) {
            return Some(Value::Array(
                rgba.iter()
                    .map(|v| Value::Number(serde_json::Number::from_f64(*v as f64).unwrap()))
                    .collect(),
            ));
        }
    }
    Some(val)
}

fn eval_typeof(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = evaluate_expression(arr.get(1)?, ctx)?;
    let type_name = match &val {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    Some(Value::String(type_name.to_string()))
}

// Color constructors
fn eval_rgb(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let r = evaluate_expression(arr.get(1)?, ctx)?.as_f64()? / 255.0;
    let g = evaluate_expression(arr.get(2)?, ctx)?.as_f64()? / 255.0;
    let b = evaluate_expression(arr.get(3)?, ctx)?.as_f64()? / 255.0;
    Some(Value::Array(vec![
        Value::Number(serde_json::Number::from_f64(r)?),
        Value::Number(serde_json::Number::from_f64(g)?),
        Value::Number(serde_json::Number::from_f64(b)?),
        Value::Number(serde_json::Number::from_f64(1.0)?),
    ]))
}

fn eval_rgba(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let r = evaluate_expression(arr.get(1)?, ctx)?.as_f64()? / 255.0;
    let g = evaluate_expression(arr.get(2)?, ctx)?.as_f64()? / 255.0;
    let b = evaluate_expression(arr.get(3)?, ctx)?.as_f64()? / 255.0;
    let a = evaluate_expression(arr.get(4)?, ctx)?.as_f64()?;
    Some(Value::Array(vec![
        Value::Number(serde_json::Number::from_f64(r)?),
        Value::Number(serde_json::Number::from_f64(g)?),
        Value::Number(serde_json::Number::from_f64(b)?),
        Value::Number(serde_json::Number::from_f64(a)?),
    ]))
}

// Helper functions
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y))
        }
        _ => false,
    }
}

fn value_to_string(val: &Value) -> String {
    match val {
        Value::Null => "".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => val.to_string(),
    }
}

fn parse_color_to_array(s: &str) -> Option<[f32; 4]> {
    crate::style::types::parse_color_string(s)
}

/// Evaluate a color expression to RGBA.
pub fn evaluate_color(expr: &Value, ctx: &EvalContext) -> Option<[f32; 4]> {
    let result = evaluate_expression(expr, ctx)?;
    
    match &result {
        Value::String(s) => parse_color_to_array(s),
        Value::Array(arr) if arr.len() >= 3 => {
            let r = arr[0].as_f64()? as f32;
            let g = arr[1].as_f64()? as f32;
            let b = arr[2].as_f64()? as f32;
            let a = arr.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            Some([r, g, b, a])
        }
        _ => None,
    }
}

/// Evaluate a number expression.
pub fn evaluate_number(expr: &Value, ctx: &EvalContext) -> Option<f64> {
    evaluate_expression(expr, ctx)?.as_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_props() -> serde_json::Map<String, Value> {
        serde_json::Map::new()
    }

    fn props_with(key: &str, val: Value) -> serde_json::Map<String, Value> {
        let mut m = serde_json::Map::new();
        m.insert(key.to_string(), val);
        m
    }

    #[test]
    fn test_get_property() {
        let props = props_with("name", Value::String("Test".to_string()));
        let ctx = EvalContext::new(&props, 10.0);
        
        let expr = serde_json::json!(["get", "name"]);
        let result = evaluate_expression(&expr, &ctx);
        assert_eq!(result, Some(Value::String("Test".to_string())));
    }

    #[test]
    fn test_zoom() {
        let props = empty_props();
        let ctx = EvalContext::new(&props, 12.5);
        
        let expr = serde_json::json!(["zoom"]);
        let result = evaluate_expression(&expr, &ctx);
        assert_eq!(result.and_then(|v| v.as_f64()), Some(12.5));
    }

    #[test]
    fn test_interpolate_linear() {
        let props = empty_props();
        let ctx = EvalContext::new(&props, 10.0);
        
        let expr = serde_json::json!([
            "interpolate", ["linear"], ["zoom"],
            5, 1,
            15, 10
        ]);
        let result = evaluate_expression(&expr, &ctx);
        // At zoom 10, halfway between 5 and 15, should be ~5.5
        assert!((result.and_then(|v| v.as_f64()).unwrap() - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_step() {
        let props = empty_props();
        
        let expr = serde_json::json!([
            "step", ["zoom"],
            "small",
            10, "medium",
            15, "large"
        ]);
        
        let ctx = EvalContext::new(&props, 5.0);
        assert_eq!(
            evaluate_expression(&expr, &ctx),
            Some(Value::String("small".to_string()))
        );
        
        let ctx = EvalContext::new(&props, 12.0);
        assert_eq!(
            evaluate_expression(&expr, &ctx),
            Some(Value::String("medium".to_string()))
        );
        
        let ctx = EvalContext::new(&props, 20.0);
        assert_eq!(
            evaluate_expression(&expr, &ctx),
            Some(Value::String("large".to_string()))
        );
    }

    #[test]
    fn test_match() {
        let props = props_with("type", Value::String("highway".to_string()));
        let ctx = EvalContext::new(&props, 10.0);
        
        let expr = serde_json::json!([
            "match", ["get", "type"],
            "highway", "#ff0000",
            "street", "#00ff00",
            "#888888"
        ]);
        let result = evaluate_expression(&expr, &ctx);
        assert_eq!(result, Some(Value::String("#ff0000".to_string())));
    }

    #[test]
    fn test_case() {
        let props = props_with("population", Value::Number(50000.into()));
        let ctx = EvalContext::new(&props, 10.0);
        
        let expr = serde_json::json!([
            "case",
            [">", ["get", "population"], 100000], "large",
            [">", ["get", "population"], 10000], "medium",
            "small"
        ]);
        let result = evaluate_expression(&expr, &ctx);
        assert_eq!(result, Some(Value::String("medium".to_string())));
    }

    #[test]
    fn test_math_operators() {
        let props = empty_props();
        let ctx = EvalContext::new(&props, 10.0);
        
        let expr = serde_json::json!(["+", 1, 2, 3]);
        assert_eq!(evaluate_expression(&expr, &ctx).and_then(|v| v.as_f64()), Some(6.0));
        
        let expr = serde_json::json!(["*", 2, 3]);
        assert_eq!(evaluate_expression(&expr, &ctx).and_then(|v| v.as_f64()), Some(6.0));
        
        let expr = serde_json::json!(["/", 10, 2]);
        assert_eq!(evaluate_expression(&expr, &ctx).and_then(|v| v.as_f64()), Some(5.0));
    }

    #[test]
    fn test_comparison() {
        let props = empty_props();
        let ctx = EvalContext::new(&props, 10.0);
        
        let expr = serde_json::json!([">", 5, 3]);
        assert_eq!(evaluate_expression(&expr, &ctx), Some(Value::Bool(true)));
        
        let expr = serde_json::json!(["<=", 5, 5]);
        assert_eq!(evaluate_expression(&expr, &ctx), Some(Value::Bool(true)));
    }

    #[test]
    fn test_coalesce() {
        let props = props_with("alt_name", Value::String("Alternative".to_string()));
        let ctx = EvalContext::new(&props, 10.0);
        
        let expr = serde_json::json!(["coalesce", ["get", "name"], ["get", "alt_name"], "Unknown"]);
        let result = evaluate_expression(&expr, &ctx);
        assert_eq!(result, Some(Value::String("Alternative".to_string())));
    }
}
