use super::*;

pub(super) fn eval_concat(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let mut result = String::new();
    for expr in &arr[1..] {
        let val = dispatch::evaluate_expression(expr, ctx)?;
        result.push_str(&value_to_string(&val));
    }
    Some(Value::String(result))
}

pub(super) fn eval_downcase(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let s = dispatch::evaluate_expression(arr.get(1)?, ctx)?
        .as_str()?
        .to_lowercase();
    Some(Value::String(s))
}

pub(super) fn eval_upcase(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let s = dispatch::evaluate_expression(arr.get(1)?, ctx)?
        .as_str()?
        .to_uppercase();
    Some(Value::String(s))
}

pub(super) fn eval_to_number(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    let num = match &val {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.parse().ok()?,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => return None,
    };
    Some(Value::Number(serde_json::Number::from_f64(num)?))
}

pub(super) fn eval_to_string(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    Some(Value::String(value_to_string(&val)))
}

pub(super) fn eval_to_boolean(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    let b = match &val {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|v| v != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        _ => true,
    };
    Some(Value::Bool(b))
}

pub(super) fn eval_to_color(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
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

pub(super) fn eval_typeof(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
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

pub(super) fn eval_rgb(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    rgba_numbers(arr, ctx, 1.0)
}

pub(super) fn eval_rgba(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let alpha = dispatch::evaluate_expression(arr.get(4)?, ctx)?.as_f64()?;
    rgba_numbers(arr, ctx, alpha)
}

fn rgba_numbers(arr: &[Value], ctx: &EvalContext, alpha: f64) -> Option<Value> {
    let r = dispatch::evaluate_expression(arr.get(1)?, ctx)?.as_f64()? / 255.0;
    let g = dispatch::evaluate_expression(arr.get(2)?, ctx)?.as_f64()? / 255.0;
    let b = dispatch::evaluate_expression(arr.get(3)?, ctx)?.as_f64()? / 255.0;
    Some(Value::Array(vec![
        Value::Number(serde_json::Number::from_f64(r)?),
        Value::Number(serde_json::Number::from_f64(g)?),
        Value::Number(serde_json::Number::from_f64(b)?),
        Value::Number(serde_json::Number::from_f64(alpha)?),
    ]))
}

fn value_to_string(val: &Value) -> String {
    match val {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => val.to_string(),
    }
}

pub(super) fn parse_color_to_array(s: &str) -> Option<[f32; 4]> {
    crate::style::types::parse_color_string(s)
}
