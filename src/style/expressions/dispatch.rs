use super::*;

pub fn evaluate_expression(expr: &Value, ctx: &EvalContext) -> Option<Value> {
    match expr {
        Value::Null => Some(Value::Null),
        Value::Bool(b) => Some(Value::Bool(*b)),
        Value::Number(n) => Some(Value::Number(n.clone())),
        Value::String(s) => Some(Value::String(s.clone())),
        Value::Array(arr) => evaluate_array_expression(arr, ctx),
        Value::Object(_) => Some(expr.clone()),
    }
}

fn evaluate_array_expression(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.is_empty() {
        return None;
    }

    let op = arr[0].as_str()?;

    match op {
        "get" => property::eval_get(arr, ctx),
        "has" => property::eval_has(arr, ctx),
        "at" => property::eval_at(arr, ctx),
        "length" => property::eval_length(arr, ctx),
        "interpolate" | "interpolate-hcl" | "interpolate-lab" => {
            control::eval_interpolate(arr, ctx)
        }
        "step" => control::eval_step(arr, ctx),
        "match" => control::eval_match(arr, ctx),
        "case" => control::eval_case(arr, ctx),
        "coalesce" => control::eval_coalesce(arr, ctx),
        "==" => comparison::eval_eq(arr, ctx),
        "!=" => comparison::eval_neq(arr, ctx),
        "<" => comparison::eval_lt(arr, ctx),
        "<=" => comparison::eval_lte(arr, ctx),
        ">" => comparison::eval_gt(arr, ctx),
        ">=" => comparison::eval_gte(arr, ctx),
        "all" => logic::eval_all(arr, ctx),
        "any" => logic::eval_any(arr, ctx),
        "!" => logic::eval_not(arr, ctx),
        "+" => math::eval_add(arr, ctx),
        "-" => math::eval_sub(arr, ctx),
        "*" => math::eval_mul(arr, ctx),
        "/" => math::eval_div(arr, ctx),
        "%" => math::eval_mod(arr, ctx),
        "^" => math::eval_pow(arr, ctx),
        "abs" => math::eval_abs(arr, ctx),
        "ceil" => math::eval_ceil(arr, ctx),
        "floor" => math::eval_floor(arr, ctx),
        "round" => math::eval_round(arr, ctx),
        "min" => math::eval_min(arr, ctx),
        "max" => math::eval_max(arr, ctx),
        "ln" => math::eval_ln(arr, ctx),
        "log10" => math::eval_log10(arr, ctx),
        "log2" => math::eval_log2(arr, ctx),
        "sin" => math::eval_sin(arr, ctx),
        "cos" => math::eval_cos(arr, ctx),
        "tan" => math::eval_tan(arr, ctx),
        "sqrt" => math::eval_sqrt(arr, ctx),
        "concat" => strings::eval_concat(arr, ctx),
        "downcase" => strings::eval_downcase(arr, ctx),
        "upcase" => strings::eval_upcase(arr, ctx),
        "to-number" => strings::eval_to_number(arr, ctx),
        "to-string" => strings::eval_to_string(arr, ctx),
        "to-boolean" => strings::eval_to_boolean(arr, ctx),
        "to-color" => strings::eval_to_color(arr, ctx),
        "typeof" => strings::eval_typeof(arr, ctx),
        "rgb" => strings::eval_rgb(arr, ctx),
        "rgba" => strings::eval_rgba(arr, ctx),
        "zoom" => Some(Value::Number(serde_json::Number::from_f64(ctx.zoom)?)),
        "geometry-type" => ctx.geometry_type.map(|s| Value::String(s.to_string())),
        "literal" => arr.get(1).cloned(),
        _ => None,
    }
}

pub fn evaluate_color(expr: &Value, ctx: &EvalContext) -> Option<[f32; 4]> {
    let result = evaluate_expression(expr, ctx)?;

    match &result {
        Value::String(s) => strings::parse_color_to_array(s),
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

pub fn evaluate_number(expr: &Value, ctx: &EvalContext) -> Option<f64> {
    evaluate_expression(expr, ctx)?.as_f64()
}
