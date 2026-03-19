use super::*;

pub(super) fn eval_interpolate(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.len() < 5 {
        return None;
    }

    let interp_type = arr.get(1)?;
    let input_expr = arr.get(2)?;
    let input = dispatch::evaluate_expression(input_expr, ctx)?.as_f64()?;

    let (is_exponential, base) = if let Some(interp_arr) = interp_type.as_array() {
        match interp_arr.first()?.as_str()? {
            "linear" => (false, 1.0),
            "exponential" => (true, interp_arr.get(1)?.as_f64().unwrap_or(1.0)),
            "cubic-bezier" => (false, 1.0),
            _ => (false, 1.0),
        }
    } else {
        (false, 1.0)
    };

    let stops: Vec<(f64, Value)> = arr[3..]
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some((chunk[0].as_f64()?, chunk[1].clone()))
            } else {
                None
            }
        })
        .collect();

    if stops.is_empty() {
        return None;
    }

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
                let range = stop_high - stop_low;
                if range == 0.0 {
                    0.0
                } else {
                    (base.powf(input - stop_low) - 1.0) / (base.powf(range) - 1.0)
                }
            } else {
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
            Some(Value::Number(serde_json::Number::from_f64(
                va + (vb - va) * t,
            )?))
        }
        (Value::Array(aa), Value::Array(ab)) if aa.len() == ab.len() => {
            let result: Option<Vec<Value>> = aa
                .iter()
                .zip(ab.iter())
                .map(|(ea, eb)| interpolate_values(ea, eb, t))
                .collect();
            result.map(Value::Array)
        }
        (Value::String(sa), Value::String(sb)) => {
            if let (Some(ca), Some(cb)) = (
                strings::parse_color_to_array(sa),
                strings::parse_color_to_array(sb),
            ) {
                let result: Vec<Value> = ca
                    .iter()
                    .zip(cb.iter())
                    .map(|(a, b)| {
                        let v = a + (b - a) * t as f32;
                        Value::Number(serde_json::Number::from_f64(v as f64).unwrap())
                    })
                    .collect();
                Some(Value::Array(result))
            } else if t < 0.5 {
                Some(a.clone())
            } else {
                Some(b.clone())
            }
        }
        _ => {
            if t < 0.5 {
                Some(a.clone())
            } else {
                Some(b.clone())
            }
        }
    }
}

pub(super) fn eval_step(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.len() < 4 {
        return None;
    }

    let input = dispatch::evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let default = arr.get(2)?;
    let stops: Vec<(f64, &Value)> = arr[3..]
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some((chunk[0].as_f64()?, &chunk[1]))
            } else {
                None
            }
        })
        .collect();

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

pub(super) fn eval_match(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.len() < 4 {
        return None;
    }

    let input = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    let pairs = &arr[2..arr.len() - 1];
    let default = arr.last()?;

    for chunk in pairs.chunks(2) {
        if chunk.len() != 2 {
            continue;
        }

        let label = &chunk[0];
        let output = &chunk[1];
        let matches = if let Some(labels) = label.as_array() {
            labels.iter().any(|l| comparison::values_equal(&input, l))
        } else {
            comparison::values_equal(&input, label)
        };

        if matches {
            return dispatch::evaluate_expression(output, ctx);
        }
    }

    dispatch::evaluate_expression(default, ctx)
}

pub(super) fn eval_case(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.len() < 3 {
        return None;
    }

    let pairs = &arr[1..arr.len() - 1];
    let default = arr.last()?;

    for chunk in pairs.chunks(2) {
        if chunk.len() != 2 {
            continue;
        }

        let condition = dispatch::evaluate_expression(&chunk[0], ctx)?;
        if condition.as_bool().unwrap_or(false) {
            return dispatch::evaluate_expression(&chunk[1], ctx);
        }
    }

    dispatch::evaluate_expression(default, ctx)
}

pub(super) fn eval_coalesce(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    for expr in &arr[1..] {
        if let Some(val) = dispatch::evaluate_expression(expr, ctx) {
            if !val.is_null() {
                return Some(val);
            }
        }
    }
    None
}
