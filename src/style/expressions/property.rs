use super::*;

pub(super) fn eval_get(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let key = arr.get(1)?.as_str()?;
    ctx.properties.get(key).cloned()
}

pub(super) fn eval_has(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let key = arr.get(1)?.as_str()?;
    Some(Value::Bool(ctx.properties.contains_key(key)))
}

pub(super) fn eval_at(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let index = dispatch::evaluate_expression(arr.get(1)?, ctx)?.as_u64()? as usize;
    let array = dispatch::evaluate_expression(arr.get(2)?, ctx)?;
    array.as_array()?.get(index).cloned()
}

pub(super) fn eval_length(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    let len = match &val {
        Value::String(s) => s.len(),
        Value::Array(a) => a.len(),
        _ => return None,
    };
    Some(Value::Number(serde_json::Number::from(len as u64)))
}
