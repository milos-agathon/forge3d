use super::*;

pub(super) fn eval_all(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    for expr in &arr[1..] {
        let val = dispatch::evaluate_expression(expr, ctx)?;
        if !val.as_bool().unwrap_or(false) {
            return Some(Value::Bool(false));
        }
    }
    Some(Value::Bool(true))
}

pub(super) fn eval_any(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    for expr in &arr[1..] {
        let val = dispatch::evaluate_expression(expr, ctx)?;
        if val.as_bool().unwrap_or(false) {
            return Some(Value::Bool(true));
        }
    }
    Some(Value::Bool(false))
}

pub(super) fn eval_not(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let val = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    Some(Value::Bool(!val.as_bool().unwrap_or(false)))
}
