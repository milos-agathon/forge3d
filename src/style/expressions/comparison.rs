use super::*;

pub(super) fn eval_eq(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    let b = dispatch::evaluate_expression(arr.get(2)?, ctx)?;
    Some(Value::Bool(values_equal(&a, &b)))
}

pub(super) fn eval_neq(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    let a = dispatch::evaluate_expression(arr.get(1)?, ctx)?;
    let b = dispatch::evaluate_expression(arr.get(2)?, ctx)?;
    Some(Value::Bool(!values_equal(&a, &b)))
}

pub(super) fn eval_lt(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    compare_numbers(arr, ctx, |a, b| a < b)
}

pub(super) fn eval_lte(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    compare_numbers(arr, ctx, |a, b| a <= b)
}

pub(super) fn eval_gt(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    compare_numbers(arr, ctx, |a, b| a > b)
}

pub(super) fn eval_gte(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    compare_numbers(arr, ctx, |a, b| a >= b)
}

fn compare_numbers(
    arr: &[Value],
    ctx: &EvalContext,
    cmp: impl FnOnce(f64, f64) -> bool,
) -> Option<Value> {
    let a = dispatch::evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = dispatch::evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Bool(cmp(a, b)))
}

pub(super) fn values_equal(a: &Value, b: &Value) -> bool {
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
