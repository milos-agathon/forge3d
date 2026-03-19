use super::*;

pub(super) fn eval_add(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    reduce_numbers(arr, ctx, 0.0, |acc, value| acc + value)
}

pub(super) fn eval_sub(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    if arr.len() == 2 {
        let a = dispatch::evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
        return Some(Value::Number(serde_json::Number::from_f64(-a)?));
    }
    binary_number(arr, ctx, |a, b| Some(a - b))
}

pub(super) fn eval_mul(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    reduce_numbers(arr, ctx, 1.0, |acc, value| acc * value)
}

pub(super) fn eval_div(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    binary_number(arr, ctx, |a, b| if b == 0.0 { None } else { Some(a / b) })
}

pub(super) fn eval_mod(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    binary_number(arr, ctx, |a, b| if b == 0.0 { None } else { Some(a % b) })
}

pub(super) fn eval_pow(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    binary_number(arr, ctx, |a, b| Some(a.powf(b)))
}

pub(super) fn eval_abs(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.abs())
}

pub(super) fn eval_ceil(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.ceil())
}

pub(super) fn eval_floor(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.floor())
}

pub(super) fn eval_round(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.round())
}

pub(super) fn eval_min(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    reduce_numbers(arr, ctx, f64::MAX, |acc, value| acc.min(value))
}

pub(super) fn eval_max(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    reduce_numbers(arr, ctx, f64::MIN, |acc, value| acc.max(value))
}

pub(super) fn eval_ln(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.ln())
}

pub(super) fn eval_log10(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.log10())
}

pub(super) fn eval_log2(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.log2())
}

pub(super) fn eval_sin(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.sin())
}

pub(super) fn eval_cos(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.cos())
}

pub(super) fn eval_tan(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.tan())
}

pub(super) fn eval_sqrt(arr: &[Value], ctx: &EvalContext) -> Option<Value> {
    unary_number(arr, ctx, |a| a.sqrt())
}

fn unary_number(arr: &[Value], ctx: &EvalContext, op: impl FnOnce(f64) -> f64) -> Option<Value> {
    let a = dispatch::evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(op(a))?))
}

fn binary_number(
    arr: &[Value],
    ctx: &EvalContext,
    op: impl FnOnce(f64, f64) -> Option<f64>,
) -> Option<Value> {
    let a = dispatch::evaluate_expression(arr.get(1)?, ctx)?.as_f64()?;
    let b = dispatch::evaluate_expression(arr.get(2)?, ctx)?.as_f64()?;
    Some(Value::Number(serde_json::Number::from_f64(op(a, b)?)?))
}

fn reduce_numbers(
    arr: &[Value],
    ctx: &EvalContext,
    init: f64,
    op: impl Fn(f64, f64) -> f64,
) -> Option<Value> {
    let mut acc = init;
    for expr in &arr[1..] {
        acc = op(acc, dispatch::evaluate_expression(expr, ctx)?.as_f64()?);
    }
    Some(Value::Number(serde_json::Number::from_f64(acc)?))
}
