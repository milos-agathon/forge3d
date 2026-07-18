use super::{Interval, Value};
use naga::MathFunction;

pub(super) fn multiply_values(left: Value, right: Value) -> Option<Value> {
    if let Some(value) = left.clone().binary_float(right.clone(), Interval::mul) {
        return Some(value);
    }
    match (left, right) {
        (Value::Composite(columns), Value::Composite(vector))
            if columns
                .first()
                .is_some_and(|column| matches!(column, Value::Composite(_)))
                && columns.len() == vector.len() =>
        {
            let rows = match columns.first()? {
                Value::Composite(rows) => rows.len(),
                _ => return None,
            };
            let mut output = vec![Interval::constant(0.0); rows];
            for (column, scalar) in columns.into_iter().zip(vector) {
                let (Value::Composite(column), Value::Float(scalar)) = (column, scalar) else {
                    return None;
                };
                for (sum, component) in output.iter_mut().zip(column) {
                    let Value::Float(component) = component else {
                        return None;
                    };
                    *sum = sum.add(component.mul(scalar));
                }
            }
            Some(Value::Composite(
                output.into_iter().map(Value::Float).collect(),
            ))
        }
        _ => None,
    }
}

pub(super) fn eval_math(
    fun: MathFunction,
    arg: Value,
    arg1: Option<Value>,
    arg2: Option<Value>,
) -> Option<Value> {
    use MathFunction as M;
    match fun {
        M::Abs => arg.unary_float(Interval::abs),
        M::Sign => arg.unary_float(Interval::sign),
        M::Min => {
            let rhs = arg1?;
            arg.clone()
                .binary_float(rhs.clone(), Interval::min)
                .or_else(|| arg.binary_int(rhs, i64::min))
        }
        M::Max => {
            let rhs = arg1?;
            arg.clone()
                .binary_float(rhs.clone(), Interval::max)
                .or_else(|| arg.binary_int(rhs, i64::max))
        }
        M::Clamp => {
            let low = arg1?;
            let high = arg2?;
            binary3(
                arg.clone(),
                low.clone(),
                high.clone(),
                |value, low, high| value.clamp(low, high),
            )
            .or_else(|| ternary_int(arg, low, high))
        }
        M::Saturate => {
            arg.unary_float(|value| value.clamp(Interval::constant(0.0), Interval::constant(1.0)))
        }
        M::Mix => binary3(arg, arg1?, arg2?, |left, right, factor| {
            left.mix(right, factor)
        }),
        M::Fma => binary3(arg, arg1?, arg2?, Interval::fma),
        M::Sqrt => arg.unary_float(Interval::sqrt),
        M::InverseSqrt => arg.unary_float(Interval::inverse_sqrt),
        M::Dot => {
            let (left, right) = (float_lanes(arg)?, float_lanes(arg1?)?);
            Some(Value::Float(crate::verify::domain::dot(&left, &right)))
        }
        M::Length => Some(Value::Float(
            float_lanes(arg)?
                .into_iter()
                .fold(Interval::constant(0.0), |sum, lane| sum.add(lane.square()))
                .nonnegative_sqrt(),
        )),
        M::Distance => {
            let delta = arg.binary_float(arg1?, Interval::sub)?;
            let lanes = float_lanes(delta)?;
            Some(Value::Float(
                crate::verify::domain::dot(&lanes, &lanes).sqrt(),
            ))
        }
        M::Normalize => {
            let lanes = float_lanes(arg)?;
            let length = lanes
                .iter()
                .copied()
                .fold(Interval::constant(0.0), |sum, lane| sum.add(lane.square()))
                .nonnegative_sqrt();
            Some(Value::Composite(
                lanes
                    .into_iter()
                    .map(|lane| Value::Float(lane.div(length)))
                    .collect(),
            ))
        }
        M::Cross => {
            let (left, right) = (float_lanes(arg)?, float_lanes(arg1?)?);
            if left.len() != 3 || right.len() != 3 {
                return None;
            }
            Some(Value::Composite(vec![
                Value::Float(left[1].mul(right[2]).sub(left[2].mul(right[1]))),
                Value::Float(left[2].mul(right[0]).sub(left[0].mul(right[2]))),
                Value::Float(left[0].mul(right[1]).sub(left[1].mul(right[0]))),
            ]))
        }
        M::Reflect => {
            let incident = arg;
            let normal = arg1?;
            let dot = crate::verify::domain::dot(
                &float_lanes(normal.clone())?,
                &float_lanes(incident.clone())?,
            );
            incident.binary_float(
                normal.binary_float(
                    Value::Float(Interval::constant(2.0).mul(dot)),
                    Interval::mul,
                )?,
                Interval::sub,
            )
        }
        M::Sin | M::Cos => arg.unary_float(|_| Interval::new(-1.0, 1.0)),
        M::Tan => arg.unary_float(|value| {
            let lo = value.lo as f64;
            let hi = value.hi as f64;
            let half_pi = std::f64::consts::FRAC_PI_2;
            let first_pole =
                ((lo - half_pi) / std::f64::consts::PI).ceil() * std::f64::consts::PI + half_pi;
            if first_pole <= hi {
                Interval::unknown()
            } else {
                let a = lo.tan();
                let b = hi.tan();
                Interval::rounded_bounds(a.min(b), a.max(b))
            }
        }),
        M::Acos => arg.unary_float(|value| {
            if value.lo >= -1.0 && value.hi <= 1.0 {
                Interval::rounded_bounds((value.hi as f64).acos(), (value.lo as f64).acos())
            } else {
                Interval::unknown()
            }
        }),
        M::Asin => arg.unary_float(|value| {
            if value.lo >= -1.0 && value.hi <= 1.0 {
                Interval::rounded_bounds((value.lo as f64).asin(), (value.hi as f64).asin())
            } else {
                Interval::unknown()
            }
        }),
        M::Atan => arg.unary_float(|value| {
            Interval::rounded_bounds((value.lo as f64).atan(), (value.hi as f64).atan())
        }),
        M::Atan2 => arg.binary_float(arg1?, |_, _| {
            Interval::rounded_bounds(-std::f64::consts::PI, std::f64::consts::PI)
        }),
        M::Exp => arg.unary_float(|value| {
            Interval::rounded_bounds((value.lo as f64).exp(), (value.hi as f64).exp())
        }),
        M::Exp2 => arg.unary_float(|value| {
            Interval::rounded_bounds((value.lo as f64).exp2(), (value.hi as f64).exp2())
        }),
        M::Log => arg.unary_float(|value| {
            if value.lo > 0.0 {
                Interval::rounded_bounds((value.lo as f64).ln(), (value.hi as f64).ln())
            } else {
                Interval::unknown()
            }
        }),
        M::Log2 => arg.unary_float(|value| {
            if value.lo > 0.0 {
                Interval::rounded_bounds((value.lo as f64).log2(), (value.hi as f64).log2())
            } else {
                Interval::unknown()
            }
        }),
        M::Pow => arg.binary_float(arg1?, |base, exponent| {
            if base.lo < 0.0 || (base.lo == 0.0 && exponent.lo <= 0.0) {
                return Interval::unknown();
            }
            if base.lo == 0.0 {
                let positive = Interval::new(f32::MIN_POSITIVE, base.hi.max(f32::MIN_POSITIVE));
                return Interval::rounded_bounds(
                    0.0,
                    (positive.hi as f64)
                        .powf(exponent.lo as f64)
                        .max((positive.hi as f64).powf(exponent.hi as f64)),
                );
            }
            let candidates = [
                (base.lo as f64).powf(exponent.lo as f64),
                (base.lo as f64).powf(exponent.hi as f64),
                (base.hi as f64).powf(exponent.lo as f64),
                (base.hi as f64).powf(exponent.hi as f64),
            ];
            Interval::rounded_bounds(
                candidates.iter().copied().fold(f64::INFINITY, f64::min),
                candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        }),
        M::Floor => arg.unary_float(|value| {
            Interval::rounded_bounds((value.lo as f64).floor(), (value.hi as f64).floor())
        }),
        M::Ceil => arg.unary_float(|value| {
            Interval::rounded_bounds((value.lo as f64).ceil(), (value.hi as f64).ceil())
        }),
        M::Round => arg.unary_float(|value| {
            Interval::rounded_bounds((value.lo as f64).round(), (value.hi as f64).round())
        }),
        M::Trunc => arg.unary_float(|value| {
            Interval::rounded_bounds(
                (value.lo as f64).trunc().min(0.0),
                (value.hi as f64).trunc().max(0.0),
            )
        }),
        M::Fract => arg.unary_float(|_| Interval::new(0.0, 1.0)),
        M::Step | M::SmoothStep => Some(shape_like_float(
            &arg1.unwrap_or(arg),
            Interval::new(0.0, 1.0),
        )),
        M::Radians => {
            arg.unary_float(|value| value.mul(Interval::constant(std::f32::consts::PI / 180.0)))
        }
        M::Degrees => {
            arg.unary_float(|value| value.mul(Interval::constant(180.0 / std::f32::consts::PI)))
        }
        _ => None,
    }
}

pub(super) fn binary3(
    left: Value,
    middle: Value,
    right: Value,
    op: impl Fn(Interval, Interval, Interval) -> Interval + Copy,
) -> Option<Value> {
    match (left, middle, right) {
        (Value::Float(left), Value::Float(middle), Value::Float(right)) => {
            Some(Value::Float(op(left, middle, right)))
        }
        (left, middle, right) => {
            let width = [&left, &middle, &right]
                .into_iter()
                .find_map(|value| match value {
                    Value::Composite(values) => Some(values.len()),
                    _ => None,
                })?;
            let lanes = |value: Value| match value {
                Value::Composite(values) if values.len() == width => Some(values),
                Value::Float(_) => Some(vec![value; width]),
                _ => None,
            };
            lanes(left)?
                .into_iter()
                .zip(lanes(middle)?)
                .zip(lanes(right)?)
                .map(|((left, middle), right)| binary3(left, middle, right, op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite)
        }
    }
}

pub(super) fn eval_int_binary(
    left: Value,
    right: Value,
    op: naga::BinaryOperator,
) -> Option<Value> {
    let (Value::Int { lo: ll, hi: lh }, Value::Int { lo: rl, hi: rh }) = (left, right) else {
        return None;
    };
    use naga::BinaryOperator as B;
    match op {
        B::ShiftRight if rl == rh && (0..32).contains(&rl) => Some(Value::Int {
            lo: ll >> rl,
            hi: lh >> rl,
        }),
        B::Divide if rl == rh && rl != 0 => {
            let values = [ll / rl, lh / rl];
            Some(Value::Int {
                lo: *values.iter().min().unwrap(),
                hi: *values.iter().max().unwrap(),
            })
        }
        B::Modulo if rl == rh && rl > 0 && ll >= 0 => Some(Value::Int {
            lo: 0,
            hi: (rh - 1).min(lh),
        }),
        B::And if rl == rh && rl >= 0 => Some(Value::Int {
            lo: 0,
            hi: rh.min(lh.max(0)),
        }),
        _ => None,
    }
}

fn ternary_int(value: Value, low: Value, high: Value) -> Option<Value> {
    match (value, low, high) {
        (Value::Int { lo, hi }, Value::Int { lo: low, .. }, Value::Int { hi: high, .. }) => {
            Some(Value::Int {
                lo: lo.max(low).min(high),
                hi: hi.max(low).min(high),
            })
        }
        (Value::Composite(value), Value::Composite(low), Value::Composite(high))
            if value.len() == low.len() && value.len() == high.len() =>
        {
            value
                .into_iter()
                .zip(low)
                .zip(high)
                .map(|((value, low), high)| ternary_int(value, low, high))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite)
        }
        _ => None,
    }
}

pub(super) fn float_lanes(value: Value) -> Option<Vec<Interval>> {
    match value {
        Value::Float(value) => Some(vec![value]),
        Value::Composite(values) => values
            .into_iter()
            .map(|value| match value {
                Value::Float(value) => Some(value),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

pub(super) fn shape_like_float(value: &Value, interval: Interval) -> Value {
    match value {
        Value::Composite(values) => Value::Composite(
            values
                .iter()
                .map(|value| shape_like_float(value, interval))
                .collect(),
        ),
        _ => Value::Float(interval),
    }
}

pub(super) fn bitcast_value(value: Value, kind: naga::ScalarKind) -> Value {
    match value {
        Value::Composite(values) => Value::Composite(
            values
                .into_iter()
                .map(|value| bitcast_value(value, kind))
                .collect(),
        ),
        Value::Float(value) if kind == naga::ScalarKind::Uint => {
            if value.is_finite_only() && value.lo >= 0.0 {
                Value::Int {
                    lo: value.lo.to_bits() as i64,
                    hi: value.hi.to_bits() as i64,
                }
            } else {
                Value::Int {
                    lo: 0,
                    hi: u32::MAX as i64,
                }
            }
        }
        Value::Int { lo, hi } if kind == naga::ScalarKind::Float => {
            if lo == hi && (0..=u32::MAX as i64).contains(&lo) {
                Value::Float(Interval::constant(f32::from_bits(lo as u32)))
            } else if lo >= 0 && hi <= 0x7f7f_ffff {
                Value::Float(Interval::new(
                    f32::from_bits(lo as u32),
                    f32::from_bits(hi as u32),
                ))
            } else {
                Value::Float(Interval::unknown())
            }
        }
        value => value,
    }
}

pub(super) fn convert_value(value: Value, kind: naga::ScalarKind) -> Value {
    match value {
        Value::Composite(values) => Value::Composite(
            values
                .into_iter()
                .map(|value| convert_value(value, kind))
                .collect(),
        ),
        Value::Int { lo, hi } if kind == naga::ScalarKind::Float => {
            Value::Float(Interval::rounded_bounds(lo as f64, hi as f64))
        }
        Value::Float(value) if kind == naga::ScalarKind::Uint => Value::Int {
            lo: value.lo.max(0.0) as i64,
            hi: value.hi.max(0.0).min(u32::MAX as f32) as i64,
        },
        Value::Float(value) if kind == naga::ScalarKind::Sint => Value::Int {
            lo: value.lo.max(i32::MIN as f32) as i64,
            hi: value.hi.min(i32::MAX as f32) as i64,
        },
        value => value,
    }
}

pub(super) fn eval_relational(fun: naga::RelationalFunction, value: Value) -> Value {
    let possible = match (fun, &value) {
        (naga::RelationalFunction::IsNan, Value::Float(value)) => {
            (value.is_finite_only(), value.may_nan)
        }
        (naga::RelationalFunction::IsInf, Value::Float(value)) => (
            !value.may_pos_inf && !value.may_neg_inf,
            value.may_pos_inf || value.may_neg_inf,
        ),
        _ => (true, true),
    };
    Value::Bool {
        can_false: possible.0,
        can_true: possible.1,
    }
}

pub(super) fn derivative_value(value: Value) -> Value {
    value
        .unary_float(|value| {
            if value.is_finite_only() {
                Interval::rounded_bounds(
                    (value.lo as f64 - value.hi as f64) * 2.0,
                    (value.hi as f64 - value.lo as f64) * 2.0,
                )
            } else {
                Interval::unknown()
            }
        })
        .unwrap_or(Value::Opaque)
}
