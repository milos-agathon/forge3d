use crate::verify::domain::Interval;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    Float(Interval),
    Int {
        lo: i64,
        hi: i64,
    },
    Bool {
        can_false: bool,
        can_true: bool,
    },
    Composite(Vec<Value>),
    Array {
        element: Box<Value>,
        length: Option<u64>,
    },
    Image {
        name: String,
        sample: Box<Value>,
        dimensions: Vec<(u64, u64)>,
    },
    Sampler,
    Pointer(Place),
    Opaque,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Place {
    pub root: Root,
    pub path: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Root {
    Argument(usize),
    Global(usize),
    Local(usize),
    Alias(usize),
}

impl Value {
    pub(super) fn from_range(
        module: &naga::Module,
        ty: naga::Handle<naga::Type>,
        lo: f32,
        hi: f32,
    ) -> Self {
        Self::from_inner(module, &module.types[ty].inner, Some((lo, hi)))
    }

    pub(super) fn zero(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Self {
        Self::zero_inner(module, &module.types[ty].inner)
    }

    pub(super) fn unknown(module: &naga::Module, inner: &naga::TypeInner) -> Self {
        Self::from_inner(module, inner, None)
    }

    fn zero_inner(module: &naga::Module, inner: &naga::TypeInner) -> Self {
        use naga::{ArraySize, ScalarKind, TypeInner};
        match inner {
            TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => match scalar.kind {
                ScalarKind::Float => Value::Float(Interval::constant(0.0)),
                ScalarKind::Uint | ScalarKind::Sint => Value::Int { lo: 0, hi: 0 },
                ScalarKind::Bool => Value::Bool {
                    can_false: true,
                    can_true: false,
                },
                _ => Value::Opaque,
            },
            TypeInner::Vector { size, scalar } => Value::Composite(
                (0..lanes(*size))
                    .map(|_| Self::zero_inner(module, &TypeInner::Scalar(*scalar)))
                    .collect(),
            ),
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => Value::Composite(
                (0..lanes(*columns))
                    .map(|_| {
                        Self::zero_inner(
                            module,
                            &TypeInner::Vector {
                                size: *rows,
                                scalar: *scalar,
                            },
                        )
                    })
                    .collect(),
            ),
            TypeInner::Struct { members, .. } => Value::Composite(
                members
                    .iter()
                    .map(|member| Self::zero(module, member.ty))
                    .collect(),
            ),
            TypeInner::Array {
                base,
                size: ArraySize::Constant(size),
                ..
            } => Value::Composite((0..size.get()).map(|_| Self::zero(module, *base)).collect()),
            TypeInner::Array {
                base,
                size: ArraySize::Dynamic,
                ..
            } => Value::Array {
                element: Box::new(Self::zero(module, *base)),
                length: None,
            },
            _ => Value::Opaque,
        }
    }

    fn from_inner(
        module: &naga::Module,
        inner: &naga::TypeInner,
        range: Option<(f32, f32)>,
    ) -> Self {
        use naga::{ArraySize, ScalarKind, TypeInner};
        match inner {
            TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => match scalar.kind {
                ScalarKind::Float => Value::Float(
                    range.map_or_else(Interval::unknown, |(lo, hi)| Interval::new(lo, hi)),
                ),
                ScalarKind::Uint => Value::Int {
                    lo: range.map_or(0, |(lo, _)| lo.max(0.0) as i64),
                    hi: range.map_or(u32::MAX as i64, |(_, hi)| hi.min(u32::MAX as f32) as i64),
                },
                ScalarKind::Sint => Value::Int {
                    lo: range.map_or(i32::MIN as i64, |(lo, _)| lo.max(i32::MIN as f32) as i64),
                    hi: range.map_or(i32::MAX as i64, |(_, hi)| hi.min(i32::MAX as f32) as i64),
                },
                ScalarKind::Bool => Value::Bool {
                    can_false: true,
                    can_true: true,
                },
                _ => Value::Opaque,
            },
            TypeInner::Vector { size, scalar } => Value::Composite(
                (0..lanes(*size))
                    .map(|_| Self::from_inner(module, &TypeInner::Scalar(*scalar), range))
                    .collect(),
            ),
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => Value::Composite(
                (0..lanes(*columns))
                    .map(|_| {
                        Self::from_inner(
                            module,
                            &TypeInner::Vector {
                                size: *rows,
                                scalar: *scalar,
                            },
                            range,
                        )
                    })
                    .collect(),
            ),
            TypeInner::Struct { members, .. } => Value::Composite(
                members
                    .iter()
                    .map(|member| Self::from_inner(module, &module.types[member.ty].inner, range))
                    .collect(),
            ),
            TypeInner::Array {
                base,
                size: ArraySize::Constant(size),
                ..
            } => Value::Composite(
                (0..size.get())
                    .map(|_| Self::from_inner(module, &module.types[*base].inner, range))
                    .collect(),
            ),
            TypeInner::Array {
                base,
                size: ArraySize::Dynamic,
                ..
            } => Value::Array {
                element: Box::new(Self::from_inner(module, &module.types[*base].inner, range)),
                length: None,
            },
            _ => Value::Opaque,
        }
    }

    pub(super) fn splat(self, size: naga::VectorSize) -> Self {
        Value::Composite((0..lanes(size)).map(|_| self.clone()).collect())
    }

    pub(super) fn access_index(&self, index: usize) -> Option<Self> {
        match self {
            Value::Composite(values) => values.get(index).cloned(),
            Value::Pointer(place) => {
                let mut place = place.clone();
                place.path.push(index);
                Some(Value::Pointer(place))
            }
            Value::Array { element, .. } => Some((**element).clone()),
            _ => None,
        }
    }

    pub(super) fn join(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Value::Float(left), Value::Float(right)) => Value::Float(left.join(*right)),
            (Value::Int { lo: ll, hi: lh }, Value::Int { lo: rl, hi: rh }) => Value::Int {
                lo: (*ll).min(*rl),
                hi: (*lh).max(*rh),
            },
            (
                Value::Bool {
                    can_false: lf,
                    can_true: lt,
                },
                Value::Bool {
                    can_false: rf,
                    can_true: rt,
                },
            ) => Value::Bool {
                can_false: *lf || *rf,
                can_true: *lt || *rt,
            },
            (Value::Composite(left), Value::Composite(right)) if left.len() == right.len() => {
                Value::Composite(
                    left.iter()
                        .zip(right)
                        .map(|(left, right)| left.join(right))
                        .collect(),
                )
            }
            (Value::Pointer(left), Value::Pointer(right)) if left == right => {
                Value::Pointer(left.clone())
            }
            (
                Value::Array {
                    element: left,
                    length: ll,
                },
                Value::Array {
                    element: right,
                    length: rl,
                },
            ) => Value::Array {
                element: Box::new(left.join(right)),
                length: if ll == rl { *ll } else { None },
            },
            (
                Value::Image {
                    name,
                    sample: left,
                    dimensions,
                },
                Value::Image { sample: right, .. },
            ) => Value::Image {
                name: name.clone(),
                sample: Box::new(left.join(right)),
                dimensions: dimensions.clone(),
            },
            (Value::Sampler, Value::Sampler) => Value::Sampler,
            _ => Value::Opaque,
        }
    }

    pub(super) fn widen(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Value::Float(left), Value::Float(right)) => Value::Float(left.widen(*right)),
            (Value::Int { lo: ll, hi: lh }, Value::Int { lo: rl, hi: rh }) => Value::Int {
                lo: if rl < ll {
                    i32::MIN as i64
                } else {
                    (*ll).min(*rl)
                },
                hi: if rh > lh {
                    u32::MAX as i64
                } else {
                    (*lh).max(*rh)
                },
            },
            (Value::Composite(left), Value::Composite(right)) if left.len() == right.len() => {
                Value::Composite(
                    left.iter()
                        .zip(right)
                        .map(|(left, right)| left.widen(right))
                        .collect(),
                )
            }
            (
                Value::Array {
                    element: left,
                    length,
                },
                Value::Array { element: right, .. },
            ) => Value::Array {
                element: Box::new(left.widen(right)),
                length: *length,
            },
            _ => self.join(rhs),
        }
    }

    pub(super) fn binary_int(self, rhs: Self, op: impl Fn(i64, i64) -> i64 + Copy) -> Option<Self> {
        match (self, rhs) {
            (Value::Int { lo: ll, hi: lh }, Value::Int { lo: rl, hi: rh }) => {
                let values = [op(ll, rl), op(ll, rh), op(lh, rl), op(lh, rh)];
                Some(Value::Int {
                    lo: *values.iter().min().unwrap(),
                    hi: *values.iter().max().unwrap(),
                })
            }
            (Value::Composite(left), Value::Composite(right)) if left.len() == right.len() => left
                .into_iter()
                .zip(right)
                .map(|(left, right)| left.binary_int(right, op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite),
            (Value::Composite(left), right @ Value::Int { .. }) => left
                .into_iter()
                .map(|left| left.binary_int(right.clone(), op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite),
            (left @ Value::Int { .. }, Value::Composite(right)) => right
                .into_iter()
                .map(|right| left.clone().binary_int(right, op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite),
            _ => None,
        }
    }

    pub(super) fn unary_float(self, op: impl Fn(Interval) -> Interval + Copy) -> Option<Self> {
        match self {
            Value::Float(value) => Some(Value::Float(op(value))),
            Value::Composite(values) => values
                .into_iter()
                .map(|value| value.unary_float(op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite),
            _ => None,
        }
    }

    pub(super) fn binary_float(
        self,
        rhs: Self,
        op: impl Fn(Interval, Interval) -> Interval + Copy,
    ) -> Option<Self> {
        match (self, rhs) {
            (Value::Float(left), Value::Float(right)) => Some(Value::Float(op(left, right))),
            (Value::Composite(left), Value::Composite(right)) if left.len() == right.len() => left
                .into_iter()
                .zip(right)
                .map(|(left, right)| left.binary_float(right, op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite),
            (Value::Composite(left), right @ Value::Float(_)) => left
                .into_iter()
                .map(|left| left.binary_float(right.clone(), op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite),
            (left @ Value::Float(_), Value::Composite(right)) => right
                .into_iter()
                .map(|right| left.clone().binary_float(right, op))
                .collect::<Option<Vec<_>>>()
                .map(Value::Composite),
            _ => None,
        }
    }

    pub(super) fn finite_only(&self) -> bool {
        match self {
            Value::Float(value) => value.is_finite_only(),
            Value::Composite(values) => values.iter().all(Value::finite_only),
            Value::Array { element, .. } => element.finite_only(),
            Value::Image { sample, .. } => sample.finite_only(),
            Value::Int { .. } | Value::Bool { .. } => true,
            Value::Sampler | Value::Pointer(_) | Value::Opaque => false,
        }
    }

    pub(super) fn definitely_nonzero_norm(&self) -> bool {
        match self {
            Value::Float(value) => value.is_finite_only() && (value.lo > 0.0 || value.hi < 0.0),
            Value::Composite(values) => values.iter().any(Value::definitely_nonzero_norm),
            _ => false,
        }
    }

    pub(super) fn within(&self, lo: f32, hi: f32) -> bool {
        match self {
            Value::Float(value) => value.is_within(lo, hi),
            Value::Int {
                lo: value_lo,
                hi: value_hi,
            } => *value_lo as f64 >= lo as f64 && *value_hi as f64 <= hi as f64,
            Value::Composite(values) => values.iter().all(|value| value.within(lo, hi)),
            Value::Array { element, .. }
            | Value::Image {
                sample: element, ..
            } => element.within(lo, hi),
            _ => false,
        }
    }

    pub(super) fn within_allow_nan(&self, lo: f32, hi: f32) -> bool {
        match self {
            Value::Float(value) => {
                !value.may_pos_inf && !value.may_neg_inf && value.lo >= lo && value.hi <= hi
            }
            Value::Int {
                lo: value_lo,
                hi: value_hi,
            } => *value_lo as f64 >= lo as f64 && *value_hi as f64 <= hi as f64,
            Value::Composite(values) => values.iter().all(|value| value.within_allow_nan(lo, hi)),
            Value::Array { element, .. }
            | Value::Image {
                sample: element, ..
            } => element.within_allow_nan(lo, hi),
            _ => false,
        }
    }

    pub(super) fn read_path(&self, path: &[usize]) -> Option<Self> {
        path.iter().try_fold(self.clone(), |value, &index| {
            if index == usize::MAX {
                match value {
                    Value::Array { element, .. } => Some(*element),
                    Value::Composite(values) => {
                        values.into_iter().reduce(|left, right| left.join(&right))
                    }
                    _ => None,
                }
            } else {
                value.access_index(index)
            }
        })
    }

    pub(super) fn write_path(&mut self, path: &[usize], value: Value) -> bool {
        let Some((&first, rest)) = path.split_first() else {
            *self = value;
            return true;
        };
        if first == usize::MAX {
            return match self {
                Value::Array { element, .. } => {
                    let joined = element.join(&value);
                    element.write_path(rest, joined)
                }
                Value::Composite(values) => {
                    for slot in values {
                        let joined = slot.join(&value);
                        if !slot.write_path(rest, joined) {
                            return false;
                        }
                    }
                    true
                }
                _ => false,
            };
        }
        match self {
            Value::Array { element, length }
                if length.is_none_or(|length| first < length as usize) =>
            {
                let joined = element.join(&value);
                element.write_path(rest, joined)
            }
            Value::Composite(values) => values
                .get_mut(first)
                .is_some_and(|slot| slot.write_path(rest, value)),
            _ => false,
        }
    }
}

pub(super) fn lanes(size: naga::VectorSize) -> usize {
    match size {
        naga::VectorSize::Bi => 2,
        naga::VectorSize::Tri => 3,
        naga::VectorSize::Quad => 4,
    }
}
