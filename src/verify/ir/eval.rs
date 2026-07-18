use super::engine::Evaluator;
use super::value::Value;
use crate::verify::domain::Interval;
use naga::{Expression, Handle, Literal};

impl Evaluator<'_> {
    pub(super) fn eval_const(&self, handle: Handle<Expression>) -> Value {
        match &self.module.const_expressions[handle] {
            Expression::Literal(Literal::F32(value)) => Value::Float(Interval::constant(*value)),
            Expression::Literal(Literal::AbstractFloat(value)) => {
                Value::Float(Interval::constant(*value as f32))
            }
            Expression::Literal(Literal::U32(value)) => Value::Int {
                lo: *value as i64,
                hi: *value as i64,
            },
            Expression::Literal(Literal::I32(value)) => Value::Int {
                lo: *value as i64,
                hi: *value as i64,
            },
            Expression::Literal(Literal::Bool(value)) => Value::Bool {
                can_false: !*value,
                can_true: *value,
            },
            Expression::ZeroValue(ty) => Value::zero(self.module, *ty),
            Expression::Compose { components, .. } => Value::Composite(
                components
                    .iter()
                    .map(|handle| self.eval_const(*handle))
                    .collect(),
            ),
            Expression::Splat { size, value } => self.eval_const(*value).splat(*size),
            Expression::Constant(constant) => {
                self.eval_const(self.module.constants[*constant].init)
            }
            _ => Value::Opaque,
        }
    }
}
