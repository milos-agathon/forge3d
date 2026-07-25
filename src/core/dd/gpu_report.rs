use crate::core::error::{RenderError, RenderResult};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TwoProdVariant {
    Fma,
    Split,
}

impl TwoProdVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fma => "fma",
            Self::Split => "split",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DdSelftestReport {
    pub passed: bool,
    pub backend: String,
    pub adapter: String,
    pub two_prod_variant: TwoProdVariant,
    pub shader_label: String,
    pub shader_hash: String,
    pub canary_count: u64,
    pub mismatch_count: u64,
    pub rejected_variants: Vec<TwoProdVariant>,
    pub failure_details: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DdOperation {
    Add,
    Mul,
    Div,
    Sqrt,
}

impl DdOperation {
    pub fn parse(value: &str) -> RenderResult<Self> {
        match value {
            "add" => Ok(Self::Add),
            "mul" => Ok(Self::Mul),
            "div" => Ok(Self::Div),
            "sqrt" => Ok(Self::Sqrt),
            _ => Err(RenderError::render(format!(
                "unknown DD operation '{value}'; expected add|mul|div|sqrt"
            ))),
        }
    }

    pub(super) fn code(self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Mul => 1,
            Self::Div => 2,
            Self::Sqrt => 3,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Sqrt => "sqrt",
        }
    }

    pub(super) fn bound(self) -> f64 {
        match self {
            Self::Add => super::DD_ADD_BOUND_U2,
            Self::Mul => super::DD_MUL_BOUND_U2,
            Self::Div => super::DD_DIV_BOUND_U2,
            Self::Sqrt => super::DD_SQRT_BOUND_U2,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DdHarnessReport {
    pub operation: DdOperation,
    pub backend: String,
    pub adapter: String,
    pub two_prod_variant: TwoProdVariant,
    pub shader_label: String,
    pub shader_hash: String,
    pub generated_count: u64,
    pub adversarial_count: u64,
    pub mismatch_count: u64,
    pub max_err_u2: f64,
    pub cited_bound_u2: f64,
    pub certificate_json: String,
}
