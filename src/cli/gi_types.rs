// src/cli/gi_types.rs
// GI-related CLI types and enums
// Extracted from args.rs for maintainability (<300 lines)

use std::fmt;

/// Toggle state used by GI-related CLI flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Toggle {
    On,
    Off,
}

impl Toggle {
    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "on" | "1" | "true" => Some(Toggle::On),
            "off" | "0" | "false" => Some(Toggle::Off),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Toggle::On => "on",
            Toggle::Off => "off",
        }
    }
}

/// GI effects supported by CLI flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GiEffect {
    Ssao,
    Ssgi,
    Ssr,
    Gtao,
}

impl GiEffect {
    pub fn as_str(self) -> &'static str {
        match self {
            GiEffect::Ssao => "ssao",
            GiEffect::Ssgi => "ssgi",
            GiEffect::Ssr => "ssr",
            GiEffect::Gtao => "gtao",
        }
    }
}

/// Parsed `--gi` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GiEntry {
    Off,
    Effect(GiEffect, Toggle),
}

/// GI visualization mode for debug output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GiVizMode {
    None,
    Composite,
    Ao,
    Ssgi,
    Ssr,
}

impl GiVizMode {
    pub fn as_str(self) -> &'static str {
        match self {
            GiVizMode::None => "none",
            GiVizMode::Composite => "composite",
            GiVizMode::Ao => "ao",
            GiVizMode::Ssgi => "ssgi",
            GiVizMode::Ssr => "ssr",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Some(GiVizMode::None),
            "composite" => Some(GiVizMode::Composite),
            "ao" => Some(GiVizMode::Ao),
            "ssgi" => Some(GiVizMode::Ssgi),
            "ssr" => Some(GiVizMode::Ssr),
            _ => None,
        }
    }
}

/// Error raised when parsing GI CLI flags.
#[derive(Debug)]
pub struct GiCliError {
    msg: String,
}

impl GiCliError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl fmt::Display for GiCliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.msg.fmt(f)
    }
}

impl std::error::Error for GiCliError {}
