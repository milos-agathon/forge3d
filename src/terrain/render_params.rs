// src/terrain_render_params.rs
// PyO3 terrain render parameter wrapper bridging Python configs to Rust
// Exists to store validated terrain settings in a native-friendly structure
// RELEVANT FILES: python/forge3d/terrain_params.py, src/overlay_layer.rs, src/terrain_renderer.rs, tests/test_terrain_render_params_native.py
#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use std::sync::Arc;

mod core;
mod decode_atmosphere;
mod decode_core;
mod decode_effects;
mod decode_lighting;
mod decode_materials;
mod decode_postfx;
mod native_effects;
mod native_lighting;
mod native_material;
mod native_overlays;
mod native_postfx;
mod parse;
mod private_impl;
mod py_api;

use native_effects::{
    BloomSettingsNative, FogSettingsNative, HeightAoSettingsNative, ReflectionSettingsNative,
    SunVisibilitySettingsNative,
};
use native_lighting::{
    ClampSettingsNative, LightSettingsNative, LodSettingsNative, PomSettingsNative,
    SamplingSettingsNative, ShadowSettingsNative, TriplanarSettingsNative,
};
use native_material::{DetailSettingsNative, MaterialLayerSettingsNative, MaterialNoiseSettingsNative};
use native_overlays::VectorOverlaySettingsNative;
use native_postfx::{
    AovSettingsNative, DenoiseMethodNative, DenoiseSettingsNative, DofMethodNative,
    DofQualityNative, DofSettingsNative, LensEffectsSettingsNative, MotionBlurSettingsNative,
    SkySettingsNative, TonemapSettingsNative, VolumetricsModeNative, VolumetricsSettingsNative,
};

pub use core::{DecodedTerrainSettings, TerrainRenderParams};
pub use native_lighting::{AddressModeNative, FilterModeNative};
