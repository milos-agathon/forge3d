#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct AovSettingsNative {
    pub enabled: bool,
    pub albedo: bool,
    pub normal: bool,
    pub depth: bool,
    pub output_dir: Option<String>,
    pub format: String,
}

#[cfg(feature = "extension-module")]
impl Default for AovSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            albedo: true,
            normal: true,
            depth: true,
            output_dir: None,
            format: "png".to_string(),
        }
    }
}

#[cfg(feature = "extension-module")]
impl AovSettingsNative {
    pub fn any_enabled(&self) -> bool {
        self.enabled && (self.albedo || self.normal || self.depth)
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct DenoiseSettingsNative {
    pub enabled: bool,
    pub method: DenoiseMethodNative,
    pub iterations: u32,
    pub sigma_color: f32,
    pub sigma_normal: f32,
    pub sigma_depth: f32,
    pub edge_stopping: f32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMethodNative {
    Atrous,
    Bilateral,
    None,
}

#[cfg(feature = "extension-module")]
impl Default for DenoiseSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            method: DenoiseMethodNative::Atrous,
            iterations: 3,
            sigma_color: 0.1,
            sigma_normal: 0.1,
            sigma_depth: 0.1,
            edge_stopping: 1.0,
        }
    }
}

#[cfg(feature = "extension-module")]
impl DenoiseSettingsNative {
    pub fn uses_guidance(&self) -> bool {
        (self.sigma_normal > 0.001 || self.sigma_depth > 0.001)
            && self.method == DenoiseMethodNative::Atrous
    }
}
