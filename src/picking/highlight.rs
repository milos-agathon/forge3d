// src/picking/highlight.rs
// Highlight rendering styles (outline, glow, color tint)
// Part of Plan 3: Premium — Unified Picking with BVH + Python Callbacks

use bytemuck::{Pod, Zeroable};

/// Highlight effect type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightEffect {
    /// No effect
    None,
    /// Solid color tint
    ColorTint,
    /// Outline around feature
    Outline,
    /// Glow effect
    Glow,
    /// Combination of outline and glow
    OutlineGlow,
}

impl Default for HighlightEffect {
    fn default() -> Self {
        Self::ColorTint
    }
}

/// Configuration for a highlight style
#[derive(Debug, Clone)]
pub struct HighlightStyle {
    /// Primary highlight color (RGBA)
    pub color: [f32; 4],
    /// Secondary color for gradients/glow (RGBA)
    pub secondary_color: [f32; 4],
    /// Effect type
    pub effect: HighlightEffect,
    /// Outline width in pixels
    pub outline_width: f32,
    /// Glow intensity (0.0 - 1.0)
    pub glow_intensity: f32,
    /// Glow radius in pixels
    pub glow_radius: f32,
    /// Animation speed (0 = no animation)
    pub pulse_speed: f32,
    /// Z-bias for depth fighting prevention
    pub depth_bias: f32,
}

impl Default for HighlightStyle {
    fn default() -> Self {
        Self {
            color: [1.0, 0.8, 0.0, 0.5],      // Yellow semi-transparent
            secondary_color: [1.0, 1.0, 1.0, 0.3], // White glow
            effect: HighlightEffect::ColorTint,
            outline_width: 2.0,
            glow_intensity: 0.5,
            glow_radius: 8.0,
            pulse_speed: 0.0,
            depth_bias: 0.001,
        }
    }
}

impl HighlightStyle {
    /// Create a color tint style
    pub fn color_tint(color: [f32; 4]) -> Self {
        Self {
            color,
            effect: HighlightEffect::ColorTint,
            ..Default::default()
        }
    }

    /// Create an outline style
    pub fn outline(color: [f32; 4], width: f32) -> Self {
        Self {
            color,
            effect: HighlightEffect::Outline,
            outline_width: width,
            ..Default::default()
        }
    }

    /// Create a glow style
    pub fn glow(color: [f32; 4], intensity: f32, radius: f32) -> Self {
        Self {
            color,
            effect: HighlightEffect::Glow,
            glow_intensity: intensity,
            glow_radius: radius,
            ..Default::default()
        }
    }

    /// Create a combined outline + glow style
    pub fn outline_glow(
        outline_color: [f32; 4],
        glow_color: [f32; 4],
        outline_width: f32,
        glow_intensity: f32,
    ) -> Self {
        Self {
            color: outline_color,
            secondary_color: glow_color,
            effect: HighlightEffect::OutlineGlow,
            outline_width,
            glow_intensity,
            ..Default::default()
        }
    }

    /// Add pulsing animation
    pub fn with_pulse(mut self, speed: f32) -> Self {
        self.pulse_speed = speed;
        self
    }
}

/// GPU-compatible highlight uniforms
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct HighlightUniforms {
    /// Primary highlight color
    pub color: [f32; 4],
    /// Secondary color for glow
    pub secondary_color: [f32; 4],
    /// Outline width, glow intensity, glow radius, pulse phase
    pub params: [f32; 4],
    /// Selected feature ID (0 = none)
    pub selected_id: u32,
    /// Hover feature ID (0 = none)
    pub hover_id: u32,
    /// Effect flags (bit 0: color tint, bit 1: outline, bit 2: glow)
    pub effect_flags: u32,
    /// Padding
    pub _pad: u32,
}

impl HighlightUniforms {
    /// Create from highlight style
    pub fn from_style(style: &HighlightStyle, selected_id: u32, hover_id: u32, time: f32) -> Self {
        let pulse_phase = if style.pulse_speed > 0.0 {
            (time * style.pulse_speed).sin() * 0.5 + 0.5
        } else {
            1.0
        };

        let effect_flags = match style.effect {
            HighlightEffect::None => 0,
            HighlightEffect::ColorTint => 1,
            HighlightEffect::Outline => 2,
            HighlightEffect::Glow => 4,
            HighlightEffect::OutlineGlow => 6,
        };

        Self {
            color: style.color,
            secondary_color: style.secondary_color,
            params: [
                style.outline_width,
                style.glow_intensity,
                style.glow_radius,
                pulse_phase,
            ],
            selected_id,
            hover_id,
            effect_flags,
            _pad: 0,
        }
    }
}

/// WGSL shader code for highlight rendering
pub const HIGHLIGHT_SHADER_FUNCTIONS: &str = r#"
// Highlight shader functions for Plan 3 picking system

struct HighlightUniforms {
    color: vec4<f32>,
    secondary_color: vec4<f32>,
    params: vec4<f32>,  // outline_width, glow_intensity, glow_radius, pulse_phase
    selected_id: u32,
    hover_id: u32,
    effect_flags: u32,
    _pad: u32,
};

// Check if feature should be highlighted
fn is_highlighted(feature_id: u32, highlight: HighlightUniforms) -> bool {
    return feature_id == highlight.selected_id || feature_id == highlight.hover_id;
}

// Apply color tint highlight
fn apply_color_tint(base_color: vec4<f32>, highlight: HighlightUniforms, is_selected: bool) -> vec4<f32> {
    if (!is_selected) {
        return base_color;
    }
    
    let tint = highlight.color;
    let pulse = highlight.params.w;
    let blend_factor = tint.a * pulse;
    
    return vec4<f32>(
        mix(base_color.rgb, tint.rgb, blend_factor),
        base_color.a
    );
}

// Calculate outline factor for edge detection
fn calculate_outline_factor(
    feature_id: u32,
    neighbor_ids: array<u32, 4>,  // left, right, up, down
    highlight: HighlightUniforms
) -> f32 {
    if (feature_id != highlight.selected_id && feature_id != highlight.hover_id) {
        return 0.0;
    }
    
    var edge_count = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (neighbor_ids[i] != feature_id) {
            edge_count = edge_count + 1u;
        }
    }
    
    return f32(edge_count) / 4.0;
}

// Apply glow effect (simplified for fragment shader)
fn apply_glow(
    base_color: vec4<f32>,
    glow_factor: f32,
    highlight: HighlightUniforms
) -> vec4<f32> {
    let glow_intensity = highlight.params.y;
    let pulse = highlight.params.w;
    
    let glow_color = highlight.secondary_color.rgb;
    let glow_strength = glow_factor * glow_intensity * pulse;
    
    return vec4<f32>(
        base_color.rgb + glow_color * glow_strength,
        base_color.a
    );
}
"#;

/// Highlight manager for tracking and applying highlights
#[derive(Debug)]
pub struct HighlightManager {
    /// Style for primary selection
    pub selection_style: HighlightStyle,
    /// Style for hover highlight
    pub hover_style: HighlightStyle,
    /// Currently selected feature ID
    pub selected_id: Option<u32>,
    /// Currently hovered feature ID
    pub hover_id: Option<u32>,
    /// Animation time
    pub time: f32,
}

impl Default for HighlightManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HighlightManager {
    /// Create new highlight manager
    pub fn new() -> Self {
        Self {
            selection_style: HighlightStyle::default(),
            hover_style: HighlightStyle {
                color: [0.5, 0.8, 1.0, 0.3], // Light blue
                effect: HighlightEffect::ColorTint,
                ..Default::default()
            },
            selected_id: None,
            hover_id: None,
            time: 0.0,
        }
    }

    /// Update animation time
    pub fn update(&mut self, delta_time: f32) {
        self.time += delta_time;
    }

    /// Set selection style
    pub fn set_selection_style(&mut self, style: HighlightStyle) {
        self.selection_style = style;
    }

    /// Set hover style
    pub fn set_hover_style(&mut self, style: HighlightStyle) {
        self.hover_style = style;
    }

    /// Set selected feature
    pub fn set_selected(&mut self, feature_id: Option<u32>) {
        self.selected_id = feature_id;
    }

    /// Set hovered feature
    pub fn set_hovered(&mut self, feature_id: Option<u32>) {
        self.hover_id = feature_id;
    }

    /// Get selection uniforms
    pub fn get_selection_uniforms(&self) -> HighlightUniforms {
        HighlightUniforms::from_style(
            &self.selection_style,
            self.selected_id.unwrap_or(0),
            0,
            self.time,
        )
    }

    /// Get hover uniforms
    pub fn get_hover_uniforms(&self) -> HighlightUniforms {
        HighlightUniforms::from_style(
            &self.hover_style,
            0,
            self.hover_id.unwrap_or(0),
            self.time,
        )
    }

    /// Get combined uniforms for both selection and hover
    pub fn get_combined_uniforms(&self) -> HighlightUniforms {
        // Use selection style but include both IDs
        HighlightUniforms::from_style(
            &self.selection_style,
            self.selected_id.unwrap_or(0),
            self.hover_id.unwrap_or(0),
            self.time,
        )
    }

    /// Check if a feature is highlighted
    pub fn is_highlighted(&self, feature_id: u32) -> bool {
        self.selected_id == Some(feature_id) || self.hover_id == Some(feature_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_style_creation() {
        let style = HighlightStyle::outline([1.0, 0.0, 0.0, 1.0], 3.0);
        assert_eq!(style.effect, HighlightEffect::Outline);
        assert_eq!(style.outline_width, 3.0);

        let style = HighlightStyle::glow([0.0, 1.0, 0.0, 1.0], 0.8, 10.0);
        assert_eq!(style.effect, HighlightEffect::Glow);
        assert_eq!(style.glow_intensity, 0.8);
    }

    #[test]
    fn test_highlight_uniforms() {
        let style = HighlightStyle::default();
        let uniforms = HighlightUniforms::from_style(&style, 42, 0, 0.0);
        
        assert_eq!(uniforms.selected_id, 42);
        assert_eq!(uniforms.hover_id, 0);
        assert_eq!(uniforms.effect_flags, 1); // ColorTint
    }

    #[test]
    fn test_highlight_manager() {
        let mut manager = HighlightManager::new();
        
        manager.set_selected(Some(10));
        manager.set_hovered(Some(20));
        
        assert!(manager.is_highlighted(10));
        assert!(manager.is_highlighted(20));
        assert!(!manager.is_highlighted(30));
    }
}
