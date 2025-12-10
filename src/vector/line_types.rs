// src/vector/line_types.rs
// Type definitions for anti-aliased line rendering
// RELEVANT FILES: shaders/line_aa.wgsl

use bytemuck::{Pod, Zeroable};

/// Line cap styles for polyline endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    /// Square cap ending exactly at vertex
    Butt = 0,
    /// Rounded cap extending beyond vertex
    Round = 1,
    /// Square cap extending beyond vertex
    Square = 2,
}

/// Line join styles for polyline vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    /// Sharp miter join with limit
    Miter = 0,
    /// Beveled join (flat cut)
    Bevel = 1,
    /// Rounded join
    Round = 2,
}

/// GPU uniform buffer for line rendering parameters.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineUniform {
    /// View-projection transform matrix.
    pub transform: [[f32; 4]; 4],
    /// RGBA stroke color.
    pub stroke_color: [f32; 4],
    /// Line width in world units.
    pub stroke_width: f32,
    /// Padding for std140 alignment.
    pub _pad0: f32,
    /// Viewport dimensions for anti-aliasing calculations.
    pub viewport_size: [f32; 2],
    /// Miter limit for sharp corner clamping.
    pub miter_limit: f32,
    /// Line cap style (see [`LineCap`]).
    pub cap_style: u32,
    /// Line join style (see [`LineJoin`]).
    pub join_style: u32,
    /// Tail padding to reach 128-byte std140 size.
    pub _pad1: [f32; 5],
}

/// Per-segment instance data for GPU line expansion.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineInstance {
    /// Segment start position in world space.
    pub start_pos: [f32; 2],
    /// Segment end position in world space.
    pub end_pos: [f32; 2],
    /// Line width in world units.
    pub width: f32,
    /// RGBA stroke color.
    pub color: [f32; 4],
    /// Miter limit for sharp corner clamping.
    pub miter_limit: f32,
    /// Alignment padding.
    pub _pad: [f32; 2],
}
