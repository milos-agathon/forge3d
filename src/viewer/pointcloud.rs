//! P5: Point cloud rendering for interactive viewer
//!
//! Supports loading and rendering LAZ/LAS point clouds with:
//! - Elevation-based coloring
//! - Point size control
//! - Frustum culling (future)

use bytemuck::{Pod, Zeroable};

/// 3D point instance for GPU rendering
/// Stores raw data so shader can compute color based on mode
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointInstance3D {
    pub position: [f32; 3],
    pub elevation_norm: f32,  // Normalized elevation [0,1] for elevation coloring
    pub rgb: [f32; 3],        // RGB color from file (or white if none)
    pub intensity: f32,       // Intensity value [0,1] (or 0.5 if none)
    pub size: f32,
    pub _pad: [f32; 3],       // Pad to 48 bytes (12 floats)
}

/// Point cloud uniforms
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointCloudUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub point_size: f32,
    pub color_mode: u32, // 0=elevation, 1=rgb, 2=intensity
    pub has_rgb: u32,       // 1 if file has RGB data
    pub has_intensity: u32, // 1 if file has meaningful intensity data
    pub _pad: [u32; 2],     // Pad to 16-byte alignment
}

/// Color mode for point cloud rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Elevation = 0,
    Rgb = 1,
    Intensity = 2,
}

impl ColorMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "rgb" => Self::Rgb,
            "intensity" => Self::Intensity,
            _ => Self::Elevation,
        }
    }
}

/// Point cloud state for the viewer
pub struct PointCloudState {
    pub points: Vec<PointInstance3D>,
    pub instance_buffer: Option<wgpu::Buffer>,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub point_count: usize,
    pub point_size: f32,
    pub visible: bool,
    pub color_mode: ColorMode,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub center: [f32; 3],
    // Data availability flags
    pub has_rgb: bool,
    pub has_intensity: bool,
    // Orbit camera state
    pub cam_phi: f32,    // Azimuth angle (horizontal rotation) in radians
    pub cam_theta: f32,  // Elevation angle (vertical rotation) in radians  
    pub cam_radius: f32, // Distance from center
}

impl PointCloudState {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        _depth_format: wgpu::TextureFormat, // Unused for now - overlay mode
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointcloud.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POINTCLOUD_SHADER)),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PointCloud.Uniforms"),
            size: std::mem::size_of::<PointCloudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PointCloud.BindGroupLayout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PointCloud.BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PointCloud.PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PointCloud.Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointInstance3D>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3, // position
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32, // elevation_norm
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3, // rgb
                        },
                        wgpu::VertexAttribute {
                            offset: 28,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32, // intensity
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32, // size
                        },
                    ],
                }],
                },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None, // Overlay mode - no depth testing
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
                    });

        Self {
            points: Vec::new(),
            instance_buffer: None,
            uniform_buffer,
            bind_group,
            pipeline,
            point_count: 0,
            point_size: 2.0,
            visible: true,
            color_mode: ColorMode::Elevation,
            bounds_min: [0.0; 3],
            bounds_max: [0.0; 3],
            center: [0.0; 3],
            has_rgb: false,
            has_intensity: false,
            cam_phi: 0.7,    // ~40 degrees
            cam_theta: 0.5,  // ~30 degrees elevation
            cam_radius: 1.0, // Will be set based on extent
        }
    }
    
    /// Handle mouse drag for orbit camera
    pub fn handle_mouse_drag(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.005;
        self.cam_phi += dx * sensitivity;
        self.cam_theta = (self.cam_theta - dy * sensitivity).clamp(0.1, 1.5); // Limit vertical angle
    }
    
    /// Handle scroll for zoom
    pub fn handle_scroll(&mut self, delta: f32) {
        let zoom_speed = 0.1;
        self.cam_radius *= 1.0 - delta * zoom_speed;
        self.cam_radius = self.cam_radius.clamp(0.1, 100.0); // Relative to extent
    }
    
    /// Handle keyboard input for camera control
    /// forward: W/S or Up/Down arrows (positive = tilt up)
    /// right: A/D or Left/Right arrows (positive = rotate right)
    /// up: Q/E (positive = zoom in)
    pub fn handle_keys(&mut self, forward: f32, right: f32, up: f32) {
        let rotate_speed = 0.02;
        let zoom_speed = 0.02;
        
        // Rotate camera (A/D or Left/Right)
        self.cam_phi += right * rotate_speed;
        
        // Tilt camera (W/S or Up/Down)
        self.cam_theta = (self.cam_theta + forward * rotate_speed).clamp(0.1, 1.5);
        
        // Zoom (Q/E)
        self.cam_radius *= 1.0 - up * zoom_speed;
        self.cam_radius = self.cam_radius.clamp(0.1, 100.0);
    }

    /// Load points from LAZ/LAS file
    pub fn load_from_file(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
        max_points: u64,
        color_mode: ColorMode,
    ) -> Result<(), String> {
        let load_result = load_laz_points(path, max_points as usize)?;
        let mut points = load_result.points;
        
        // Store data availability flags
        self.has_rgb = load_result.has_rgb;
        self.has_intensity = load_result.has_intensity;
        
        println!("[pointcloud] Data flags - has_rgb: {}, has_intensity: {}", 
                 self.has_rgb, self.has_intensity);
        
        if points.is_empty() {
            return Err("No points loaded".to_string());
        }

        // Compute bounds
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in &points {
            for i in 0..3 {
                min[i] = min[i].min(p.position[i]);
                max[i] = max[i].max(p.position[i]);
            }
        }
        
        // Compute center and normalize points to origin
        let center = [
            (min[0] + max[0]) / 2.0,
            (min[1] + max[1]) / 2.0,
            (min[2] + max[2]) / 2.0,
        ];
        
        // Shift all points so center is at origin (for camera to work with reasonable coords)
        for p in &mut points {
            p.position[0] -= center[0];
            p.position[1] -= center[1];
            p.position[2] -= center[2];
        }
        
        // Update bounds relative to origin
        self.bounds_min = [min[0] - center[0], min[1] - center[1], min[2] - center[2]];
        self.bounds_max = [max[0] - center[0], max[1] - center[1], max[2] - center[2]];
        self.center = [0.0, 0.0, 0.0]; // Points now centered at origin
        
        eprintln!("[pointcloud] Original center: ({:.1}, {:.1}, {:.1})", center[0], center[1], center[2]);
        eprintln!("[pointcloud] Extent: ({:.1}, {:.1}, {:.1})", 
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2]);

        self.points = points;
        self.point_count = self.points.len();
        self.color_mode = color_mode;

        // Upload to GPU
        self.upload_points(device, queue);

        Ok(())
    }

    fn upload_points(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.points.is_empty() {
            return;
        }

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PointCloud.InstanceBuffer"),
            size: (self.points.len() * std::mem::size_of::<PointInstance3D>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.points));
        self.instance_buffer = Some(buffer);
    }

    /// Render point cloud
    pub fn render<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        queue: &wgpu::Queue,
        view_proj: [[f32; 4]; 4],
        viewport_size: [f32; 2],
    ) {
        if !self.visible || self.point_count == 0 {
            return;
        }

        let Some(instance_buffer) = &self.instance_buffer else {
            return;
        };

        // Update uniforms
        let uniforms = PointCloudUniforms {
            view_proj,
            viewport_size,
            point_size: self.point_size,
            color_mode: self.color_mode as u32,
            has_rgb: if self.has_rgb { 1 } else { 0 },
            has_intensity: if self.has_intensity { 1 } else { 0 },
            _pad: [0, 0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
        render_pass.draw(0..4, 0..self.point_count as u32);
    }

    pub fn set_point_size(&mut self, size: f32) {
        self.point_size = size.max(0.5).min(50.0);
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn clear(&mut self) {
        self.points.clear();
        self.instance_buffer = None;
        self.point_count = 0;
    }
}

/// Result of loading points with data availability info
struct LoadResult {
    points: Vec<PointInstance3D>,
    has_rgb: bool,
    has_intensity: bool,
}

/// Load points from LAZ/LAS file using the las crate
fn load_laz_points(path: &str, max_points: usize) -> Result<LoadResult, String> {
    use las::{Read, Reader};
    
    eprintln!("[pointcloud] Opening file: {}", path);
    
    let mut reader = Reader::from_path(path)
        .map_err(|e| format!("Failed to open LAS/LAZ file: {}", e))?;
    
    let header = reader.header();
    let total_points = header.number_of_points() as usize;
    let bounds = header.bounds();
    
    eprintln!("[pointcloud] File has {} points", total_points);
    eprintln!("[pointcloud] Bounds: X({:.1}, {:.1}) Y({:.1}, {:.1}) Z({:.1}, {:.1})",
        bounds.min.x, bounds.max.x,
        bounds.min.y, bounds.max.y,
        bounds.min.z, bounds.max.z);
    
    let min_z = bounds.min.z;
    let max_z = bounds.max.z;
    let z_range = max_z - min_z;
    
    // Determine sampling stride if we have more points than max
    let stride = if total_points > max_points {
        total_points / max_points
    } else {
        1
    };
    
    let n_read = (total_points / stride).min(max_points);
    eprintln!("[pointcloud] Loading {} points (stride {})", n_read, stride);
    
    let mut points = Vec::with_capacity(n_read);
    let mut has_rgb = false;
    let mut intensity_min: u16 = u16::MAX;
    let mut intensity_max: u16 = 0;
    
    for (i, point_result) in reader.points().enumerate() {
        // Skip points based on stride for subsampling
        if stride > 1 && i % stride != 0 {
            continue;
        }
        
        if points.len() >= max_points {
            break;
        }
        
        let point = point_result.map_err(|e| format!("Error reading point: {}", e))?;
        
        // LAS: X=easting, Y=northing, Z=elevation
        // 3D (Y-up): X=easting, Y=elevation, Z=northing
        let px = point.x as f32;
        let py = point.z as f32;  // Z (elevation) becomes Y (up)
        let pz = point.y as f32;  // Y (northing) becomes Z (depth)
        
        // Normalize elevation for coloring
        let elevation_norm = if z_range > 0.0 {
            ((point.z - min_z) / z_range).clamp(0.0, 1.0) as f32
        } else {
            0.5
        };
        
        // Extract RGB if available (default to white)
        let rgb = if let Some(color) = point.color {
            has_rgb = true;
            [
                color.red as f32 / 65535.0,
                color.green as f32 / 65535.0,
                color.blue as f32 / 65535.0,
            ]
        } else {
            [1.0, 1.0, 1.0]
        };
        
        // Track intensity range for normalization
        intensity_min = intensity_min.min(point.intensity);
        intensity_max = intensity_max.max(point.intensity);
        
        // Store raw intensity - will normalize in second pass
        let intensity = point.intensity as f32;
        
        points.push(PointInstance3D {
            position: [px, py, pz],
            elevation_norm,
            rgb,
            intensity,
            size: 1.0,
            _pad: [0.0; 3],
        });
        
        if points.len() % 100000 == 0 {
            eprintln!("[pointcloud] Loaded {} points...", points.len());
        }
    }
    
    // Log data availability (use println for visibility)
    println!("[pointcloud] Loaded {} points total", points.len());
    println!("[pointcloud] Has RGB: {}", has_rgb);
    println!("[pointcloud] Intensity range: {} - {}", intensity_min, intensity_max);
    
    // Sample first point for debugging
    if let Some(p) = points.first() {
        println!("[pointcloud] Sample point - rgb: {:?}, intensity: {}", p.rgb, p.intensity);
    }
    
    // Normalize intensity values to 0-1 range based on actual data range
    let intensity_range = (intensity_max - intensity_min) as f32;
    if intensity_range > 0.0 {
        let intensity_min_f = intensity_min as f32;
        for p in points.iter_mut() {
            p.intensity = (p.intensity - intensity_min_f) / intensity_range;
        }
        println!("[pointcloud] Normalized intensity to 0-1 range");
    } else {
        // All same intensity - use 0.5
        for p in points.iter_mut() {
            p.intensity = 0.5;
        }
        println!("[pointcloud] All intensities same, using 0.5");
    }
    
    // Print final sample
    if let Some(p) = points.first() {
        println!("[pointcloud] Final sample - rgb: {:?}, intensity: {:.3}, elev: {:.3}", 
                 p.rgb, p.intensity, p.elevation_norm);
    }
    
    // Determine data availability
    let has_meaningful_intensity = intensity_range > 0.0;
    
    Ok(LoadResult {
        points,
        has_rgb,
        has_intensity: has_meaningful_intensity,
    })
}

/// Generate elevation-based color
fn elevation_color(t: f32) -> [f32; 4] {
    // Terrain colormap: blue -> green -> yellow -> brown -> white
    let colors = [
        [0.2, 0.4, 0.6, 1.0], // Low: blue-gray
        [0.3, 0.5, 0.2, 1.0], // Green
        [0.6, 0.6, 0.3, 1.0], // Yellow-green
        [0.5, 0.4, 0.3, 1.0], // Brown
        [0.9, 0.9, 0.9, 1.0], // High: white
    ];
    
    let n = colors.len() - 1;
    let idx = (t * n as f32).min(n as f32 - 0.001);
    let i = idx.floor() as usize;
    let frac = idx - i as f32;
    
    let c0 = colors[i];
    let c1 = colors[(i + 1).min(n)];
    
    [
        c0[0] + (c1[0] - c0[0]) * frac,
        c0[1] + (c1[1] - c0[1]) * frac,
        c0[2] + (c1[2] - c0[2]) * frac,
        1.0,
    ]
}

/// Point cloud shader
const POINTCLOUD_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    point_size: f32,
    color_mode: u32,  // 0=elevation, 1=rgb, 2=intensity
    has_rgb: u32,     // 1 if file has RGB data
    has_intensity: u32, // 1 if file has meaningful intensity
    _pad: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) elevation_norm: f32,
    @location(2) rgb: vec3<f32>,
    @location(3) intensity: f32,
    @location(4) size: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

// Elevation colormap: blue -> green -> yellow -> brown -> white
fn elevation_color(t: f32) -> vec3<f32> {
    // 5-stop terrain colormap
    let c0 = vec3<f32>(0.2, 0.4, 0.6);  // Low: blue-gray
    let c1 = vec3<f32>(0.3, 0.5, 0.2);  // Green
    let c2 = vec3<f32>(0.6, 0.6, 0.3);  // Yellow-green
    let c3 = vec3<f32>(0.5, 0.4, 0.3);  // Brown
    let c4 = vec3<f32>(0.9, 0.9, 0.9);  // High: white
    
    let t_scaled = clamp(t, 0.0, 1.0) * 4.0;
    
    if t_scaled < 1.0 {
        return mix(c0, c1, t_scaled);
    } else if t_scaled < 2.0 {
        return mix(c1, c2, t_scaled - 1.0);
    } else if t_scaled < 3.0 {
        return mix(c2, c3, t_scaled - 2.0);
    } else {
        return mix(c3, c4, t_scaled - 3.0);
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Quad vertices for point sprite (use switch for runtime indexing)
    var offset: vec2<f32>;
    switch vertex_index {
        case 0u: { offset = vec2<f32>(-1.0, -1.0); }
        case 1u: { offset = vec2<f32>(1.0, -1.0); }
        case 2u: { offset = vec2<f32>(-1.0, 1.0); }
        case 3u: { offset = vec2<f32>(1.0, 1.0); }
        default: { offset = vec2<f32>(0.0, 0.0); }
    }
    out.uv = offset * 0.5 + 0.5;
    
    // Transform point to clip space
    let clip_pos = uniforms.view_proj * vec4<f32>(instance.position, 1.0);
    
    // Calculate point size in clip space
    let size_px = uniforms.point_size * instance.size;
    let size_clip = vec2<f32>(
        size_px / uniforms.viewport_size.x * 2.0,
        size_px / uniforms.viewport_size.y * 2.0,
    );
    
    // Apply quad offset
    out.clip_position = clip_pos + vec4<f32>(offset * size_clip * clip_pos.w, 0.0, 0.0);
    
    // Compute color based on mode (with fallback to grayscale elevation if data unavailable)
    var color: vec3<f32>;
    switch uniforms.color_mode {
        case 1u: {
            // RGB mode - use stored RGB, fallback to tinted elevation if no RGB data
            if uniforms.has_rgb == 1u {
                color = instance.rgb;
            } else {
                // Fallback to reddish elevation to indicate missing RGB
                color = elevation_color(instance.elevation_norm) * vec3<f32>(1.0, 0.5, 0.5);
            }
        }
        case 2u: {
            // Intensity mode - grayscale, fallback to tinted elevation if no intensity data
            if uniforms.has_intensity == 1u {
                color = vec3<f32>(instance.intensity);
            } else {
                // Fallback to green-tinted elevation to indicate missing Intensity
                color = elevation_color(instance.elevation_norm) * vec3<f32>(0.5, 1.0, 0.5);
            }
        }
        default: {
            // Elevation mode (0 or fallback)
            color = elevation_color(instance.elevation_norm);
        }
    }
    out.color = vec4<f32>(color, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circular point with soft edge
    let uv = in.uv * 2.0 - 1.0;
    let dist = length(uv);
    
    if dist > 1.0 {
        discard;
    }
    
    // Soft edge
    let alpha = 1.0 - smoothstep(0.7, 1.0, dist);
    
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;
