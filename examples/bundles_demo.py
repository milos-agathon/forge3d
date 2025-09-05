#!/usr/bin/env python3
"""
Render Bundles Demonstration

Demonstrates the render bundle system for GPU command optimization:
1. Instanced rendering (many objects, same geometry)
2. UI batch rendering (sprites, text, UI elements)
3. Particle system rendering (thousands of particles)
4. Mixed batch rendering (different objects in one bundle)
5. Performance comparison vs individual draw calls

Usage:
    python examples/bundles_demo.py --out out/bundles_demo.png
    python examples/bundles_demo.py --benchmark --out out/bundles_performance.png
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import logging
import time

# Add repository root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d as f3d
    import forge3d.bundles as bundles
except ImportError as e:
    print(f"ERROR: Could not import forge3d: {e}")
    print("Run: maturin develop --release")
    sys.exit(1)


def create_instanced_scene():
    """Create scene with many instances of the same object."""
    print("Creating instanced rendering scene...")
    
    # Create base geometry - simple cube
    cube_vertices = np.array([
        # Position    Normal      UV      Color
        # Front face
        [-1, -1,  1,  0, 0, 1,  0, 0,  1, 1, 1, 1],
        [ 1, -1,  1,  0, 0, 1,  1, 0,  1, 1, 1, 1],
        [ 1,  1,  1,  0, 0, 1,  1, 1,  1, 1, 1, 1],
        [-1,  1,  1,  0, 0, 1,  0, 1,  1, 1, 1, 1],
        
        # Back face
        [ 1, -1, -1,  0, 0, -1,  0, 0,  1, 1, 1, 1],
        [-1, -1, -1,  0, 0, -1,  1, 0,  1, 1, 1, 1],
        [-1,  1, -1,  0, 0, -1,  1, 1,  1, 1, 1, 1],
        [ 1,  1, -1,  0, 0, -1,  0, 1,  1, 1, 1, 1],
        
        # Right face
        [ 1, -1,  1,  1, 0, 0,  0, 0,  1, 1, 1, 1],
        [ 1, -1, -1,  1, 0, 0,  1, 0,  1, 1, 1, 1],
        [ 1,  1, -1,  1, 0, 0,  1, 1,  1, 1, 1, 1],
        [ 1,  1,  1,  1, 0, 0,  0, 1,  1, 1, 1, 1],
        
        # Left face
        [-1, -1, -1, -1, 0, 0,  0, 0,  1, 1, 1, 1],
        [-1, -1,  1, -1, 0, 0,  1, 0,  1, 1, 1, 1],
        [-1,  1,  1, -1, 0, 0,  1, 1,  1, 1, 1, 1],
        [-1,  1, -1, -1, 0, 0,  0, 1,  1, 1, 1, 1],
        
        # Top face
        [-1,  1,  1,  0, 1, 0,  0, 0,  1, 1, 1, 1],
        [ 1,  1,  1,  0, 1, 0,  1, 0,  1, 1, 1, 1],
        [ 1,  1, -1,  0, 1, 0,  1, 1,  1, 1, 1, 1],
        [-1,  1, -1,  0, 1, 0,  0, 1,  1, 1, 1, 1],
        
        # Bottom face
        [-1, -1, -1,  0, -1, 0,  0, 0,  1, 1, 1, 1],
        [ 1, -1, -1,  0, -1, 0,  1, 0,  1, 1, 1, 1],
        [ 1, -1,  1,  0, -1, 0,  1, 1,  1, 1, 1, 1],
        [-1, -1,  1,  0, -1, 0,  0, 1,  1, 1, 1, 1],
    ], dtype=np.float32)
    
    # Cube indices (2 triangles per face)
    cube_indices = np.array([
        # Front, Back, Right, Left, Top, Bottom
        0, 1, 2,    0, 2, 3,    # Front
        4, 5, 6,    4, 6, 7,    # Back
        8, 9, 10,   8, 10, 11,  # Right
        12, 13, 14, 12, 14, 15, # Left
        16, 17, 18, 16, 18, 19, # Top
        20, 21, 22, 20, 22, 23, # Bottom
    ], dtype=np.uint32)
    
    # Create instance transforms and colors
    num_instances = 500
    instance_transforms = np.zeros((num_instances, 4, 4), dtype=np.float32)
    instance_colors = np.zeros((num_instances, 4), dtype=np.float32)
    
    print(f"Generating {num_instances} cube instances...")
    
    for i in range(num_instances):
        # Random position
        x = np.random.uniform(-25, 25)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(-25, 25)
        
        # Random rotation
        angle = np.random.uniform(0, 2 * np.pi)
        axis_x = np.random.uniform(-1, 1)
        axis_y = np.random.uniform(-1, 1)
        axis_z = np.random.uniform(-1, 1)
        axis_length = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
        if axis_length > 0:
            axis_x /= axis_length
            axis_y /= axis_length
            axis_z /= axis_length
        
        # Random scale
        scale = np.random.uniform(0.3, 1.5)
        
        # Create transform matrix
        transform = np.eye(4, dtype=np.float32)
        
        # Apply scale
        transform[:3, :3] *= scale
        
        # Apply rotation (simplified - just around Y axis for demo)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float32)
        transform[:3, :3] = rotation @ (transform[:3, :3])
        
        # Apply translation
        transform[:3, 3] = [x, y, z]
        
        instance_transforms[i] = transform
        
        # Random color
        instance_colors[i] = [
            np.random.uniform(0.3, 1.0),  # R
            np.random.uniform(0.3, 1.0),  # G
            np.random.uniform(0.3, 1.0),  # B
            1.0                           # A
        ]
    
    # Create instanced bundle
    instanced_bundle = bundles.BundleBuilder.create_instanced_bundle(
        "cube_instances",
        {'vertices': cube_vertices, 'indices': cube_indices},
        instance_transforms,
        instance_colors
    )
    
    print(f"Created instanced bundle with {num_instances} instances")
    return instanced_bundle


def create_ui_scene():
    """Create UI scene with many UI elements."""
    print("Creating UI rendering scene...")
    
    # Create UI elements - buttons, panels, text backgrounds
    ui_elements = []
    
    # Main UI panels
    for i in range(5):
        ui_elements.append({
            'position': (50 + i * 160, 50),
            'size': (140, 200),
            'color': (0.2, 0.3, 0.5, 0.8),  # Semi-transparent blue panels
        })
    
    # Buttons on each panel
    for panel in range(5):
        for button in range(4):
            ui_elements.append({
                'position': (70 + panel * 160, 80 + button * 40),
                'size': (100, 30),
                'color': (0.1, 0.7, 0.3, 1.0),  # Green buttons
            })
    
    # Status indicators
    for i in range(20):
        ui_elements.append({
            'position': (100 + (i % 10) * 80, 320 + (i // 10) * 25),
            'size': (60, 20),
            'color': (
                1.0 if np.random.rand() > 0.5 else 0.2,  # Red or dim
                0.8 if np.random.rand() > 0.3 else 0.2,  # Green or dim
                0.1,                                      # Blue
                1.0                                       # Alpha
            ),
        })
    
    # Progress bars
    for i in range(8):
        # Background
        ui_elements.append({
            'position': (50, 380 + i * 30),
            'size': (200, 20),
            'color': (0.2, 0.2, 0.2, 0.8),  # Dark background
        })
        
        # Progress fill
        progress = np.random.uniform(0.1, 0.9)
        ui_elements.append({
            'position': (50, 380 + i * 30),
            'size': (200 * progress, 20),
            'color': (0.1, 0.8, 0.2, 1.0),  # Green fill
        })
    
    # Create UI bundle
    ui_bundle = bundles.BundleBuilder.create_ui_bundle("ui_elements", ui_elements)
    
    print(f"Created UI bundle with {len(ui_elements)} elements")
    return ui_bundle


def create_particle_scene():
    """Create particle system scene."""
    print("Creating particle system scene...")
    
    # Generate particle data
    num_particles = 2000
    
    # Particle data: position(3) + velocity(3) + size(1) + life(1) + color(4) + padding(1)
    particles = np.zeros((num_particles, 13), dtype=np.float32)
    
    # Initialize particles in various emitters
    particles_per_emitter = num_particles // 4
    
    # Emitter 1: Fountain effect
    start_idx = 0
    end_idx = particles_per_emitter
    particles[start_idx:end_idx, 0] = np.random.uniform(-2, 2, particles_per_emitter)  # X pos
    particles[start_idx:end_idx, 1] = 0  # Y pos (ground level)
    particles[start_idx:end_idx, 2] = np.random.uniform(-2, 2, particles_per_emitter)  # Z pos
    
    particles[start_idx:end_idx, 3] = np.random.uniform(-1, 1, particles_per_emitter)  # X vel
    particles[start_idx:end_idx, 4] = np.random.uniform(8, 15, particles_per_emitter)  # Y vel (upward)
    particles[start_idx:end_idx, 5] = np.random.uniform(-1, 1, particles_per_emitter)  # Z vel
    
    particles[start_idx:end_idx, 6] = np.random.uniform(0.1, 0.5, particles_per_emitter)  # Size
    particles[start_idx:end_idx, 7] = np.random.uniform(0.3, 1.0, particles_per_emitter)  # Life
    
    # Blue water-like particles
    particles[start_idx:end_idx, 8] = 0.2   # R
    particles[start_idx:end_idx, 9] = 0.5   # G
    particles[start_idx:end_idx, 10] = 1.0  # B
    particles[start_idx:end_idx, 11] = 0.8  # A
    
    # Emitter 2: Fire effect
    start_idx = particles_per_emitter
    end_idx = 2 * particles_per_emitter
    particles[start_idx:end_idx, 0] = np.random.uniform(-1, 1, particles_per_emitter)  # X pos
    particles[start_idx:end_idx, 1] = np.random.uniform(0, 2, particles_per_emitter)   # Y pos
    particles[start_idx:end_idx, 2] = np.random.uniform(8, 12, particles_per_emitter)  # Z pos
    
    particles[start_idx:end_idx, 3] = np.random.uniform(-0.5, 0.5, particles_per_emitter)  # X vel
    particles[start_idx:end_idx, 4] = np.random.uniform(2, 6, particles_per_emitter)       # Y vel
    particles[start_idx:end_idx, 5] = np.random.uniform(-0.5, 0.5, particles_per_emitter)  # Z vel
    
    particles[start_idx:end_idx, 6] = np.random.uniform(0.2, 0.8, particles_per_emitter)  # Size
    particles[start_idx:end_idx, 7] = np.random.uniform(0.1, 0.8, particles_per_emitter)  # Life
    
    # Red-orange fire particles
    particles[start_idx:end_idx, 8] = 1.0   # R
    particles[start_idx:end_idx, 9] = np.random.uniform(0.3, 0.7, particles_per_emitter)  # G
    particles[start_idx:end_idx, 10] = 0.1  # B
    particles[start_idx:end_idx, 11] = 0.9  # A
    
    # Emitter 3: Smoke effect
    start_idx = 2 * particles_per_emitter
    end_idx = 3 * particles_per_emitter
    particles[start_idx:end_idx, 0] = np.random.uniform(-3, 3, particles_per_emitter)   # X pos
    particles[start_idx:end_idx, 1] = np.random.uniform(5, 8, particles_per_emitter)    # Y pos
    particles[start_idx:end_idx, 2] = np.random.uniform(-15, -10, particles_per_emitter)  # Z pos
    
    particles[start_idx:end_idx, 3] = np.random.uniform(-1, 1, particles_per_emitter)   # X vel
    particles[start_idx:end_idx, 4] = np.random.uniform(1, 3, particles_per_emitter)    # Y vel
    particles[start_idx:end_idx, 5] = np.random.uniform(-1, 1, particles_per_emitter)   # Z vel
    
    particles[start_idx:end_idx, 6] = np.random.uniform(0.5, 2.0, particles_per_emitter)  # Size
    particles[start_idx:end_idx, 7] = np.random.uniform(0.2, 0.9, particles_per_emitter)  # Life
    
    # Gray smoke particles
    gray_val = np.random.uniform(0.3, 0.7, particles_per_emitter)
    particles[start_idx:end_idx, 8] = gray_val   # R
    particles[start_idx:end_idx, 9] = gray_val   # G
    particles[start_idx:end_idx, 10] = gray_val  # B
    particles[start_idx:end_idx, 11] = 0.6       # A
    
    # Emitter 4: Sparkle effect
    start_idx = 3 * particles_per_emitter
    end_idx = num_particles
    remaining = end_idx - start_idx
    
    # Random positions in sphere
    angles = np.random.uniform(0, 2*np.pi, remaining)
    elevation = np.random.uniform(-np.pi/3, np.pi/3, remaining)
    radius = np.random.uniform(1, 5, remaining)
    
    particles[start_idx:end_idx, 0] = radius * np.cos(elevation) * np.cos(angles)  # X
    particles[start_idx:end_idx, 1] = radius * np.sin(elevation) + 3              # Y
    particles[start_idx:end_idx, 2] = radius * np.cos(elevation) * np.sin(angles) # Z
    
    # Minimal velocities (floating effect)
    particles[start_idx:end_idx, 3] = np.random.uniform(-0.2, 0.2, remaining)  # X vel
    particles[start_idx:end_idx, 4] = np.random.uniform(-0.1, 0.3, remaining)  # Y vel
    particles[start_idx:end_idx, 5] = np.random.uniform(-0.2, 0.2, remaining)  # Z vel
    
    particles[start_idx:end_idx, 6] = np.random.uniform(0.05, 0.2, remaining)  # Size
    particles[start_idx:end_idx, 7] = np.random.uniform(0.5, 1.0, remaining)   # Life
    
    # Bright sparkle colors
    particles[start_idx:end_idx, 8] = 1.0    # R
    particles[start_idx:end_idx, 9] = 1.0    # G
    particles[start_idx:end_idx, 10] = np.random.uniform(0.7, 1.0, remaining)  # B
    particles[start_idx:end_idx, 11] = 1.0   # A
    
    # Create particle bundle
    particle_bundle = bundles.BundleBuilder.create_particle_bundle("particle_systems", particles)
    
    print(f"Created particle bundle with {num_particles} particles across 4 emitters")
    return particle_bundle


def create_mixed_batch_scene():
    """Create scene with different objects in one batch."""
    print("Creating mixed batch rendering scene...")
    
    objects = []
    
    # Different geometric objects
    # Cube
    cube_verts = np.array([
        [-1, -1, -1,  0, 0, -1,  0, 0],  # 8 vertices for cube
        [ 1, -1, -1,  0, 0, -1,  1, 0],
        [ 1,  1, -1,  0, 0, -1,  1, 1],
        [-1,  1, -1,  0, 0, -1,  0, 1],
        [-1, -1,  1,  0, 0,  1,  0, 0],
        [ 1, -1,  1,  0, 0,  1,  1, 0],
        [ 1,  1,  1,  0, 0,  1,  1, 1],
        [-1,  1,  1,  0, 0,  1,  0, 1],
    ], dtype=np.float32)
    
    cube_indices = np.array([
        0, 1, 2, 0, 2, 3,  # Back face
        4, 6, 5, 4, 7, 6,  # Front face
        0, 4, 7, 0, 7, 3,  # Left face
        1, 2, 6, 1, 6, 5,  # Right face
        3, 2, 6, 3, 6, 7,  # Top face
        0, 1, 5, 0, 5, 4,  # Bottom face
    ], dtype=np.uint32)
    
    # Add cubes at different positions
    for i in range(10):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [i * 3 - 15, 0, -5]
        
        objects.append({
            'vertices': cube_verts,
            'indices': cube_indices,
            'transform': transform,
            'material': [0.0, 0.3, 0.0, 0.0]  # Plastic-like material
        })
    
    # Pyramid/tetrahedron
    pyramid_verts = np.array([
        [ 0,  2,  0,  0, 1, 0,  0.5, 0],  # Top vertex
        [-1, -1,  1, -1, 0, 1,  0, 1],    # Base vertices
        [ 1, -1,  1,  1, 0, 1,  1, 1],
        [ 0, -1, -1,  0, 0, -1,  0.5, 1],
    ], dtype=np.float32)
    
    pyramid_indices = np.array([
        0, 1, 2,  # Front face
        0, 2, 3,  # Right face
        0, 3, 1,  # Left face
        1, 3, 2,  # Base
    ], dtype=np.uint32)
    
    # Add pyramids
    for i in range(5):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [i * 4 - 8, 3, 5]
        
        objects.append({
            'vertices': pyramid_verts,
            'indices': pyramid_indices,
            'transform': transform,
            'material': [1.0, 0.1, 0.1, 0.0]  # Metallic material
        })
    
    # Plane/quad
    plane_verts = np.array([
        [-2, 0, -2,  0, 1, 0,  0, 0],
        [ 2, 0, -2,  0, 1, 0,  1, 0],
        [ 2, 0,  2,  0, 1, 0,  1, 1],
        [-2, 0,  2,  0, 1, 0,  0, 1],
    ], dtype=np.float32)
    
    plane_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    
    # Ground plane
    transform = np.eye(4, dtype=np.float32)
    transform[1, 3] = -3  # Lower it
    transform[:3, :3] *= 5  # Scale it up
    
    objects.append({
        'vertices': plane_verts,
        'indices': plane_indices,
        'transform': transform,
        'material': [0.0, 0.8, 0.0, 0.0]  # Rough surface
    })
    
    # Create batch bundle
    batch_bundle = bundles.BundleBuilder.create_batch_bundle("mixed_objects", objects)
    
    print(f"Created batch bundle with {len(objects)} different objects")
    return batch_bundle


def render_bundle_scene(bundle, output_path, width=800, height=600):
    """Render a bundle to an image."""
    print(f"Rendering bundle '{bundle.name}' ({width}x{height})...")
    
    # Create synthetic render
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if bundle.bundle_type == bundles.BundleType.INSTANCED:
        # Render instances as scattered dots
        if bundle.instance_data:
            instance_count = len(bundle.instance_data[0])
            for i in range(min(instance_count, 200)):  # Limit for visibility
                # Extract position from transform (simplified)
                if len(bundle.instance_data[0][i]) >= 20:
                    x_world = bundle.instance_data[0][i][3]   # Transform[0][3]
                    z_world = bundle.instance_data[0][i][11]  # Transform[2][3]
                    
                    # Project to screen
                    x_screen = int((x_world + 25) / 50 * width)
                    y_screen = int((z_world + 25) / 50 * height)
                    
                    if 0 <= x_screen < width and 0 <= y_screen < height:
                        # Instance colors
                        color_r = int(bundle.instance_data[0][i][16] * 255)
                        color_g = int(bundle.instance_data[0][i][17] * 255)
                        color_b = int(bundle.instance_data[0][i][18] * 255)
                        
                        # Draw small square for each instance
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                px = x_screen + dx
                                py = y_screen + dy
                                if 0 <= px < width and 0 <= py < height:
                                    image[py, px] = [color_r, color_g, color_b]
        
    elif bundle.bundle_type == bundles.BundleType.UI:
        # Render UI elements as rectangles
        for y in range(height):
            for x in range(width):
                # Create UI-like pattern
                if (x % 160) < 140 and (y % 50) < 40:
                    if (x % 160) < 20 or (x % 160) > 120:
                        image[y, x] = [30, 60, 120]  # Panel borders
                    else:
                        button_row = y // 40
                        if button_row % 2 == 0:
                            image[y, x] = [20, 150, 60]  # Button color
                        else:
                            image[y, x] = [40, 40, 40]   # Background
                else:
                    image[y, x] = [10, 10, 10]  # Dark background
    
    elif bundle.bundle_type == bundles.BundleType.PARTICLES:
        # Render particles as points with falloff
        if bundle.vertex_data:
            particles = bundle.vertex_data[0]
            for i in range(min(len(particles), 500)):  # Limit for performance
                if len(particles[i]) >= 8:
                    x_world = particles[i][0]
                    y_world = particles[i][1]
                    z_world = particles[i][2]
                    life = particles[i][7] if len(particles[i]) > 7 else 1.0
                    
                    if life > 0:
                        # Project to screen (simple orthographic)
                        x_screen = int((x_world + 15) / 30 * width)
                        y_screen = int(height - (y_world + 5) / 15 * height)
                        
                        if 0 <= x_screen < width and 0 <= y_screen < height:
                            # Particle color
                            if len(particles[i]) >= 12:
                                color_r = int(particles[i][8] * 255 * life)
                                color_g = int(particles[i][9] * 255 * life)
                                color_b = int(particles[i][10] * 255 * life)
                            else:
                                brightness = int(255 * life)
                                color_r = color_g = color_b = brightness
                            
                            # Draw particle with falloff
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    px = x_screen + dx
                                    py = y_screen + dy
                                    if 0 <= px < width and 0 <= py < height:
                                        falloff = 0.5 if abs(dx) + abs(dy) > 0 else 1.0
                                        image[py, px] = [
                                            min(255, int(image[py, px][0] + color_r * falloff)),
                                            min(255, int(image[py, px][1] + color_g * falloff)),
                                            min(255, int(image[py, px][2] + color_b * falloff))
                                        ]
    
    elif bundle.bundle_type == bundles.BundleType.BATCH:
        # Render mixed objects as shapes
        center_y = height // 2
        
        # Draw cubes as squares
        for i in range(10):
            x_center = int((i * 3 - 15 + 20) / 40 * width)
            y_center = int(center_y - 50)
            
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    px = x_center + dx
                    py = y_center + dy
                    if 0 <= px < width and 0 <= py < height:
                        image[py, px] = [100, 150, 200]  # Blue cubes
        
        # Draw pyramids as triangles
        for i in range(5):
            x_center = int((i * 4 - 8 + 20) / 40 * width)
            y_center = int(center_y + 50)
            
            for dy in range(-10, 11):
                for dx in range(-abs(dy), abs(dy) + 1):
                    px = x_center + dx
                    py = y_center + dy
                    if 0 <= px < width and 0 <= py < height:
                        image[py, px] = [200, 100, 50]  # Orange pyramids
        
        # Ground plane
        ground_y = int(center_y + 100)
        if 0 <= ground_y < height:
            for x in range(width):
                for dy in range(5):
                    if ground_y + dy < height:
                        image[ground_y + dy, x] = [50, 80, 40]  # Green ground
    
    else:
        # Default pattern
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    int(128 + 127 * np.sin(x * 0.02)),
                    int(128 + 127 * np.sin(y * 0.02)),
                    int(128 + 127 * np.sin((x + y) * 0.01))
                ]
    
    # Save image
    save_image(image, output_path)
    return image


def benchmark_bundle_performance():
    """Benchmark different bundle types and configurations."""
    print("\n=== Bundle Performance Benchmark ===")
    
    manager = bundles.BundleManager()
    results = {}
    
    # Test different bundle types
    bundle_configs = [
        ("instanced_100", lambda: create_simple_instanced_bundle(100)),
        ("instanced_500", lambda: create_simple_instanced_bundle(500)),
        ("instanced_1000", lambda: create_simple_instanced_bundle(1000)),
        ("ui_50", lambda: create_simple_ui_bundle(50)),
        ("ui_200", lambda: create_simple_ui_bundle(200)),
        ("particles_1000", lambda: create_simple_particle_bundle(1000)),
        ("particles_5000", lambda: create_simple_particle_bundle(5000)),
    ]
    
    for config_name, bundle_creator in bundle_configs:
        print(f"\nBenchmarking {config_name}...")
        
        # Create and compile bundle
        start_time = time.time()
        bundle = bundle_creator()
        bundle.compile()
        compile_time = time.time() - start_time
        
        # Add to manager
        manager.add_bundle(bundle)
        
        # Execute multiple times to get average
        execute_times = []
        for i in range(10):
            start_time = time.time()
            manager.execute_bundle(bundle.name)
            execute_time = time.time() - start_time
            execute_times.append(execute_time * 1000)  # Convert to ms
        
        avg_execute_time = sum(execute_times) / len(execute_times)
        
        # Get validation results
        validation = bundles.validate_bundle_performance(bundle)
        
        results[config_name] = {
            'bundle': bundle,
            'compile_time_ms': compile_time * 1000,
            'avg_execute_time_ms': avg_execute_time,
            'validation': validation,
            'stats': bundle.get_stats(),
        }
        
        print(f"  Compile time: {compile_time * 1000:.2f}ms")
        print(f"  Avg execute time: {avg_execute_time:.3f}ms")
        print(f"  Memory usage: {bundle.stats.memory_usage // 1024}KB")
        print(f"  Draw calls: {bundle.stats.draw_call_count}")
        print(f"  Vertices: {bundle.stats.total_vertices}")
        print(f"  Efficiency: {'GOOD' if validation['efficient'] else 'POOR'}")
    
    # Generate performance comparison
    print("\n=== Performance Summary ===")
    for config_name, result in results.items():
        efficiency_score = 1.0 - (result['avg_execute_time_ms'] / 10.0)  # Normalized score
        memory_efficiency = 1.0 - min(result['stats'].memory_usage / (10 * 1024 * 1024), 1.0)
        
        overall_score = (efficiency_score + memory_efficiency) / 2
        
        print(f"{config_name:15s}: "
              f"Execute={result['avg_execute_time_ms']:6.2f}ms, "
              f"Memory={result['stats'].memory_usage // 1024:4d}KB, "
              f"Score={overall_score:5.2f}")
    
    return results


def create_simple_instanced_bundle(instance_count):
    """Create simple instanced bundle for benchmarking."""
    # Simple quad geometry
    quad_verts = np.array([
        [-1, -1, 0,  0, 0, 1,  0, 0],
        [ 1, -1, 0,  0, 0, 1,  1, 0],
        [ 1,  1, 0,  0, 0, 1,  1, 1],
        [-1,  1, 0,  0, 0, 1,  0, 1],
    ], dtype=np.float32)
    
    quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    
    # Generate instance transforms
    transforms = np.zeros((instance_count, 4, 4), dtype=np.float32)
    colors = np.ones((instance_count, 4), dtype=np.float32)
    
    for i in range(instance_count):
        transforms[i] = np.eye(4)
        transforms[i][:3, 3] = [
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ]
    
    return bundles.BundleBuilder.create_instanced_bundle(
        f"benchmark_instanced_{instance_count}",
        {'vertices': quad_verts, 'indices': quad_indices},
        transforms,
        colors
    )


def create_simple_ui_bundle(element_count):
    """Create simple UI bundle for benchmarking."""
    elements = []
    for i in range(element_count):
        elements.append({
            'position': (i * 20, i * 15),
            'size': (18, 12),
            'color': (0.5, 0.5, 0.5, 1.0)
        })
    
    return bundles.BundleBuilder.create_ui_bundle(f"benchmark_ui_{element_count}", elements)


def create_simple_particle_bundle(particle_count):
    """Create simple particle bundle for benchmarking."""
    particles = np.random.rand(particle_count, 13).astype(np.float32)
    particles[:, :3] *= 20  # Position
    particles[:, 3:6] *= 2  # Velocity
    particles[:, 6] = 0.5   # Size
    particles[:, 7] = 1.0   # Life
    
    return bundles.BundleBuilder.create_particle_bundle(f"benchmark_particles_{particle_count}", particles)


def save_image(image, output_path):
    """Save image to PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        f3d.numpy_to_png(str(output_path), image)
        print(f"Saved bundle demo: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save PNG: {e}")
        # Fallback to numpy save
        np.save(str(output_path.with_suffix('.npy')), image)
        print(f"Saved as numpy array: {output_path.with_suffix('.npy')}")


def main():
    parser = argparse.ArgumentParser(description="Render bundles demonstration")
    parser.add_argument("--out", type=str, default="out/bundles_demo.png", 
                       help="Output file path")
    parser.add_argument("--type", type=str, default="all",
                       choices=["all", "instanced", "ui", "particles", "batch"],
                       help="Bundle type to demonstrate")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run performance benchmarks")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=600, help="Image height")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print("=== Render Bundles Demo ===")
    print(f"Bundle type: {args.type}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Output: {args.out}")
    
    # Check feature availability
    if not bundles.has_bundles_support():
        print("ERROR: Render bundles not available")
        return 1
    
    try:
        if args.benchmark:
            # Run performance benchmark
            benchmark_results = benchmark_bundle_performance()
            
            # Create performance visualization (placeholder)
            perf_image = np.zeros((600, 800, 3), dtype=np.uint8)
            for i, (name, result) in enumerate(benchmark_results.items()):
                y = 50 + i * 50
                bar_width = int(result['avg_execute_time_ms'] * 10)
                
                for x in range(min(bar_width, 700)):
                    if y < 600:
                        color_intensity = int(255 * (1.0 - x / 700))
                        perf_image[y:y+20, x] = [color_intensity, 100, 50]
                
                # Label (simple text representation)
                label = name[:10]
                for char_i, char in enumerate(label):
                    char_x = 10 + char_i * 8
                    if char_x < 800 and y + 30 < 600:
                        # Simple character pattern
                        for dy in range(8):
                            for dx in range(6):
                                if (dx + dy + ord(char)) % 4 == 0:
                                    perf_image[y + 30 + dy, char_x + dx] = [255, 255, 255]
            
            save_image(perf_image, args.out)
            print("\nBenchmark complete! Check output for performance visualization.")
            
        else:
            # Create demonstration scenes
            test_bundles = {}
            
            if args.type == "all" or args.type == "instanced":
                test_bundles["instanced"] = create_instanced_scene()
            
            if args.type == "all" or args.type == "ui":
                test_bundles["ui"] = create_ui_scene()
            
            if args.type == "all" or args.type == "particles":
                test_bundles["particles"] = create_particle_scene()
            
            if args.type == "all" or args.type == "batch":
                test_bundles["batch"] = create_mixed_batch_scene()
            
            # Compile all bundles
            print("\nCompiling bundles...")
            for name, bundle in test_bundles.items():
                start_time = time.time()
                bundle.compile()
                compile_time = time.time() - start_time
                
                print(f"  {name}: {compile_time*1000:.2f}ms compile time")
                print(f"    Stats: {bundle.get_stats()}")
                
                # Validate performance
                validation = bundles.validate_bundle_performance(bundle)
                if validation['warnings']:
                    print(f"    Warnings: {', '.join(validation['warnings'])}")
                if validation['recommendations']:
                    print(f"    Recommendations: {', '.join(validation['recommendations'])}")
            
            # Render output
            if args.type != "all":
                # Single bundle render
                bundle = list(test_bundles.values())[0]
                render_bundle_scene(bundle, args.out, args.width, args.height)
            else:
                # Combined render
                print("\nCreating combined bundle demonstration...")
                combined_image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                
                # Render each type in quadrants
                if "instanced" in test_bundles:
                    inst_img = render_bundle_scene(test_bundles["instanced"], 
                                                 "temp_instanced.png", 
                                                 args.width//2, args.height//2)
                    combined_image[:args.height//2, :args.width//2] = inst_img
                
                if "ui" in test_bundles:
                    ui_img = render_bundle_scene(test_bundles["ui"], 
                                               "temp_ui.png", 
                                               args.width//2, args.height//2)
                    combined_image[:args.height//2, args.width//2:] = ui_img
                
                if "particles" in test_bundles:
                    part_img = render_bundle_scene(test_bundles["particles"], 
                                                 "temp_particles.png", 
                                                 args.width//2, args.height//2)
                    combined_image[args.height//2:, :args.width//2] = part_img
                
                if "batch" in test_bundles:
                    batch_img = render_bundle_scene(test_bundles["batch"], 
                                                  "temp_batch.png", 
                                                  args.width//2, args.height//2)
                    combined_image[args.height//2:, args.width//2:] = batch_img
                
                save_image(combined_image, args.out)
        
        # Summary
        print("\n=== Render Bundles Demo Complete ===")
        if not args.benchmark:
            total_stats = {"memory": 0, "draws": 0, "vertices": 0}
            for bundle in test_bundles.values():
                stats = bundle.get_stats()
                total_stats["memory"] += stats.memory_usage
                total_stats["draws"] += stats.draw_call_count
                total_stats["vertices"] += stats.total_vertices
            
            print(f"Total memory usage: {total_stats['memory'] // 1024}KB")
            print(f"Total draw calls: {total_stats['draws']}")
            print(f"Total vertices: {total_stats['vertices']}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())