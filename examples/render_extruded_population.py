#!/usr/bin/env python3
"""Render extruded 3D polygons colored by population using viridis colormap."""

from __future__ import annotations
import os
os.environ.setdefault("WGPU_BACKENDS", "METAL")
os.environ.setdefault("WGPU_BACKEND", "metal")

import argparse
import numpy as np
import geopandas as gpd
import forge3d as f3d
from forge3d.io import save_obj


def viridis_color(value: float) -> tuple[float, float, float]:
    """Map a normalized value [0,1] to viridis colormap RGB."""
    # Viridis control points (approximate)
    colors = np.array([
        [0.267004, 0.004874, 0.329415],  # Dark purple
        [0.282623, 0.140926, 0.457517],  # Purple
        [0.253935, 0.265254, 0.529983],  # Blue-purple
        [0.206756, 0.371758, 0.553117],  # Blue
        [0.163625, 0.471133, 0.558148],  # Cyan-blue
        [0.127568, 0.566949, 0.550556],  # Cyan
        [0.134692, 0.658636, 0.517649],  # Green-cyan
        [0.266941, 0.748751, 0.440573],  # Green
        [0.477504, 0.821444, 0.318195],  # Yellow-green
        [0.741388, 0.873449, 0.149561],  # Yellow
        [0.993248, 0.906157, 0.143936],  # Bright yellow
    ])
    
    # Interpolate
    idx = value * (len(colors) - 1)
    i0 = int(np.floor(idx))
    i1 = min(i0 + 1, len(colors) - 1)
    t = idx - i0
    
    color = colors[i0] * (1 - t) + colors[i1] * t
    return tuple(color)


def main():
    parser = argparse.ArgumentParser(description="Render extruded 3D population map with viridis colors")
    parser.add_argument("gpkg", help="Path to GeoPackage file")
    parser.add_argument("--layer", default="population", help="Layer name")
    parser.add_argument("--value-col", default="population", help="Column name for values")
    parser.add_argument("--output", default="reports/extruded_population.png", help="Output PNG path")
    parser.add_argument("--width", type=int, default=1200, help="Output width")
    parser.add_argument("--height", type=int, default=900, help="Output height")
    parser.add_argument("--max-height", type=float, default=100.0, help="Maximum extrusion height")
    parser.add_argument("--log-scale", action="store_true", help="Use log scale for height")
    args = parser.parse_args()

    # Load GeoPackage
    print(f"Loading {args.gpkg} layer {args.layer}...")
    gdf = gpd.read_file(args.gpkg, layer=args.layer)
    print(f"Loaded {len(gdf)} features")
    
    values = gdf[args.value_col].values
    print(f"Value range: [{values.min():.1f}, {values.max():.1f}]")
    
    # Normalize values for color mapping
    v_min, v_max = values.min(), values.max()
    if v_max > v_min:
        norm_values = (values - v_min) / (v_max - v_min)
    else:
        norm_values = np.ones_like(values) * 0.5
    
    # Compute heights
    if args.log_scale:
        # Log scale for height (handle zeros)
        log_vals = np.log1p(values)
        log_min, log_max = log_vals.min(), log_vals.max()
        if log_max > log_min:
            heights = args.max_height * (log_vals - log_min) / (log_max - log_min)
        else:
            heights = np.ones_like(values) * args.max_height * 0.1
    else:
        heights = args.max_height * norm_values
    
    # Add minimum height for visibility
    heights = np.maximum(heights, args.max_height * 0.01)
    
    print(f"Height range: [{heights.min():.2f}, {heights.max():.2f}]")
    
    # Get bounds for normalization
    min_x, min_y, max_x, max_y = gdf.total_bounds
    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    dx, dy = max_x - min_x, max_y - min_y
    scale = 100.0 / max(dx, dy)
    
    print("Extruding and coloring polygons...")
    all_vertices = []
    all_indices = []
    all_colors = []
    vertex_offset = 0
    
    for idx, (geom, height, norm_val) in enumerate(zip(gdf.geometry, heights, norm_values)):
        if geom is None or not hasattr(geom, 'exterior'):
            continue
        
        # Get exterior coordinates and normalize
        coords = np.asarray(geom.exterior.coords, dtype=np.float64)
        coords_norm = np.zeros_like(coords, dtype=np.float32)
        coords_norm[:, 0] = (coords[:, 0] - cx) * scale
        coords_norm[:, 1] = (coords[:, 1] - cy) * scale
        
        # Extrude polygon
        try:
            vertices_flat, indices, normals_flat, uvs_flat = f3d._forge3d.extrude_polygon_py(coords_norm, float(height))
            
            # Reshape flat arrays to proper dimensions
            vertices = vertices_flat.reshape(-1, 3)
            normals = normals_flat.reshape(-1, 3)
            uvs = uvs_flat.reshape(-1, 2)
            
            if len(vertices) > 0:
                all_vertices.append(vertices)
                all_indices.append(indices + vertex_offset)
                vertex_offset += len(vertices)
                
                # Store normals and UVs
                all_normals = [] if idx == 0 else all_normals
                all_uvs = [] if idx == 0 else all_uvs
                
                if idx == 0:
                    all_normals = []
                    all_uvs = []
                    all_norm_values = []
                
                all_normals.append(normals)
                all_uvs.append(uvs)
                
                # Store normalized value for coloring
                all_norm_values.append(norm_val)
        except Exception as e:
            print(f"  Skipped polygon {idx}: {e}")
            continue
    
    if not all_vertices:
        print("No polygons successfully extruded!")
        return
    
    # Merge all geometry
    vertices = np.vstack(all_vertices)
    indices = np.hstack(all_indices)
    normals = np.vstack(all_normals)
    uvs = np.vstack(all_uvs)
    
    print(f"Total vertices: {len(vertices)}, indices: {len(indices)}, triangles: {len(indices)//3}")
    
    # Compute vertex colors from population values
    # Assign colors to vertices based on which polygon they belong to
    vertex_colors = np.zeros((len(vertices), 3), dtype=np.float32)
    vertex_offset = 0
    for verts, norm_val in zip(all_vertices, all_norm_values):
        color = viridis_color(norm_val)
        vertex_colors[vertex_offset:vertex_offset + len(verts)] = color
        vertex_offset += len(verts)
    
    # Create mesh
    from forge3d.geometry import MeshBuffers
    mesh = MeshBuffers(
        positions=vertices,
        normals=normals,
        uvs=uvs,
        indices=indices,
    )
    
    # Save to OBJ for inspection
    obj_path = args.output.replace('.png', '.obj')
    try:
        save_obj(mesh, obj_path)
        print(f"Saved 3D mesh to {obj_path}")
    except Exception as e:
        print(f"OBJ export warning: {e}")
    
    # Create a simple top-down 2D visualization with viridis colors
    print(f"\nCreating 2D top-down visualization with viridis colors...")
    print(f"Note: The full 3D extruded mesh is in {obj_path}")
    print(f"You can view it in Blender, MeshLab, or other 3D viewers.")
    
    # Create rasterized view from above
    from PIL import Image
    img = np.zeros((args.height, args.width, 4), dtype=np.uint8)
    img[:, :, 3] = 255  # Set alpha to opaque
    
    # Project triangles to 2D and rasterize with viridis colors
    xy_min = vertices[:, :2].min(axis=0)
    xy_max = vertices[:, :2].max(axis=0)
    xy_range = xy_max - xy_min
    
    # Rasterize each triangle
    for i in range(0, len(indices), 3):
        # Get triangle vertices and colors
        tri_indices = indices[i:i+3]
        tri_verts = vertices[tri_indices]
        tri_colors = vertex_colors[tri_indices]
        
        # Project to image coordinates
        xy_norm = (tri_verts[:, :2] - xy_min) / xy_range
        px = (xy_norm[:, 0] * (args.width - 1)).astype(int)
        py = ((1.0 - xy_norm[:, 1]) * (args.height - 1)).astype(int)
        
        # Average color for this triangle
        color_avg = tri_colors.mean(axis=0)
        color_u8 = (color_avg * 255).astype(np.uint8)
        
        # Simple point rasterization (mark triangle vertices)
        for x, y in zip(px, py):
            if 0 <= x < args.width and 0 <= y < args.height:
                img[y, x, :3] = color_u8
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(img).save(args.output)
    print(f"Saved 2D visualization to {args.output}")


if __name__ == "__main__":
    main()
