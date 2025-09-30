"""Vector graphics API for forge3d.

This module provides high-level Python wrappers around the vector graphics
functionality, including polygons, lines, points, and graphs.
"""

import numpy as np
from typing import Optional, List, Union, Tuple
from ._forge3d import (
    add_polygons_py,
    add_lines_py, 
    add_points_py,
    add_graph_py,
    clear_vectors_py,
    get_vector_counts_py,
)

__all__ = [
    'add_polygons',
    'add_lines',
    'add_points',
    'add_graph',
    'clear_vectors',
    'get_vector_counts',
    'VectorScene',
]

def add_polygons(
    exterior_coords: np.ndarray,
    holes: Optional[List[np.ndarray]] = None,
    fill_color: Optional[Tuple[float, float, float, float]] = None,
    stroke_color: Optional[Tuple[float, float, float, float]] = None,
    stroke_width: float = 1.0
) -> List[int]:
    """Add polygons to the vector scene.
    
    Parameters
    ----------
    exterior_coords : np.ndarray
        Exterior ring coordinates as (N, 2) array of [x, y] positions.
        Must be C-contiguous and float64.
    holes : List[np.ndarray], optional
        List of hole ring coordinates, each as (M, 2) arrays.
    fill_color : tuple of float, optional
        RGBA fill color [0,1]. Default is (0.2, 0.4, 0.8, 1.0).
    stroke_color : tuple of float, optional
        RGBA stroke color [0,1]. Default is (0.0, 0.0, 0.0, 1.0).
    stroke_width : float
        Stroke width in world units. Default is 1.0.
        
    Returns
    -------
    List[int]
        List of polygon IDs for later reference.
        
    Raises
    ------
    ValueError
        If arrays have wrong shape or non-finite values.
    RuntimeError
        If array is not C-contiguous or validation fails.
    """
    # Validate exterior coordinates
    exterior_coords = np.asarray(exterior_coords, dtype=np.float64)
    if exterior_coords.ndim != 2 or exterior_coords.shape[1] != 2:
        raise ValueError(f"exterior_coords must have shape (N, 2), got {exterior_coords.shape}")
    if exterior_coords.shape[0] < 3:
        raise ValueError("Polygon exterior must have at least 3 vertices")
    
    # Ensure C-contiguous
    if not exterior_coords.flags.c_contiguous:
        exterior_coords = np.ascontiguousarray(exterior_coords)
    
    # Validate holes if provided
    validated_holes = None
    if holes is not None:
        validated_holes = []
        for i, hole in enumerate(holes):
            hole_array = np.asarray(hole, dtype=np.float64)
            if hole_array.ndim != 2 or hole_array.shape[1] != 2:
                raise ValueError(f"Hole {i} must have shape (M, 2), got {hole_array.shape}")
            if hole_array.shape[0] < 3:
                raise ValueError(f"Hole {i} must have at least 3 vertices")
                
            if not hole_array.flags.c_contiguous:
                hole_array = np.ascontiguousarray(hole_array)
            validated_holes.append(hole_array)
    
    # Validate colors
    if fill_color is not None:
        if len(fill_color) != 4:
            raise ValueError("fill_color must be (r, g, b, a) tuple")
        if any(c < 0 or c > 1 for c in fill_color):
            raise ValueError("fill_color components must be in [0, 1]")
    
    if stroke_color is not None:
        if len(stroke_color) != 4:
            raise ValueError("stroke_color must be (r, g, b, a) tuple")
        if any(c < 0 or c > 1 for c in stroke_color):
            raise ValueError("stroke_color components must be in [0, 1]")
            
    if stroke_width <= 0:
        raise ValueError("stroke_width must be positive")
    
    return add_polygons_py(
        exterior_coords,
        validated_holes,
        list(fill_color) if fill_color else None,
        list(stroke_color) if stroke_color else None,
        stroke_width
    )


def add_lines(
    path_coords: np.ndarray,
    stroke_color: Optional[Tuple[float, float, float, float]] = None,
    stroke_width: float = 1.0
) -> List[int]:
    """Add lines to the vector scene.
    
    Parameters
    ----------
    path_coords : np.ndarray
        Line path coordinates as (N, 2) array of [x, y] positions.
        Must be C-contiguous and float64.
    stroke_color : tuple of float, optional
        RGBA stroke color [0,1]. Default is (0.0, 0.0, 0.0, 1.0).
    stroke_width : float
        Stroke width in world units. Default is 1.0.
        
    Returns
    -------
    List[int]
        List of line IDs for later reference.
        
    Raises
    ------
    ValueError
        If arrays have wrong shape or non-finite values.
    RuntimeError
        If array is not C-contiguous or validation fails.
    """
    # Validate path coordinates
    path_coords = np.asarray(path_coords, dtype=np.float64)
    if path_coords.ndim != 2 or path_coords.shape[1] != 2:
        raise ValueError(f"path_coords must have shape (N, 2), got {path_coords.shape}")
    if path_coords.shape[0] < 2:
        raise ValueError("Line path must have at least 2 vertices")
    
    # Ensure C-contiguous
    if not path_coords.flags.c_contiguous:
        path_coords = np.ascontiguousarray(path_coords)
    
    # Validate color
    if stroke_color is not None:
        if len(stroke_color) != 4:
            raise ValueError("stroke_color must be (r, g, b, a) tuple")
        if any(c < 0 or c > 1 for c in stroke_color):
            raise ValueError("stroke_color components must be in [0, 1]")
            
    if stroke_width <= 0:
        raise ValueError("stroke_width must be positive")
    
    return add_lines_py(
        path_coords,
        list(stroke_color) if stroke_color else None,
        stroke_width
    )


def add_points(
    positions: np.ndarray,
    fill_color: Optional[Tuple[float, float, float, float]] = None,
    point_size: float = 4.0
) -> List[int]:
    """Add points to the vector scene.
    
    Parameters
    ----------
    positions : np.ndarray
        Point positions as (N, 2) array of [x, y] coordinates.
        Must be C-contiguous and float64.
    fill_color : tuple of float, optional
        RGBA fill color [0,1]. Default is (1.0, 0.0, 0.0, 1.0).
    point_size : float
        Point size in pixels. Default is 4.0.
        
    Returns
    -------
    List[int]
        List of point IDs for later reference.
        
    Raises
    ------
    ValueError
        If arrays have wrong shape or non-finite values.
    RuntimeError
        If array is not C-contiguous or validation fails.
    """
    # Validate positions
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {positions.shape}")
    if positions.shape[0] == 0:
        raise ValueError("Must provide at least one point")
    
    # Ensure C-contiguous
    if not positions.flags.c_contiguous:
        positions = np.ascontiguousarray(positions)
    
    # Validate color
    if fill_color is not None:
        if len(fill_color) != 4:
            raise ValueError("fill_color must be (r, g, b, a) tuple")
        if any(c < 0 or c > 1 for c in fill_color):
            raise ValueError("fill_color components must be in [0, 1]")
            
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    
    return add_points_py(
        positions,
        list(fill_color) if fill_color else None,
        point_size
    )


def add_graph(
    nodes: np.ndarray,
    edges: np.ndarray,
    node_fill_color: Optional[Tuple[float, float, float, float]] = None,
    node_size: float = 4.0,
    edge_stroke_color: Optional[Tuple[float, float, float, float]] = None,
    edge_width: float = 1.0
) -> int:
    """Add a graph to the vector scene.
    
    Parameters
    ----------
    nodes : np.ndarray
        Node positions as (N, 2) array of [x, y] coordinates.
        Must be C-contiguous and float64.
    edges : np.ndarray
        Edge connections as (M, 2) array of [from_node, to_node] indices.
        Must be C-contiguous and uint32.
    node_fill_color : tuple of float, optional
        RGBA node fill color [0,1]. Default is (1.0, 0.0, 0.0, 1.0).
    node_size : float
        Node size in pixels. Default is 4.0.
    edge_stroke_color : tuple of float, optional
        RGBA edge stroke color [0,1]. Default is (0.0, 0.0, 0.0, 1.0).
    edge_width : float
        Edge width in world units. Default is 1.0.
        
    Returns
    -------
    int
        Graph ID for later reference.
        
    Raises
    ------
    ValueError
        If arrays have wrong shape or non-finite values.
    RuntimeError
        If array is not C-contiguous or validation fails.
    """
    # Validate nodes
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (N, 2), got {nodes.shape}")
    if nodes.shape[0] == 0:
        raise ValueError("Graph must have at least one node")
    
    # Ensure C-contiguous
    if not nodes.flags.c_contiguous:
        nodes = np.ascontiguousarray(nodes)
    
    # Validate edges
    edges = np.asarray(edges, dtype=np.uint32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (M, 2), got {edges.shape}")
    
    # Ensure C-contiguous
    if not edges.flags.c_contiguous:
        edges = np.ascontiguousarray(edges)
    
    # Validate edge indices
    node_count = nodes.shape[0]
    if edges.size > 0:  # Only check if there are edges
        max_edge_index = edges.max()
        if max_edge_index >= node_count:
            raise ValueError(f"Edge index {max_edge_index} exceeds node count {node_count}")
    
    # Validate colors
    if node_fill_color is not None:
        if len(node_fill_color) != 4:
            raise ValueError("node_fill_color must be (r, g, b, a) tuple")
        if any(c < 0 or c > 1 for c in node_fill_color):
            raise ValueError("node_fill_color components must be in [0, 1]")
    
    if edge_stroke_color is not None:
        if len(edge_stroke_color) != 4:
            raise ValueError("edge_stroke_color must be (r, g, b, a) tuple")
        if any(c < 0 or c > 1 for c in edge_stroke_color):
            raise ValueError("edge_stroke_color components must be in [0, 1]")
            
    if node_size <= 0:
        raise ValueError("node_size must be positive")
    if edge_width <= 0:
        raise ValueError("edge_width must be positive")
    
    return add_graph_py(
        nodes,
        edges,
        list(node_fill_color) if node_fill_color else None,
        node_size,
        list(edge_stroke_color) if edge_stroke_color else None,
        edge_width
    )


def clear_vectors() -> None:
    """Clear all vector primitives from the scene.
    
    This removes all previously added polygons, lines, points, and graphs.
    """
    clear_vectors_py()


def get_vector_counts() -> Tuple[int, int, int, int]:
    """Get the count of vector primitives in the scene.
    
    Returns
    -------
    tuple of int
        (polygons, lines, points, graphs) counts.
    """
    return get_vector_counts_py()


class VectorScene:
    """High-level batched vector scene for OIT rendering and GPU picking.

    Collect points and polylines, then render using weighted blended OIT
    or generate a full R32Uint picking map using Python helpers that call
    into the native extension.
    """

    def __init__(self) -> None:
        # Points
        self._points: list[tuple[float, float]] = []
        self._point_rgba: list[tuple[float, float, float, float]] = []
        self._point_size: list[float] = []
        # Polylines
        self._polylines: list[list[tuple[float, float]]] = []
        self._polyline_rgba: list[tuple[float, float, float, float]] = []
        self._stroke_width: list[float] = []

        # Defaults
        self._default_point_rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
        self._default_point_size: float = 8.0
        self._default_polyline_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        self._default_stroke_width: float = 2.0

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._points.clear(); self._point_rgba.clear(); self._point_size.clear()
        self._polylines.clear(); self._polyline_rgba.clear(); self._stroke_width.clear()

    def add_point(self, x: float, y: float, rgba: tuple[float, float, float, float] | None = None, size: float | None = None) -> None:
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError("point coordinates must be finite")
        col = rgba if rgba is not None else self._default_point_rgba
        if len(col) != 4:
            raise ValueError("rgba must be (r,g,b,a)")
        if any((c < 0.0 or c > 1.0) for c in col):
            raise ValueError("rgba components must be in [0,1]")
        s = float(size) if size is not None else float(self._default_point_size)
        if s <= 0.0:
            raise ValueError("point size must be positive")
        self._points.append((float(x), float(y)))
        self._point_rgba.append((float(col[0]), float(col[1]), float(col[2]), float(col[3])))
        self._point_size.append(s)

    def add_polyline(self, path: list[tuple[float, float]] | tuple[tuple[float, float], ...], rgba: tuple[float, float, float, float] | None = None, width: float | None = None) -> None:
        if not isinstance(path, (list, tuple)) or len(path) < 2:
            raise ValueError("polyline path must be a sequence of at least 2 (x,y) points")
        p2: list[tuple[float, float]] = []
        for (x, y) in path:
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError("polyline coordinates must be finite")
            p2.append((float(x), float(y)))
        col = rgba if rgba is not None else self._default_polyline_rgba
        if len(col) != 4:
            raise ValueError("rgba must be (r,g,b,a)")
        if any((c < 0.0 or c > 1.0) for c in col):
            raise ValueError("rgba components must be in [0,1]")
        w = float(width) if width is not None else float(self._default_stroke_width)
        if w <= 0.0:
            raise ValueError("stroke width must be positive")
        self._polylines.append(p2)
        self._polyline_rgba.append((float(col[0]), float(col[1]), float(col[2]), float(col[3])))
        self._stroke_width.append(w)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_oit(self, width: int, height: int) -> np.ndarray:
        """Render collected vectors using weighted blended OIT to an RGBA image."""
        from . import vector_render_oit_py as _render
        return _render(
            int(width), int(height),
            points_xy=self._points or None,
            point_rgba=self._point_rgba or None,
            point_size=self._point_size or None,
            polylines=self._polylines or None,
            polyline_rgba=self._polyline_rgba or None,
            stroke_width=self._stroke_width or None,
        )

    def render_pick_map(self, width: int, height: int, base_pick_id: int = 1) -> np.ndarray:
        """Render full R32Uint picking map for the collected vectors."""
        from . import vector_render_pick_map_py as _pick
        return _pick(
            int(width), int(height),
            points_xy=self._points or None,
            polylines=self._polylines or None,
            base_pick_id=int(base_pick_id),
        )

    def render_oit_and_pick(self, width: int, height: int, base_pick_id: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Render combined OIT RGBA and full pick map in one call.

        Returns (rgba: np.ndarray(H,W,4) uint8, picks: np.ndarray(H,W) uint32)
        """
        from . import vector_render_oit_and_pick_py as _both
        return _both(
            int(width), int(height),
            points_xy=self._points or None,
            point_rgba=self._point_rgba or None,
            point_size=self._point_size or None,
            polylines=self._polylines or None,
            polyline_rgba=self._polyline_rgba or None,
            stroke_width=self._stroke_width or None,
            base_pick_id=int(base_pick_id),
        )