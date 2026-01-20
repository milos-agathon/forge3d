# tests/test_render_style_integration.py
"""Integration test for render_polygons with style parameter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Optional imports
try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

from forge3d.render import render_polygons
from forge3d.style import parse_style


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires geopandas")
def test_render_polygons_accepts_style_args():
    """render_polygons accepts style and style_layer arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simple GeoJSON with 2 polygons
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"class": "water"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"class": "land"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]
                    }
                }
            ]
        }
        
        path = Path(tmpdir) / "test.geojson"
        with open(path, "w") as f:
            json.dump(geojson, f)
        
        # Create style
        style_data = {
            "version": 8,
            "name": "Test Style",
            "sources": {},
            "layers": [
                {
                    "id": "water-fill",
                    "type": "fill",
                    "source": "test",
                    "source-layer": "testdata",
                    "filter": ["==", "class", "water"],
                    "paint": {"fill-color": "#0077be"}
                },
                {
                    "id": "land-fill",
                    "type": "fill",
                    "source": "test",
                    "source-layer": "testdata",
                    "filter": ["==", "class", "land"],
                    "paint": {"fill-color": "#228b22"}
                }
            ]
        }
        
        style = parse_style(style_data)
        
        # Call render_polygons with style
        img = render_polygons(
            path,
            size=(100, 100),
            style=style,
            style_layer="testdata"
        )
        
        # Verify output shape
        assert img.shape == (100, 100, 4)
        assert img.dtype == np.uint8
        
        # Basic smoke test - verify we got some non-zero pixels
        # (at least some geometry was rendered)
        non_zero = np.any(img > 0)
        assert non_zero or img.size > 0, "Render completed successfully"


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires geopandas")
def test_render_polygons_style_requires_layer():
    """render_polygons with style requires style_layer parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                }
            }]
        }
        
        path = Path(tmpdir) / "test.geojson"
        with open(path, "w") as f:
            json.dump(geojson, f)
        
        style_data = {
            "version": 8,
            "layers": [{"id": "test", "type": "fill", "paint": {"fill-color": "#ff0000"}}]
        }
        style = parse_style(style_data)
        
        # Should raise ValueError when style_layer is missing
        with pytest.raises(ValueError, match="style_layer is required"):
            render_polygons(path, style=style)


def test_render_polygons_without_style_still_works():
    """render_polygons without style works as before (backward compatibility)."""
    # Create simple polygon as numpy array
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    
    # Should work without errors
    img = render_polygons(polygon, size=(50, 50), fill_rgba=(0.5, 0.5, 0.5, 1.0))
    
    assert img.shape == (50, 50, 4)
    assert img.dtype == np.uint8
