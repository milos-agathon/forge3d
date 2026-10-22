from pathlib import Path
import pytest
from forge3d.tiles3d import BoundingVolume, Tile, TileContent, Tileset, Tiles3dRenderer

@pytest.mark.parametrize("refine", ["ADD", "REPLACE"])
def test_depth_limit_retains_content(refine):
    bounds = BoundingVolume("sphere", [0, 0, 0, 1])
    child = Tile(bounds, 0, content=TileContent("child.pnts"))
    root = Tile(bounds, 1000, refine=refine, content=TileContent("root.pnts"), children=[child])
    tileset = Tileset(Path("."), "1.0", 1000, root)
    selected = Tiles3dRenderer(max_depth=0).get_visible_tiles(tileset, (0, 0, 10))
    assert [x.tile.content_uri() for x in selected] == ["root.pnts"]
