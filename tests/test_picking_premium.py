
import pytest
import sys
from pathlib import Path

# Ensure we can import forge3d
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import forge3d
    from forge3d import (
        RichPickResult, HighlightStyle, LassoState, HeightfieldHit, 
        SelectionStyle, PickResult
    )
except ImportError:
    pytest.skip("forge3d module not available", allow_module_level=True)

def test_rich_pick_result():
    """Test RichPickResult creation and attributes."""
    r = RichPickResult(42, "layer_name")
    assert r.feature_id == 42
    assert r.layer_name == "layer_name"
    assert r.world_pos == (0.0, 0.0, 0.0)
    assert r.attributes == {}
    
    # Check repr
    assert "RichPickResult" in repr(r)
    assert "feature_id=42" in repr(r)

def test_highlight_style():
    """Test HighlightStyle creation and presets."""
    # Default
    h = HighlightStyle()
    assert h.effect == "color_tint"
    
    # Custom
    h2 = HighlightStyle((1.0, 0.0, 0.0, 1.0), "glow", 2.0, 0.8, 10.0, 1.0)
    assert h2.color == (1.0, 0.0, 0.0, 1.0)
    assert h2.effect == "glow"
    assert h2.glow_intensity == pytest.approx(0.8)
    
    # Static methods
    outline = HighlightStyle.outline((0.0, 1.0, 0.0, 1.0), 3.0)
    assert outline.effect == "outline"
    assert outline.outline_width == pytest.approx(3.0)
    
    glow = HighlightStyle.glow((0.0, 0.0, 1.0, 1.0), 0.5, 5.0)
    assert glow.effect == "glow"
    assert glow.glow_intensity == pytest.approx(0.5)

def test_lasso_state():
    """Test LassoState class."""
    s = LassoState.inactive()
    assert s.is_inactive()
    assert not s.is_drawing()
    
    s = LassoState.drawing()
    assert s.is_drawing()
    
    s = LassoState.complete()
    assert s.is_complete()
    
    assert "LassoState" in repr(s)

def test_heightfield_hit():
    """Test HeightfieldHit creation."""
    # Since we can't easily construct this from Python (it's mainly a return type),
    # we just check if the class exists and is importable, which we did.
    pass

def test_selection_style_plan2():
    """Test Plan 2 SelectionStyle (backward compatibility)."""
    s = SelectionStyle((1.0, 1.0, 1.0, 1.0))
    assert s.color == (1.0, 1.0, 1.0, 1.0)
    assert not s.outline
    
    s2 = SelectionStyle((1.0, 0.0, 0.0, 1.0), outline=True, outline_width=2.0)
    assert s2.outline
    assert s2.outline_width == 2.0

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_rich_pick_result()
        test_highlight_style()
        test_lasso_state()
        test_selection_style_plan2()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
