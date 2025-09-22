# python/forge3d/viewer.pyi
# Typing stub for viewer MSAA controls.
# Exists to describe public API exposed by python/forge3d/viewer.py.
# RELEVANT FILES:python/forge3d/viewer.py,python/forge3d/__init__.pyi,tests/test_b1_msaa.py

def set_msaa(samples: int) -> int: ...
