import sys, pathlib
from pathlib import Path

try:
    from vulkan_forge import Renderer
except Exception as e:
    print("IMPORT_FAIL:", e)
    sys.exit(2)

out = Path("triangle.png")
r = Renderer(512, 512)
arr = r.render_triangle_rgba()   # should return HxWx4 uint8
r.render_triangle_png(str(out))
print("shape:", a.shape, "dtype:", a.dtype)
assert a.shape == (512,512,4) and a.dtype == np.uint8
print("OK: A1.4 acceptance: shape/dtype correct")
if not out.exists():
    print("RENDER_FAIL: triangle.png was not created")
    sys.exit(3)
print("OK: triangle.png written, array shape:", getattr(arr, "shape", None))