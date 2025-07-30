from pathlib import Path
import numpy as np
from PIL import Image
from vulkan_forge import Renderer

def main():
    r = Renderer(width=512, height=512)
    print(r.info())
    arr = r.render_triangle_rgba()
    out = Path("triangle.png")
    try:
        Image.fromarray(arr, mode="RGBA").save(out)
    except Exception:
        r.render_triangle_png(str(out))
    print("Saved", out.resolve())

if __name__ == "__main__":
    main()
