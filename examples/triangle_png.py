from pathlib import Path
import vulkan_forge as vf

def main():
    r = vf.Renderer(256,256)
    r.render_triangle_png(Path("ex_triangle.png"))
    print("triangle.png written")

if __name__ == "__main__":
    main()