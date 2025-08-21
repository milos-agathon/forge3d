from pathlib import Path
import forge3d as f3d

def main():
    r = f3d.Renderer(256,256)
    r.render_triangle_png(Path("ex_triangle.png"))
    print("triangle.png written")

if __name__ == "__main__":
    main()