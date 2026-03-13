import argparse
from pathlib import Path


def _import_forge3d():
    try:
        import forge3d as f3d

        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import

        ensure_repo_import()
        import forge3d as f3d

        return f3d


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a simple triangle PNG.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", default="triangle.png")
    args = parser.parse_args()

    f3d = _import_forge3d()
    renderer = f3d.Renderer(args.width, args.height)
    renderer.render_triangle_png(Path(args.output))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
