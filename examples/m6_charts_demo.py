# examples/m6_charts_demo.py
# Workstream M6: Charts (Seaborn/Plotly) Demo
# - Render a Plotly figure (via kaleido) or a Seaborn/Matplotlib chart to RGBA and save PNG

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forge3d.adapters import (
    render_chart_to_rgba,
    is_plotly_available,
    is_seaborn_available,
)
from forge3d.helpers.offscreen import save_png_deterministic
from _m_3d_preview import (
    render_meshes_preview,
    make_displaced_plane_from_heightmap,
)
try:
    from forge3d.io import save_obj as _save_obj
except Exception:
    _save_obj = None


def build_plotly_fig():
    import plotly.graph_objects as go
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
    fig.update_layout(title="M6 Plotly → RGBA Demo")
    return fig


def build_seaborn_chart():
    import seaborn as sns
    import matplotlib.pyplot as plt
    tips = sns.load_dataset('tips')
    ax = sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time')
    ax.set_title("M6 Seaborn → RGBA Demo")
    return ax


def main() -> int:
    parser = argparse.ArgumentParser(description="M6 Charts → RGBA + 3D preview demo")
    parser.add_argument("--out", type=Path, default=Path("reports/m6_charts.png"))
    parser.add_argument("--backend", type=str, default="plotly", choices=["plotly", "seaborn"]) 
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=150)
    # 3D options
    parser.add_argument("--make-3d", action="store_true", default=True, help="Displace a plane from luminance and save 3D preview")
    parser.add_argument("--height-scale", type=float, default=12.0, help="Displacement scale for heightmap")
    parser.add_argument("--plane-res", type=int, nargs=2, default=(192, 192), metavar=("W","H"), help="Plane resolution for displacement")
    parser.add_argument("--preview-size", type=int, nargs=2, default=(800, 600), metavar=("W","H"), help="3D preview image size")
    parser.add_argument("--preview-dpi", type=int, default=150, help="3D preview DPI")
    parser.add_argument("--save-obj", action="store_true", help="Also save the displaced mesh as OBJ (requires native)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    obj = None
    if args.backend == "plotly":
        if not is_plotly_available():
            print("Plotly/kaleido not available. pip install plotly kaleido")
            return 0
        obj = build_plotly_fig()
        rgba = render_chart_to_rgba(obj, width=args.width, height=args.height, scale=1.0)
    else:
        if not is_seaborn_available():
            print("Seaborn/Matplotlib not available. pip install seaborn matplotlib")
            return 0
        obj = build_seaborn_chart()
        rgba = render_chart_to_rgba(obj, dpi=args.dpi)

    save_png_deterministic(str(args.out), rgba)
    print(f"Wrote {args.out}")

    # Optional 3D displaced preview
    if args.make_3d and rgba is not None:
        try:
            arr = rgba.astype(np.float32)
            if arr.dtype == np.uint8:
                arr = arr / 255.0
            hm = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            mesh = make_displaced_plane_from_heightmap(
                hm, scale=float(args.height_scale), resolution=(int(args.plane_res[0]), int(args.plane_res[1]))
            )
            if mesh is not None:
                if args.save_obj and _save_obj is not None:
                    obj_path = args.out.with_suffix("")
                    obj_path = obj_path.with_name(obj_path.name + "_mesh.obj")
                    try:
                        _save_obj(mesh, str(obj_path))
                        print(f"Wrote {obj_path}")
                    except Exception as exc:
                        print(f"OBJ export failed: {exc}")
                prev = args.out.with_suffix("")
                prev = prev.with_name(prev.name + "_3d.png")
                out3d = render_meshes_preview(
                    [mesh],
                    str(prev),
                    width=int(args.preview_size[0]),
                    height=int(args.preview_size[1]),
                    dpi=int(args.preview_dpi),
                )
                if out3d:
                    print(f"Wrote {out3d}")
            else:
                print("3D displacement skipped (geometry unavailable)")
        except Exception as exc:
            print(f"3D preview failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

