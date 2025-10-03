# examples/m1_mpl_image_demo.py
# Workstream M1: Matplotlib Adapter (Image)
# - Rasterize a Matplotlib figure or axes to RGBA and save a deterministic PNG

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forge3d.adapters import (
    mpl_rasterize_figure,
    mpl_rasterize_axes,
    mpl_height_from_luminance,
    is_matplotlib_available_image,
)
from forge3d.helpers.offscreen import save_png_deterministic
# 3D mesh preview helpers no longer used in this example; using Matplotlib surface instead
try:
    from forge3d.io import save_obj as _save_obj
except Exception:
    _save_obj = None


def build_demo_figure():
    import matplotlib.pyplot as plt
    x = np.linspace(0, 2 * np.pi, 512)
    y = np.sin(3 * x) * np.cos(2 * x)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(x, y, label="sin(3x) cos(2x)")
    ax.set_title("M1 Matplotlib → RGBA Demo")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax, x, y


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 Matplotlib → RGBA + 3D preview demo")
    parser.add_argument("--out", type=Path, default=Path("reports/m1_mpl_image.png"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--axes", action="store_true", help="Rasterize only the Axes (tight bbox)")
    parser.add_argument("--heightmap", action="store_true", help="Also save a height-from-luminance PNG")
    # 3D options
    parser.add_argument("--make-3d", action="store_true", default=True, help="Extrude luminance into a 3D displaced plane and save preview")
    parser.add_argument("--height-scale", type=float, default=1.0, help="Displacement scale for heightmap/extrusion")
    parser.add_argument("--plane-res", type=int, nargs=2, default=(128, 128), metavar=("W","H"), help="Plane resolution used for displacement")
    parser.add_argument("--preview-size", type=int, nargs=2, default=(800, 600), metavar=("W","H"), help="3D preview image size")
    parser.add_argument("--preview-dpi", type=int, default=150, help="3D preview DPI for rasterization")
    parser.add_argument("--save-obj", action="store_true", help="Also save the displaced mesh as OBJ (requires native)")
    # From-image extrusion options
    parser.add_argument("--from-image", type=Path, default=None, help="Extrude this image into a 3D surface (PNG/JPG)")
    parser.add_argument("--extent", type=float, nargs=4, default=None, metavar=("XMIN","XMAX","YMIN","YMAX"), help="Axis extent mapping for the image")
    parser.add_argument("--xlabel", type=str, default=None, help="X-axis label for 3D plot")
    parser.add_argument("--ylabel", type=str, default=None, help="Y-axis label for 3D plot")
    parser.add_argument("--zlabel", type=str, default="Height", help="Z-axis label for 3D plot")
    parser.add_argument("--title", type=str, default="Extruded Image", help="Title for 3D plot")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for 3D surface")
    parser.add_argument("--invert", action="store_true", help="Invert luminance for height")
    parser.add_argument("--cbar-label", type=str, default="Height", help="Colorbar label")
    parser.add_argument("--recreate-attached", action="store_true", help="Programmatically recreate the demo contour and extrude it to 3D (self-contained)")
    # Camera
    parser.add_argument("--elev", type=float, default=30.0, help="3D camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=35.0, help="3D camera azimuth in degrees")
    args = parser.parse_args()

    if not is_matplotlib_available_image():
        print("Matplotlib not available. Install with: pip install matplotlib")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Helper: robustly normalize a heightmap to [0,1] to avoid extreme Z due to tiny dynamic range
    def _robust_normalize(h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float32)
        # Use robust percentiles to avoid outliers or near-constant images exploding the scale
        try:
            p1 = float(np.percentile(h, 1.0))
            p99 = float(np.percentile(h, 99.0))
        except Exception:
            p1 = float(h.min())
            p99 = float(h.max())
        denom = max(p99 - p1, 1e-2)  # clamp denom to avoid huge amplification
        hn = np.clip((h - p1) / denom, 0.0, 1.0)
        return hn.astype(np.float32, copy=False)

    # If requested, recreate the attached image programmatically and extrude it
    if args.recreate_attached:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import Normalize
            from matplotlib import tri as mtri

            # FEM-like data from the user's snippet
            nodes_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            nodes_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
            nodal_values = np.array([1.0, 0.9, 1.1, 0.9, 2.1, 2.1, 0.9, 1.0, 1.0, 0.9, 0.8], dtype=np.float32)
            elements_tris = [[2, 6, 5], [5, 6, 10], [10, 9, 5]]
            elements_quads = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 8, 7], [4, 5, 9, 8]]

            def quads_to_tris(quads):
                tris = [[None for _ in range(3)] for _ in range(2 * len(quads))]
                for i in range(len(quads)):
                    j = 2 * i
                    n0, n1, n2, n3 = quads[i]
                    tris[j][0] = n0
                    tris[j][1] = n1
                    tris[j][2] = n2
                    tris[j + 1][0] = n2
                    tris[j + 1][1] = n3
                    tris[j + 1][2] = n0
                return tris

            elements_all_tris = elements_tris + quads_to_tris(elements_quads)
            triangulation = mtri.Triangulation(nodes_x, nodes_y, elements_all_tris)

            # 2D tripcolor with discrete bins (no mesh edges), with labels and colorbar
            fig2d, ax2d = plt.subplots(figsize=(6, 4), constrained_layout=True, dpi=int(args.dpi))
            cmap2d = plt.get_cmap(str(args.cmap))
            levels = np.linspace(float(nodal_values.min()), float(nodal_values.max()), 10)
            from matplotlib.colors import BoundaryNorm
            bnorm = BoundaryNorm(levels, cmap2d.N, clip=True)
            tpc = ax2d.tripcolor(triangulation, nodal_values, cmap=cmap2d, norm=bnorm, shading='flat', edgecolors='none')
            ax2d.set_aspect('equal', adjustable='box')
            ax2d.set_xlim(float(nodes_x.min()), float(nodes_x.max()))
            ax2d.set_ylim(float(nodes_y.min()), float(nodes_y.max()))
            ax2d.set_xlabel('x')
            ax2d.set_ylabel('y')
            ax2d.set_title('Recreated Contour (2D)')
            mappable2d = cm.ScalarMappable(norm=bnorm, cmap=cmap2d)
            mappable2d.set_array([])
            cbar2d = fig2d.colorbar(mappable2d, ax=ax2d, boundaries=levels)
            cbar2d.set_label('value')
            rgba2d = mpl_rasterize_figure(fig2d, dpi=int(args.dpi), facecolor='white')
            save_png_deterministic(str(args.out), rgba2d)
            print(f"Wrote {args.out}")

            # 3D extruded surface using the same triangulation (omit edges)
            nv_min = float(nodal_values.min())
            nv_max = float(nodal_values.max())
            nv_denom = (nv_max - nv_min) or 1.0
            Z = (((nodal_values - nv_min) / nv_denom) * float(args.height_scale)).astype(np.float32)
            fig3d = plt.figure(figsize=(float(args.preview_size[0]) / float(args.preview_dpi),
                                        float(args.preview_size[1]) / float(args.preview_dpi)),
                                dpi=int(args.preview_dpi), constrained_layout=True)
            ax3d = fig3d.add_subplot(1, 1, 1, projection='3d')
            # Build discrete facecolors using the same bins as 2D
            cmap = plt.get_cmap(str(args.cmap))
            from matplotlib.colors import BoundaryNorm
            bnorm = BoundaryNorm(levels, cmap.N, clip=True)
            tris_idx = triangulation.triangles
            tri_vals = nodal_values[tris_idx].mean(axis=1)
            facecolors = cmap(bnorm(tri_vals))

            # Render triangles exactly via Poly3DCollection to match 2D fill
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            tris_idx = triangulation.triangles
            polys = []
            for tri in tris_idx:
                verts = list(zip(nodes_x[tri], nodes_y[tri], Z[tri]))
                polys.append(verts)

            coll = Poly3DCollection(polys, facecolors=facecolors, edgecolors='none', linewidths=0.0)
            try:
                coll.set_depthshade(False)
            except Exception:
                pass
            ax3d.add_collection3d(coll)
            # Camera angle
            ax3d.view_init(elev=float(args.elev), azim=float(args.azim))
            # Ortho projection + equal XY aspect
            try:
                ax3d.set_proj_type('ortho')
            except Exception:
                pass
            try:
                xrange = float(nodes_x.max() - nodes_x.min()) or 1.0
                yrange = float(nodes_y.max() - nodes_y.min()) or 1.0
                zrange = float(Z.max() - Z.min()) or 1.0
                ax3d.set_box_aspect((xrange, yrange, zrange))
            except Exception:
                pass
            # Remove pane fills and grid to avoid gray overlays
            try:
                for pane_axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
                    pane_axis.pane.fill = False
                    pane_axis.pane.set_edgecolor('white')
                ax3d.grid(False)
            except Exception:
                pass
            # Match 2D limits
            ax3d.set_xlim(float(nodes_x.min()), float(nodes_x.max()))
            ax3d.set_ylim(float(nodes_y.min()), float(nodes_y.max()))
            ax3d.set_zlim(float(Z.min()), float(Z.max()))
            
            ax3d.set_xlabel('x')
            ax3d.set_ylabel('y')
            ax3d.set_zlabel('height')
            ax3d.set_title('Extruded Contour (3D)')
            mappable = cm.ScalarMappable(norm=bnorm, cmap=cmap)
            mappable.set_array([])
            try:
                cbar3d = fig3d.colorbar(mappable, ax=ax3d, shrink=0.7, boundaries=levels)
            except Exception:
                cbar3d = fig3d.colorbar(mappable, ax=ax3d, shrink=0.7)
            cbar3d.set_label('height')

            rgba3d = mpl_rasterize_figure(fig3d, dpi=int(args.preview_dpi), facecolor='white')
            out3d = args.out.with_suffix("")
            out3d = out3d.with_name(out3d.name + "_3d.png")
            save_png_deterministic(str(out3d), rgba3d)
            print(f"Wrote {out3d}")
            return 0
        except Exception as exc:
            print(f"Failed to recreate+extrude: {exc}")

    # If an external image is provided, extrude that image directly to a 3D surface and return
    if args.from_image is not None:
        try:
            # Lazy imports to keep optional deps light
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import Normalize
            from PIL import Image

            # Load image and compute luminance-based height
            img = Image.open(str(args.from_image)).convert("RGBA")
            rgba = np.asarray(img, dtype=np.float32) / 255.0
            # Compute luminance from RGB
            hm = 0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2]
            if args.invert:
                hm = 1.0 - hm

            H, W = hm.shape
            if args.extent is not None:
                xmin, xmax, ymin, ymax = map(float, args.extent)
                xs = np.linspace(xmin, xmax, W, dtype=np.float32)
                ys = np.linspace(ymin, ymax, H, dtype=np.float32)
            else:
                xs = np.linspace(0, float(W - 1), W, dtype=np.float32)
                ys = np.linspace(0, float(H - 1), H, dtype=np.float32)
            X, Y = np.meshgrid(xs, ys)
            # Normalize heights robustly to [0,1] then apply height_scale
            hn = _robust_normalize(hm)
            Z = hn * float(args.height_scale)

            fig_w = float(args.preview_size[0]) / float(args.preview_dpi)
            fig_h = float(args.preview_size[1]) / float(args.preview_dpi)
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(args.preview_dpi), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            norm = Normalize(vmin=float(Z.min()), vmax=float(Z.max()))
            cmap = plt.get_cmap(str(args.cmap))
            facecolors = cmap(norm(Z))

            ax.plot_surface(
                X, Y, Z,
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                linewidth=0.0,
                antialiased=True,
                edgecolor='none',
                shade=False
            )
            # Camera angle
            ax.view_init(elev=float(args.elev), azim=float(args.azim))
            # Ortho projection + equal XY aspect
            try:
                ax.set_proj_type('ortho')
            except Exception:
                pass
            try:
                xrange = float(X.max() - X.min()) or 1.0
                yrange = float(Y.max() - Y.min()) or 1.0
                zrange = float(Z.max() - Z.min()) or 1.0
                ax.set_box_aspect((xrange, yrange, zrange))
            except Exception:
                pass
            # Match 2D limits
            ax.set_xlim(float(X.min()), float(X.max()))
            ax.set_ylim(float(Y.min()), float(Y.max()))
            ax.set_zlim(float(Z.min()), float(Z.max()))

            ax.set_title(str(args.title))
            ax.set_xlabel(str(args.xlabel) if args.xlabel else "X")
            ax.set_ylabel(str(args.ylabel) if args.ylabel else "Y")
            ax.set_zlabel(str(args.zlabel))

            # Colorbar legend representing height
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
            cbar.set_label(str(args.cbar_label))

            # Rasterize deterministically and save
            rgba_out = mpl_rasterize_figure(fig, dpi=int(args.preview_dpi), facecolor='white')
            out3d = args.out.with_suffix("")
            out3d = out3d.with_name(out3d.name + "_3d.png")
            save_png_deterministic(str(out3d), rgba_out)
            print(f"Wrote {out3d}")
            return 0
        except Exception as exc:
            print(f"Failed to extrude image to 3D: {exc}")
            # Continue to standard M1 path if image extrusion fails

    fig, ax, x, y = build_demo_figure()

    if args.axes:
        rgba = mpl_rasterize_axes(ax, dpi=args.dpi, bbox_inches='tight', pad_inches=0.0)
    else:
        rgba = mpl_rasterize_figure(fig, dpi=args.dpi, facecolor='white')

    save_png_deterministic(str(args.out), rgba)
    print(f"Wrote {args.out}")

    if args.heightmap:
        hm = mpl_height_from_luminance(rgba)
        # map to grayscale
        hm_rgba = np.stack([hm, hm, hm, np.ones_like(hm)], axis=-1)
        hm_png = args.out.with_suffix("")
        hm_png = hm_png.with_name(hm_png.name + "_height.png")
        save_png_deterministic(str(hm_png), (hm_rgba * 255).astype(np.uint8))
        print(f"Wrote {hm_png}")

    # Optional 3D displaced preview (Matplotlib-based, shadow-free)
    if args.make_3d:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize

            # Build a thin curtain extrusion from the actual 2D line (x,y)
            x_arr = np.asarray(x, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32)
            N = x_arr.size
            D = max(2, int(args.plane_res[1]))

            x_rng = float(x_arr.max() - x_arr.min()) or 1.0
            depth_extent = 0.15 * x_rng
            depth = np.linspace(0.0, depth_extent, D, dtype=np.float32)

            # Robustly normalize Y to [0,1], then scale to a sane Z amplitude relative to X span
            y_n = _robust_normalize(y_arr)
            z_amp = min(float(args.height_scale), 0.15 * x_rng)
            z_line = y_n * z_amp

            X = np.tile(x_arr[:, None], (1, D))
            Y = np.tile(depth[None, :], (N, 1))
            Z = np.tile(z_line[:, None], (1, D))

            fig_w = float(args.preview_size[0]) / float(args.preview_dpi)
            fig_h = float(args.preview_size[1]) / float(args.preview_dpi)
            fig3 = plt.figure(figsize=(fig_w, fig_h), dpi=int(args.preview_dpi), constrained_layout=True)
            ax3 = fig3.add_subplot(1, 1, 1, projection='3d')

            norm = Normalize(vmin=float(Z.min()), vmax=float(Z.max()))
            cmap = plt.get_cmap(str(args.cmap))
            facecolors = cmap(norm(Z))

            ax3.plot_surface(
                X, Y, Z,
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                linewidth=0.0,
                antialiased=True,
                edgecolor='none',
                shade=False,
            )
            ax3.view_init(elev=float(args.elev), azim=float(args.azim))
            try:
                ax3.set_proj_type('ortho')
            except Exception:
                pass
            try:
                ax3.set_box_aspect((x_rng, depth_extent, z_amp or 1.0))
            except Exception:
                pass
            try:
                for pane_axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
                    pane_axis.pane.fill = False
                    pane_axis.pane.set_edgecolor('white')
                ax3.grid(False)
            except Exception:
                pass

            ax3.set_xlabel('x')
            ax3.set_ylabel('depth')
            ax3.set_zlabel('height')
            ax3.set_title('Extruded Plot (3D)')

            rgba_out = mpl_rasterize_figure(fig3, dpi=int(args.preview_dpi), facecolor='white')
            prev_path = args.out.with_suffix("")
            prev_path = prev_path.with_name(prev_path.name + "_3d.png")
            save_png_deterministic(str(prev_path), rgba_out)
            print(f"Wrote {prev_path}")
        except Exception as exc:
            print(f"3D preview failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

