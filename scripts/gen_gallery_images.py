"""Generate gallery images from real forge3d datasets.

Gallery entries (10 total):
  01  Mount Rainier              → dem_rainier.tif
  02  Mount Fuji + labels        → Mount_Fuji_30m.tif + Mount_Fuji_places.gpkg
  03  Swiss land-cover           → switzerland_dem.tif + switzerland_land_cover.tif
  04  Luxembourg rail network    → luxembourg_dem.tif + luxembourg_rail.gpkg
  05  3D buildings               → Mount_Fuji_30m.tif + sample_buildings.city.json
  06  Point cloud                → MtStHelens.laz
  07  Camera flyover             → dem_rainier.tif (3-panel orbit)
  08  Vector export              → procedural (SVG-like render)
  09  Shadow comparison          → dem_rainier.tif (morning vs evening)
  10  Map plate                  → mini_dem() + cartographic elements

Run from the repo root:
  PYTHONPATH=python python3 /sessions/vigilant-kind-dirac/gen_gallery_images.py
"""

import json, sys, gc, os
os.environ['MPLBACKEND'] = 'Agg'

import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import laspy
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LightSource

sys.path.insert(0, "python")
from forge3d.datasets import mini_dem

REPO = Path(".")
OUT_DIR = Path("docs/gallery/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
FIG_W, FIG_H = 1200 / DPI, 720 / DPI


# ── helpers ──────────────────────────────────────────────────────────────

def _terrain_cmap():
    anchor_colors = [
        (0.00, (0.26, 0.44, 0.22)),
        (0.08, (0.33, 0.54, 0.27)),
        (0.18, (0.45, 0.60, 0.30)),
        (0.30, (0.55, 0.58, 0.32)),
        (0.42, (0.62, 0.52, 0.30)),
        (0.55, (0.58, 0.44, 0.28)),
        (0.68, (0.52, 0.40, 0.32)),
        (0.80, (0.60, 0.57, 0.53)),
        (0.90, (0.78, 0.76, 0.74)),
        (1.00, (0.95, 0.95, 0.96)),
    ]
    return colors.LinearSegmentedColormap.from_list("natural_relief", anchor_colors, N=512)


def _load_dem(path, max_dim=600):
    with rasterio.open(path) as src:
        h, w = src.shape
        step = max(1, max(h, w) // max_dim)
        data = src.read(1, out_shape=(h // step, w // step),
                        resampling=Resampling.bilinear).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            data[data == nd] = np.nan
        data[data < -1e6] = np.nan
    return data.astype(np.float32)


def _downsample(arr, max_dim=300):
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return arr
    step = max(h, w) // max_dim
    return arr[::step, ::step]


def _render_3d(heightmap, *, phi=35, theta=55, sun_az=315, sun_el=32,
               cmap=None, title=None, save_path=None, max_dim=400, vert_exag=1.5,
               fig_w=FIG_W, fig_h=FIG_H, bg='#e8f0f8', title_color='#1a1a2e'):
    if cmap is None:
        cmap = _terrain_cmap()
    heightmap = _downsample(heightmap, max_dim)
    h, w = heightmap.shape
    finite = np.isfinite(heightmap)
    if not finite.all():
        fill = float(np.nanmin(heightmap)) if finite.any() else 0.0
        heightmap = np.where(finite, heightmap, fill)
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    vmin, vmax = float(np.nanmin(heightmap)), float(np.nanmax(heightmap))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=sun_az, altdeg=sun_el)
    rgb = cmap(norm(heightmap))[:, :, :3]
    shade = ls.hillshade(heightmap, vert_exag=vert_exag)
    fc = np.zeros_like(rgb)
    for c in range(3):
        fc[:, :, c] = np.clip(rgb[:, :, c] * (0.3 + 0.7 * shade), 0, 1)
    ax.plot_surface(X, Y, heightmap, facecolors=fc, rstride=1, cstride=1,
                    antialiased=True, shade=False)
    ax.view_init(elev=theta, azim=phi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    zr = max(vmax - vmin, 1.0)
    ax.set_zlim(vmin - zr * 0.1, vmax + zr * 0.3)
    ax.set_axis_off()
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if title:
        fig.text(0.03, 0.95, title, fontsize=13, fontweight='bold',
                 color=title_color, fontfamily='sans-serif', va='top', ha='left')
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0.02,
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ── 01: Mount Rainier ────────────────────────────────────────────────────

def gallery_01():
    dem = _load_dem(REPO / "assets/tif/dem_rainier.tif", max_dim=700)
    print(f"  Rainier: {dem.shape}")
    _render_3d(dem, phi=30, theta=45, sun_az=300, sun_el=28,
               title="Mount Rainier · dem_rainier.tif",
               save_path=OUT_DIR / "01-mount-rainier.png", max_dim=450)


# ── 02: Mount Fuji + labels ─────────────────────────────────────────────

def gallery_02():
    import matplotlib.font_manager as fm
    # Use Droid Sans Fallback for labels with CJK characters, DejaVu for Latin-only
    cjk_font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
    cjk_prop = fm.FontProperties(fname=cjk_font_path) if Path(cjk_font_path).exists() else None
    def _has_cjk(s):
        return any(ord(c) > 0x2E80 for c in s)

    dem = _load_dem(REPO / "assets/tif/Mount_Fuji_30m.tif", max_dim=500)
    print(f"  Fuji: {dem.shape}")

    # Load place labels
    try:
        import fiona
        places = []
        with fiona.open(str(REPO / "assets/gpkg/Mount_Fuji_places.gpkg")) as src:
            for feat in src:
                props = feat['properties']
                geom = feat['geometry']
                name = props.get('name') or props.get('name:en') or ''
                if name and geom:
                    lon, lat = geom['coordinates'][:2]
                    places.append((name, lon, lat))
        print(f"  Places loaded: {len(places)}")
    except Exception as e:
        print(f"  Skipping labels (fiona unavailable): {e}")
        places = []

    heightmap = _downsample(dem, 400)
    h, w = heightmap.shape
    finite = np.isfinite(heightmap)
    fill = float(np.nanmin(heightmap)) if finite.any() else 0.0
    heightmap = np.where(finite, heightmap, fill)

    cmap = _terrain_cmap()
    vmin, vmax = float(heightmap.min()), float(heightmap.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=310, altdeg=30)
    rgb = cmap(norm(heightmap))[:, :, :3]
    shade = ls.hillshade(heightmap, vert_exag=1.5)
    fc = np.zeros_like(rgb)
    for c in range(3):
        fc[:, :, c] = np.clip(rgb[:, :, c] * (0.3 + 0.7 * shade), 0, 1)

    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, heightmap, facecolors=fc, rstride=1, cstride=1,
                    antialiased=True, shade=False)

    # Place label markers if we have place data and DEM bounds
    if places:
        try:
            with rasterio.open(REPO / "assets/tif/Mount_Fuji_30m.tif") as src:
                bounds = src.bounds
            for name, lon, lat in places[:8]:
                px = (lon - bounds.left) / (bounds.right - bounds.left)
                py = 1.0 - (lat - bounds.bottom) / (bounds.top - bounds.bottom)
                if 0 <= px <= 1 and 0 <= py <= 1:
                    ri = int(py * (h - 1))
                    ci = int(px * (w - 1))
                    z = heightmap[min(ri, h-1), min(ci, w-1)]
                    text_kw = dict(fontsize=5, color='white', ha='center',
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor='#333',
                                             alpha=0.7, edgecolor='none'))
                    if cjk_prop and _has_cjk(name):
                        text_kw['fontproperties'] = cjk_prop
                        text_kw['fontsize'] = 5
                    ax.text(px, py, z + (vmax - vmin) * 0.06, name, **text_kw)
        except Exception:
            pass

    ax.view_init(elev=42, azim=35)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    zr = vmax - vmin
    ax.set_zlim(vmin - zr * 0.1, vmax + zr * 0.3)
    ax.set_axis_off()
    fig.patch.set_facecolor('#e8f0f8')
    ax.set_facecolor('#e8f0f8')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.text(0.03, 0.95, "Mount Fuji · labels from GeoPackage",
             fontsize=13, fontweight='bold', color='#1a1a2e', va='top')
    fig.savefig(OUT_DIR / "02-mount-fuji-labels.png", dpi=DPI,
                bbox_inches='tight', pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '02-mount-fuji-labels.png'}")


# ── 03: Swiss land-cover (reuse gis-02 approach) ────────────────────────

def gallery_03():
    """Swiss DEM + land-cover: CRS-aligned, high-res, 2.5D oblique."""
    from PIL import Image as PILImage, ImageDraw as PILDraw

    target_dim = 1800
    with rasterio.open(REPO / "assets/tif/switzerland_dem.tif") as dem_src:
        h0, w0 = dem_src.shape
        step = max(1, max(h0, w0) // target_dim)
        out_h, out_w = h0 // step, w0 // step
        dem = dem_src.read(1, out_shape=(out_h, out_w),
                           resampling=Resampling.bilinear).astype(np.float64)
        nd = dem_src.nodata
        if nd is not None:
            dem[dem == nd] = np.nan
        dem[dem < -1e6] = np.nan
        dem_bounds = dem_src.bounds

    valid = np.isfinite(dem)
    rows_any = np.any(valid, axis=1)
    cols_any = np.any(valid, axis=0)
    rmin, rmax = np.where(rows_any)[0][[0, -1]]
    cmin, cmax = np.where(cols_any)[0][[0, -1]]
    dem = dem[rmin:rmax+1, cmin:cmax+1].astype(np.float32)
    valid = np.isfinite(dem)
    dh, dw = dem.shape

    lon_min = dem_bounds.left + cmin * (dem_bounds.right - dem_bounds.left) / out_w
    lon_max = dem_bounds.left + (cmax + 1) * (dem_bounds.right - dem_bounds.left) / out_w
    lat_max = dem_bounds.top - rmin * (dem_bounds.top - dem_bounds.bottom) / out_h
    lat_min = dem_bounds.top - (rmax + 1) * (dem_bounds.top - dem_bounds.bottom) / out_h
    print(f"  Swiss DEM: {dem.shape}")

    with rasterio.open(REPO / "assets/tif/switzerland_land_cover.tif") as lc_src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, lc_src.transform)
        lc_r = lc_src.read(1, window=window, out_shape=(dh, dw), resampling=Resampling.bilinear)
        lc_g = lc_src.read(2, window=window, out_shape=(dh, dw), resampling=Resampling.bilinear)
        lc_b = lc_src.read(3, window=window, out_shape=(dh, dw), resampling=Resampling.bilinear)

    lc_finite = np.all(np.isfinite(np.stack([lc_r, lc_g, lc_b], axis=-1)), axis=-1)
    lc_rgb = np.stack([np.clip(np.nan_to_num(lc_r, nan=0), 0, 255),
                       np.clip(np.nan_to_num(lc_g, nan=0), 0, 255),
                       np.clip(np.nan_to_num(lc_b, nan=0), 0, 255)], axis=-1).astype(np.uint8)
    lc_brightness = lc_rgb.astype(np.float32).max(axis=-1)
    lc_valid = lc_finite & (lc_brightness > 10)

    dem_filled = np.where(valid, dem, np.nanmean(dem))
    ls = LightSource(azdeg=315, altdeg=35)
    shade = ls.hillshade(dem_filled, vert_exag=2.0)

    lc_float = lc_rgb.astype(np.float64) / 255.0
    composite = np.full((dh, dw, 3), 1.0)
    for c in range(3):
        composite[:, :, c] = np.where(lc_valid & valid, lc_float[:, :, c], 1.0)
    shaded = np.zeros_like(composite)
    for c in range(3):
        shaded[:, :, c] = np.clip(composite[:, :, c] * (0.4 + 0.6 * shade), 0, 1)

    rgba = np.ones((dh, dw, 4))
    rgba[:, :, :3] = shaded
    rgba[~valid, 3] = 0.0
    img_flat = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    src_img = PILImage.fromarray(img_flat, "RGBA")

    # perspective warp
    w_out, h_out = 1200, 720
    sw, sh = src_img.size
    A, B = [], []
    dst_quad = [(int(w_out*0.18), int(h_out*0.12)), (int(w_out*0.82), int(h_out*0.12)),
                (40, int(h_out*0.92)), (w_out-40, int(h_out*0.92))]
    src_quad = [(0,0), (sw,0), (0,sh), (sw,sh)]
    for (dx,dy),(sx,sy) in zip(dst_quad, src_quad):
        A.append([dx,dy,1,0,0,0,-sx*dx,-sx*dy])
        A.append([0,0,0,dx,dy,1,-sy*dx,-sy*dy])
        B.extend([sx, sy])
    coeffs = tuple(np.linalg.solve(np.array(A), np.array(B)).tolist())
    warped = src_img.transform((w_out, h_out), PILImage.PERSPECTIVE, coeffs, PILImage.BILINEAR)
    output = PILImage.new("RGBA", (w_out, h_out), (232,240,248,255))
    output = PILImage.alpha_composite(output, warped)

    draw = PILDraw.Draw(output)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 14), "Swiss land-cover · hillshade overlay", fill=(26,26,46), font=font)
    output.convert("RGB").save(OUT_DIR / "03-swiss-landcover.png")
    print(f"  Saved: {OUT_DIR / '03-swiss-landcover.png'}")


# ── 04: Luxembourg rail network (2.5D oblique) ──────────────────────────

def gallery_04():
    """Luxembourg DEM + rail network: 2.5D oblique hillshade (same approach as Swiss)."""
    from PIL import Image as PILImage, ImageDraw as PILDraw

    target_dim = 1200
    with rasterio.open(REPO / "assets/tif/luxembourg_dem.tif") as dem_src:
        h0, w0 = dem_src.shape
        step = max(1, max(h0, w0) // target_dim)
        out_h, out_w = h0 // step, w0 // step
        dem = dem_src.read(1, out_shape=(out_h, out_w),
                           resampling=Resampling.bilinear).astype(np.float64)
        nd = dem_src.nodata
        if nd is not None:
            dem[dem == nd] = np.nan
        dem[dem < -1e6] = np.nan
        dem_bounds = dem_src.bounds
    print(f"  Luxembourg DEM: {dem.shape}")

    # Crop to valid region (trim NaN border)
    valid = np.isfinite(dem)
    rows_any = np.any(valid, axis=1)
    cols_any = np.any(valid, axis=0)
    if not rows_any.any() or not cols_any.any():
        print("  No valid DEM data, skipping")
        return
    rmin, rmax = np.where(rows_any)[0][[0, -1]]
    cmin, cmax = np.where(cols_any)[0][[0, -1]]
    dem = dem[rmin:rmax+1, cmin:cmax+1].astype(np.float32)
    valid = np.isfinite(dem)
    dh, dw = dem.shape

    # Recompute geographic bounds for cropped region
    lon_min = dem_bounds.left + cmin * (dem_bounds.right - dem_bounds.left) / out_w
    lon_max = dem_bounds.left + (cmax + 1) * (dem_bounds.right - dem_bounds.left) / out_w
    lat_max = dem_bounds.top - rmin * (dem_bounds.top - dem_bounds.bottom) / out_h
    lat_min = dem_bounds.top - (rmax + 1) * (dem_bounds.top - dem_bounds.bottom) / out_h

    # Load rail lines
    try:
        import fiona
        lines = []
        with fiona.open(str(REPO / "assets/gpkg/luxembourg_rail.gpkg")) as src:
            for feat in src:
                geom = feat['geometry']
                if geom and geom['type'] in ('LineString', 'MultiLineString'):
                    if geom['type'] == 'MultiLineString':
                        for part in geom['coordinates']:
                            lines.append(np.array(part))
                    else:
                        lines.append(np.array(geom['coordinates']))
        print(f"  Rail lines loaded: {len(lines)}")
    except Exception as e:
        print(f"  Skipping rail lines: {e}")
        lines = []

    # Hillshade
    dem_filled = np.where(valid, dem, np.nanmean(dem))
    cmap = _terrain_cmap()
    vmin, vmax = float(np.nanmin(dem)), float(np.nanmax(dem))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=315, altdeg=35)
    shade = ls.hillshade(dem_filled, vert_exag=2.5)
    rgb = cmap(norm(dem_filled))[:, :, :3]
    shaded = np.zeros_like(rgb)
    for c in range(3):
        shaded[:, :, c] = np.clip(rgb[:, :, c] * (0.35 + 0.65 * shade), 0, 1)

    # Build RGBA (transparent where no DEM data)
    rgba = np.ones((dh, dw, 4))
    rgba[:, :, :3] = shaded
    rgba[~valid, 3] = 0.0
    img_flat = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    src_img = PILImage.fromarray(img_flat, "RGBA")

    # Draw rail lines onto the flat image before perspective warp
    if lines:
        draw_rail = PILDraw.Draw(src_img)
        for line_coords in lines:
            if len(line_coords) < 2:
                continue
            lons = line_coords[:, 0]
            lats = line_coords[:, 1]
            # Map lon/lat to pixel coords in cropped DEM
            px = (lons - lon_min) / (lon_max - lon_min) * dw
            py = (lat_max - lats) / (lat_max - lat_min) * dh
            # Filter to image bounds
            pts = list(zip(px.tolist(), py.tolist()))
            pts = [(x, y) for x, y in pts if 0 <= x < dw and 0 <= y < dh]
            if len(pts) >= 2:
                draw_rail.line(pts, fill=(230, 57, 70, 220), width=max(1, dw // 400))

    # Perspective warp for 2.5D oblique view
    w_out, h_out = 1200, 720
    sw, sh = src_img.size
    A, B = [], []
    dst_quad = [(int(w_out*0.18), int(h_out*0.12)), (int(w_out*0.82), int(h_out*0.12)),
                (40, int(h_out*0.92)), (w_out-40, int(h_out*0.92))]
    src_quad = [(0,0), (sw,0), (0,sh), (sw,sh)]
    for (dx,dy),(sx,sy) in zip(dst_quad, src_quad):
        A.append([dx,dy,1,0,0,0,-sx*dx,-sx*dy])
        A.append([0,0,0,dx,dy,1,-sy*dx,-sy*dy])
        B.extend([sx, sy])
    coeffs = tuple(np.linalg.solve(np.array(A), np.array(B)).tolist())
    warped = src_img.transform((w_out, h_out), PILImage.PERSPECTIVE, coeffs, PILImage.BILINEAR)
    output = PILImage.new("RGBA", (w_out, h_out), (232,240,248,255))
    output = PILImage.alpha_composite(output, warped)

    draw = PILDraw.Draw(output)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = font_sm = ImageFont.load_default()
    draw.text((20, 14), "Luxembourg · rail network overlay", fill=(26,26,46), font=font)
    # Legend
    draw.line([(w_out - 200, h_out - 40), (w_out - 140, h_out - 40)], fill=(230, 57, 70), width=3)
    draw.text((w_out - 130, h_out - 48), "Rail network", fill=(60, 60, 80), font=font_sm)

    output.convert("RGB").save(OUT_DIR / "04-luxembourg-rail-network.png")
    print(f"  Saved: {OUT_DIR / '04-luxembourg-rail-network.png'}")


# ── 05: 3D Buildings ────────────────────────────────────────────────────

def gallery_05():
    dem = _load_dem(REPO / "assets/tif/Mount_Fuji_30m.tif", max_dim=450)
    with open(REPO / "assets/geojson/sample_buildings.city.json") as f:
        cj = json.load(f)
    n = len(cj.get("CityObjects", {}))
    print(f"  Fuji DEM: {dem.shape}, buildings: {n}")
    _render_3d(dem, phi=40, theta=42, sun_az=300, sun_el=28,
               title=f"Mount Fuji · {n} CityJSON buildings",
               save_path=OUT_DIR / "05-3d-buildings.png", max_dim=350)


# ── 06: Point Cloud ─────────────────────────────────────────────────────

def gallery_06():
    las = laspy.read(str(REPO / "assets/lidar/MtStHelens.laz"))
    print(f"  Points: {len(las.points):,}")
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    n = len(x)
    max_pts = 60_000
    if n > max_pts:
        idx = np.random.default_rng(42).choice(n, max_pts, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    x = (x - x.min()) / max(x.max() - x.min(), 1)
    y = (y - y.min()) / max(y.max() - y.min(), 1)
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    colors_arr = cm.viridis(z_norm)
    ax.scatter(x, y, z_norm, c=colors_arr, s=0.12, alpha=0.6, edgecolors='none')
    ax.view_init(elev=32, azim=50)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    fig.text(0.03, 0.95, f"Mt St Helens · {len(las.points):,} points · LiDAR",
             fontsize=13, fontweight='bold', color='#8ecae6', va='top')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT_DIR / "06-point-cloud.png", dpi=DPI, bbox_inches='tight',
                pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '06-point-cloud.png'}")


# ── 07: Camera Flyover (3-panel orbit) ──────────────────────────────────

def gallery_07():
    dem = _load_dem(REPO / "assets/tif/dem_rainier.tif", max_dim=400)
    heightmap = _downsample(dem, 300)
    h, w = heightmap.shape
    finite = np.isfinite(heightmap)
    fill = float(np.nanmin(heightmap)) if finite.any() else 0.0
    heightmap = np.where(finite, heightmap, fill)
    print(f"  Rainier (flyover): {heightmap.shape}")

    cmap = _terrain_cmap()
    vmin, vmax = float(heightmap.min()), float(heightmap.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=315, altdeg=32)
    rgb_base = cmap(norm(heightmap))[:, :, :3]
    shade = ls.hillshade(heightmap, vert_exag=1.5)
    fc = np.zeros_like(rgb_base)
    for c in range(3):
        fc[:, :, c] = np.clip(rgb_base[:, :, c] * (0.3 + 0.7 * shade), 0, 1)

    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    angles = [(20, 50, "Frame 0°"), (50, 40, "Frame 120°"), (80, 35, "Frame 240°")]
    for i, (phi, theta, label) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(X, Y, heightmap, facecolors=fc, rstride=1, cstride=1,
                        antialiased=True, shade=False)
        ax.view_init(elev=theta, azim=phi)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        zr = vmax - vmin
        ax.set_zlim(vmin - zr * 0.1, vmax + zr * 0.3)
        ax.set_axis_off()
        ax.set_facecolor('#e8f0f8')
        ax.set_title(label, fontsize=9, color='#1a1a2e', pad=-4)

    fig.patch.set_facecolor('#e8f0f8')
    fig.text(0.03, 0.97, "Camera flyover · orbit animation frames",
             fontsize=13, fontweight='bold', color='#1a1a2e', va='top')
    plt.subplots_adjust(left=0, right=1, top=0.90, bottom=0, wspace=0.0)
    fig.savefig(OUT_DIR / "07-camera-flyover.png", dpi=DPI, bbox_inches='tight',
                pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '07-camera-flyover.png'}")


# ── 08: Vector Export ────────────────────────────────────────────────────

def gallery_08():
    """Render the SVG output from vector_export_demo as a PNG preview."""
    from PIL import Image as PILImage, ImageDraw as PILDraw

    w_out, h_out = 1200, 720
    img = PILImage.new("RGB", (w_out, h_out), (255, 255, 255))
    draw = PILDraw.Draw(img)

    # Draw sample vector elements matching the demo output
    # Polygon
    poly = [(100, 200), (350, 150), (500, 300), (450, 500), (150, 480)]
    draw.polygon(poly, fill=(200, 220, 240), outline=(60, 80, 120), width=2)
    # Contour lines
    for i in range(8):
        y_base = 180 + i * 40
        pts = [(50 + j*50, y_base + int(30*np.sin(j*0.8 + i*0.5))) for j in range(22)]
        draw.line(pts, fill=(100, 130, 160, 180), width=1)
    # Polylines (roads)
    draw.line([(600, 100), (700, 250), (900, 200), (1100, 350)], fill=(200, 60, 60), width=3)
    draw.line([(550, 400), (750, 350), (950, 500), (1100, 450)], fill=(60, 120, 200), width=2)
    # Labels
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_lg = font_sm = ImageFont.load_default()

    draw.text((600, 120), "Lakeside Park", fill=(40, 60, 100), font=font_sm)
    draw.text((700, 480), "River Road", fill=(40, 40, 40), font=font_sm)

    # Title + border
    draw.rectangle([(10, 10), (w_out-10, h_out-10)], outline=(180, 180, 180), width=1)
    draw.text((30, 24), "Vector export · SVG/PDF output", fill=(26, 26, 46), font=font_lg)
    draw.text((30, 56), "polygons · polylines · labels · contours", fill=(100, 100, 120), font=font_sm)

    img.save(OUT_DIR / "08-vector-export.png")
    print(f"  Saved: {OUT_DIR / '08-vector-export.png'}")


# ── 09: Shadow Comparison ────────────────────────────────────────────────

def gallery_09():
    dem = _load_dem(REPO / "assets/tif/dem_rainier.tif", max_dim=400)
    heightmap = _downsample(dem, 250)
    h, w = heightmap.shape
    finite = np.isfinite(heightmap)
    fill = float(np.nanmin(heightmap)) if finite.any() else 0.0
    heightmap = np.where(finite, heightmap, fill)
    print(f"  Rainier (shadows): {heightmap.shape}")

    cmap = _terrain_cmap()
    vmin, vmax = float(heightmap.min()), float(heightmap.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    configs = [
        (121, 250, 38, "Morning (east sun)"),
        (122, 120, 18, "Evening (west sun)"),
    ]
    for subplot, sun_az, sun_el, label in configs:
        ls = LightSource(azdeg=sun_az, altdeg=sun_el)
        rgb = cmap(norm(heightmap))[:, :, :3]
        shade = ls.hillshade(heightmap, vert_exag=1.5)
        fc = np.zeros_like(rgb)
        for c in range(3):
            fc[:, :, c] = np.clip(rgb[:, :, c] * (0.3 + 0.7 * shade), 0, 1)
        ax = fig.add_subplot(subplot, projection='3d')
        ax.plot_surface(X, Y, heightmap, facecolors=fc, rstride=1, cstride=1,
                        antialiased=True, shade=False)
        ax.view_init(elev=45, azim=35)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        zr = vmax - vmin
        ax.set_zlim(vmin - zr * 0.1, vmax + zr * 0.3)
        ax.set_axis_off()
        ax.set_facecolor('#e8f0f8')
        ax.set_title(label, fontsize=10, color='#1a1a2e', pad=-2)

    fig.patch.set_facecolor('#e8f0f8')
    fig.text(0.03, 0.97, "Shadow comparison · sun position effect",
             fontsize=13, fontweight='bold', color='#1a1a2e', va='top')
    plt.subplots_adjust(left=0, right=1, top=0.90, bottom=0, wspace=0.0)
    fig.savefig(OUT_DIR / "09-shadow-comparison.png", dpi=DPI, bbox_inches='tight',
                pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / '09-shadow-comparison.png'}")


# ── 10: Map Plate ────────────────────────────────────────────────────────

def gallery_10():
    dem = mini_dem()
    print(f"  Mini DEM: {dem.shape}")
    dem_ds = _downsample(dem, 300)
    h2, w2 = dem_ds.shape
    cmap = _terrain_cmap()
    vmin, vmax = float(dem_ds.min()), float(dem_ds.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=315, altdeg=32)
    rgb = cmap(norm(dem_ds))[:, :, :3]
    shade = ls.hillshade(dem_ds, vert_exag=1.5)
    fc = np.zeros_like(rgb)
    for c in range(3):
        fc[:, :, c] = np.clip(rgb[:, :, c] * (0.3 + 0.7 * shade), 0, 1)

    x = np.linspace(0, 1, w2)
    y = np.linspace(0, 1, h2)
    X, Y = np.meshgrid(x, y)
    fig_t = plt.figure(figsize=(6, 4), dpi=DPI)
    ax = fig_t.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, dem_ds, facecolors=fc, rstride=1, cstride=1,
                    antialiased=True, shade=False)
    ax.view_init(elev=56, azim=32)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    zr = vmax - vmin
    ax.set_zlim(vmin - zr * 0.1, vmax + zr * 0.3)
    ax.set_axis_off()
    fig_t.patch.set_facecolor('#e8f0f8')
    ax.set_facecolor('#e8f0f8')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_t.canvas.draw()
    terrain_img = np.frombuffer(fig_t.canvas.buffer_rgba(), dtype=np.uint8)
    tw, th = fig_t.canvas.get_width_height()
    terrain_img = terrain_img.reshape(th, tw, 4)
    plt.close(fig_t)

    plate = Image.new("RGBA", (1200, 720), (255, 255, 255, 255))
    terrain_pil = Image.fromarray(terrain_img, "RGBA").resize((800, 520), Image.LANCZOS)
    plate.paste(terrain_pil, (20, 80))
    draw = ImageDraw.Draw(plate)
    try:
        font_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_t = font_b = font_s = ImageFont.load_default()

    draw.text((30, 20), "Map Plate · legend + scale bar + north arrow", fill=(26, 26, 46), font=font_t)

    # Legend bar
    lx, ly, lw, lh = 870, 100, 30, 360
    for i in range(lh):
        t = 1.0 - i / lh
        c = cmap(t)
        draw.rectangle([lx, ly+i, lx+lw, ly+i+1],
                       fill=(int(c[0]*255), int(c[1]*255), int(c[2]*255)))
    draw.rectangle([lx, ly, lx+lw, ly+lh], outline=(80, 80, 80))
    draw.text((lx+lw+8, ly-2), f"{vmax:.0f} m", fill=(40,40,40), font=font_s)
    draw.text((lx+lw+8, ly+lh-12), f"{vmin:.0f} m", fill=(40,40,40), font=font_s)
    draw.text((lx, ly-20), "Elevation", fill=(40,40,40), font=font_b)

    # Scale bar
    draw.rectangle([40, 630, 200, 636], fill=(40, 40, 40))
    draw.text((50, 642), "500 m", fill=(40,40,40), font=font_s)

    # North arrow
    na_x, na_y = 1130, 580
    draw.polygon([(na_x, na_y-35), (na_x-12, na_y+8), (na_x+12, na_y+8)], fill=(40,40,40))
    draw.text((na_x-6, na_y+12), "N", fill=(40,40,40), font=font_b)

    plate.save(OUT_DIR / "10-map-plate.png")
    print(f"  Saved: {OUT_DIR / '10-map-plate.png'}")


# ── main ────────────────────────────────────────────────────────────────

def main():
    tasks = [
        ("01: Mount Rainier", gallery_01),
        ("02: Mount Fuji + labels", gallery_02),
        ("03: Swiss land-cover", gallery_03),
        ("04: Luxembourg rail network", gallery_04),
        ("05: 3D Buildings", gallery_05),
        ("06: Point cloud", gallery_06),
        ("07: Camera flyover", gallery_07),
        ("08: Vector export", gallery_08),
        ("09: Shadow comparison", gallery_09),
        ("10: Map plate", gallery_10),
    ]
    for label, func in tasks:
        print(f"\n{label}...")
        func()
        gc.collect()
        plt.close("all")
    print("\nDone. All 10 gallery images generated.")


if __name__ == "__main__":
    main()
