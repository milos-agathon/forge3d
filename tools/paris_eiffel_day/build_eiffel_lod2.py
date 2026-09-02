#!/usr/bin/env python3
"""Build the Eiffel-centred Paris building rings: BD TOPO footprints, LiDAR HD heights.

Paris has NO openly-licensed true LoD2 model -- no surveyed roof planes, no ridge
lines. The full source hunt is recorded in C:/tmp/citylod2/paris/RECON.md; the
short version is that IGN's own LoD2 demonstrator (batiment3d.ign.fr) is
visualization-only 3D Tiles with no stated licence, BDNB is LoD1, and the one
Paris product with real roof planes is commercial with no published terms.

So this builds the honest best available: IGN **BD TOPO(R) V3** `batiment`
footprints, with each building's roof altitude **re-measured from the IGN
LiDAR HD surface model** (p90 of the MNS inside the footprint -- p90 rather than
max, because max latches onto chimneys, antennae and overhanging trees). Both
are Etalab 2.0.

These are PRISM heights: one surveyed height per building, flat top. They are
real, LiDAR-accurate heights. They are not roof planes.

This is the Notre-Dame-centred pipeline in C:/tmp/citylod2/paris re-centred on
the Eiffel Tower, plus one rule that centre did not need (see LANDMARK below).

Usage:
    python tools/paris_eiffel_day/build_eiffel_lod2.py [--radius 2000] [--refresh]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import city_lod2_common as C  # noqa: E402

GML = "{http://www.opengis.net/gml/3.2}"
NS = "{http://BDTOPO_V3}"

EIFFEL_LON, EIFFEL_LAT = 2.2945, 48.8584
WORK = Path("C:/tmp/citylod2/paris_eiffel")
RAW = WORK / "raw"
OUT = WORK / "paris_eiffel_disc2000.npz"

WFS = "https://data.geopf.fr/wfs/ows"
WMS = "https://data.geopf.fr/wms-r/wms"
MNS_LAYER = "IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
MNT_LAYER = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"

PAGE = 5000
MNS_GRID = 4096
MIN_CELLS = 4
OUTLIER_M = 12.0

# LANDMARK rule -- the one thing the Notre-Dame disc never had to handle.
#
# The Eiffel Tower is in BD TOPO as nature="Tour, donjon" with hauteur 286.4 m,
# but its altitude_minimale_toit is ~58 m: that is the first platform, not the
# summit. Meanwhile the LiDAR p90 inside its footprint measures the PARK, not the
# tower, because an open lattice is mostly empty from above -- so both the
# authored roof altitude and the LiDAR refinement put the tower at a few tens of
# metres, and the existing +-12 m outlier guard would then reject the one value
# that is right.
#
# For these features the height that is actually surveyed and actually correct is
# `hauteur` above `altitude_minimale_sol`. Applied only to tall tower-class
# features so it cannot quietly inflate ordinary buildings.
LANDMARK_NATURES = {"Tour, donjon"}
LANDMARK_MIN_H = 60.0


def fetch_bdtopo(cx2154: float, cy2154: float, radius_m: float, *, refresh: bool) -> list[Path]:
    """Page BD TOPO V3 `batiment` over the disc bbox."""
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    existing = sorted(RAW.glob("bdtopo_batiment_*.gml"))
    if existing and not refresh:
        print(f"[BDTOPO] reusing {len(existing)} cached page(s)")
        return existing

    pad = 200.0
    x0, y0 = cx2154 - radius_m - pad, cy2154 - radius_m - pad
    x1, y1 = cx2154 + radius_m + pad, cy2154 + radius_m + pad
    sess = requests.Session()
    pages: list[Path] = []
    start = 0
    t0 = time.time()
    while True:
        resp = sess.get(
            WFS,
            timeout=300,
            params={
                "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                "TYPENAMES": "BDTOPO_V3:batiment", "SRSNAME": "EPSG:2154",
                "BBOX": f"{x0},{y0},{x1},{y1},urn:ogc:def:crs:EPSG::2154",
                "COUNT": str(PAGE), "STARTINDEX": str(start),
            },
        )
        resp.raise_for_status()
        body = resp.content
        n = body.count(b"<wfs:member>")
        path = RAW / f"bdtopo_batiment_{start:06d}.gml"
        path.write_bytes(body)
        pages.append(path)
        print(f"[BDTOPO] page {start}: {n} features, {len(body)/1e6:.1f} MB, {time.time()-t0:.0f}s")
        if n < PAGE:
            break
        start += PAGE
    return pages


def parse_rings(pages, cx: float, cy: float, radius_m: float):
    """GML -> ring arrays, keeping the attributes the landmark rule needs."""
    r2 = radius_m * radius_m
    chunks, offsets, kinds, holes, bidx = [], [0], [], [], []
    ring_landmark_z: list[float] = []
    total = nrings = nbld = 0
    n_seen = n_nodata = n_landmark = 0

    for page in pages:
        for _event, el in ET.iterparse(str(page), events=("end",)):
            if el.tag != f"{NS}batiment":
                continue
            n_seen += 1

            def num(tag):
                node = el.find(f"{NS}{tag}")
                if node is None or not node.text:
                    return None
                try:
                    return float(node.text)
                except ValueError:
                    return None

            def text(tag):
                node = el.find(f"{NS}{tag}")
                return (node.text or "").strip() if node is not None else ""

            min_sol = num("altitude_minimale_sol")
            hauteur = num("hauteur")
            nature = text("nature")

            landmark_z = np.nan
            is_landmark = (
                nature in LANDMARK_NATURES
                and hauteur is not None
                and hauteur >= LANDMARK_MIN_H
                and min_sol is not None
            )
            if is_landmark:
                landmark_z = float(min_sol) + float(hauteur)

            got = False
            for poly in el.iter(f"{GML}Polygon"):
                for tag, is_hole in ((f"{GML}exterior", 0), (f"{GML}interior", 1)):
                    for env in poly.findall(tag):
                        for ring in env.iter(f"{GML}LinearRing"):
                            for pos in ring.iter(f"{GML}posList"):
                                if not pos.text:
                                    continue
                                v = np.fromstring(pos.text, sep=" ", dtype=np.float64)
                                if v.size < 9 or v.size % 3:
                                    continue
                                c = v.reshape(-1, 3)
                                mx, my = float(c[:, 0].mean()), float(c[:, 1].mean())
                                if (mx - cx) ** 2 + (my - cy) ** 2 > r2:
                                    continue
                                gz = float(c[:, 2].mean())
                                # BD TOPO writes -1000 for "altitude unknown".
                                # Left in place it silently asserts a building
                                # where the survey has none.
                                if not np.isfinite(gz) or gz < -500.0:
                                    n_nodata += 1
                                    c[:, 2] = np.nan
                                chunks.append(c)
                                total += c.shape[0]
                                offsets.append(total)
                                kinds.append(2)      # roof: the only kind rasterised
                                holes.append(is_hole)
                                bidx.append(nbld)
                                ring_landmark_z.append(landmark_z)
                                nrings += 1
                                got = True
            if got:
                nbld += 1
                if is_landmark:
                    n_landmark += 1
            el.clear()

    print(
        f"[BDTOPO] {n_seen} scanned, {nbld} buildings in disc, {nrings} rings "
        f"(holes {int(np.sum(holes))}), {n_nodata} altitude-unknown rings, "
        f"{n_landmark} landmark tower(s)"
    )
    return (
        np.concatenate(chunks, axis=0),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(kinds, dtype=np.int8),
        np.asarray(holes, dtype=np.int8),
        np.asarray(bidx, dtype=np.int32),
        np.asarray(ring_landmark_z, dtype=np.float64),
    )


def refine_with_lidar(xyz, off, landmark_z, cx32, cy32, radius_m, *, refresh: bool):
    """Replace each ring's Z with the LiDAR HD p90 inside its footprint."""
    half = radius_m + 100.0
    mns = C.wms_dem(
        cx32, cy32, half, MNS_GRID, crs="EPSG:32631", layer=MNS_LAYER, url=WMS,
        cache=WORK / f"mns_lidar_{MNS_GRID}.npy",
    )
    terrain = C.wms_dem(
        cx32, cy32, half, MNS_GRID, crs="EPSG:32631", layer=MNT_LAYER, url=WMS,
        cache=WORK / f"terrain_lidargrid_{MNS_GRID}.npy",
    )
    print(f"[LiDAR] MNS {MNS_GRID}^2 over {2*half:.0f} m ({2*half/MNS_GRID:.2f} m/px), "
          f"z {mns.min():.1f}..{mns.max():.1f}")

    to_dst = Transformer.from_crs("EPSG:2154", "EPSG:32631", always_xy=True)
    scale = MNS_GRID / (2.0 * half)
    n = len(off) - 1
    old_z = np.array([np.nanmean(xyz[off[i]:off[i + 1], 2]) for i in range(n)])
    new_z = np.full(n, np.nan)
    n_small = 0

    for i in range(n):
        c = xyz[off[i]:off[i + 1]]
        ux, uy = to_dst.transform(c[:, 0], c[:, 1])
        px = (np.asarray(ux) - cx32 + half) * scale
        py = (half - (np.asarray(uy) - cy32)) * scale
        x0 = max(0, int(np.floor(px.min()))); x1 = min(MNS_GRID, int(np.ceil(px.max())) + 1)
        y0 = max(0, int(np.floor(py.min()))); y1 = min(MNS_GRID, int(np.ceil(py.max())) + 1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        m = Image.new("1", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon(
            [(float(a - x0), float(b - y0)) for a, b in zip(px, py)], fill=1, outline=1)
        mm = np.array(m, dtype=bool)
        if mm.sum() < MIN_CELLS:
            n_small += 1
            continue
        vals = mns[y0:y1, x0:x1][mm]
        vals = vals[np.isfinite(vals)]
        if vals.size < MIN_CELLS:
            continue
        new_z[i] = float(np.percentile(vals, 90))

    ok = np.isfinite(new_z)
    had_bd = np.isfinite(old_z)
    both = ok & had_bd
    dz = (new_z - old_z)[both]
    print(f"[LiDAR] sampled {ok.sum()}/{n} rings ({100*ok.mean():.1f}%); "
          f"{n_small} footprints under {MIN_CELLS} cells")
    print(f"[LiDAR] p90 minus BD TOPO roof Z: median {np.median(dz):+.2f} m, "
          f"absdiff median {np.median(np.abs(dz)):.2f} m; within 3 m "
          f"{100*np.mean(np.abs(dz) < 3.0):.1f}%")

    wild = both & (np.abs(new_z - old_z) > OUTLIER_M)
    print(f"[LiDAR] outliers >{OUTLIER_M:.0f} m from BD TOPO: {int(wild.sum())} "
          f"({100*wild.mean():.2f}%) -> keep BD TOPO Z")

    final = np.where(ok & ~wild, new_z, old_z)

    # Landmarks last, overriding both: see LANDMARK above.
    lm = np.isfinite(landmark_z)
    if lm.any():
        before = final[lm].copy()
        final[lm] = landmark_z[lm]
        print(f"[LANDMARK] {int(lm.sum())} ring(s) lifted to altitude_minimale_sol + "
              f"hauteur: z {np.nanmin(before):.1f}->{np.nanmin(final[lm]):.1f} .. "
              f"{np.nanmax(before):.1f}->{np.nanmax(final[lm]):.1f} m")

    keep = np.isfinite(final)
    gy, gx = None, None
    cen_x = np.array([xyz[off[i]:off[i + 1], 0].mean() for i in range(n)])
    cen_y = np.array([xyz[off[i]:off[i + 1], 1].mean() for i in range(n)])
    tux, tuy = to_dst.transform(cen_x, cen_y)
    cpx = np.clip(((np.asarray(tux) - cx32 + half) * scale).astype(int), 0, MNS_GRID - 1)
    cpy = np.clip(((half - (np.asarray(tuy) - cy32)) * scale).astype(int), 0, MNS_GRID - 1)
    above = final - terrain[cpy, cpx]
    print(f"[LiDAR] height above terrain: median {np.nanmedian(above[keep]):.1f} m, "
          f"p95 {np.nanpercentile(above[keep], 95):.1f} m, max {np.nanmax(above[keep]):.1f} m")
    return final, keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=2000.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    t2154 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    t32631 = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    cx, cy = t2154.transform(EIFFEL_LON, EIFFEL_LAT)
    cx32, cy32 = t32631.transform(EIFFEL_LON, EIFFEL_LAT)
    print(f"[Eiffel] centre EPSG:2154 {cx:.1f},{cy:.1f}  EPSG:32631 {cx32:.1f},{cy32:.1f}  "
          f"radius {args.radius:.0f} m")

    pages = fetch_bdtopo(cx, cy, args.radius, refresh=args.refresh)
    xyz, off, kinds, holes, bidx, landmark_z = parse_rings(pages, cx, cy, args.radius)
    final_z, keep = refine_with_lidar(xyz, off, landmark_z, cx32, cy32, args.radius,
                                      refresh=args.refresh)

    idx = np.where(keep)[0]
    parts = []
    for i in idx:
        c = xyz[off[i]:off[i + 1]].copy()
        c[:, 2] = final_z[i]
        parts.append(c)
    new_off = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    np.savez_compressed(
        OUT,
        xyz=np.concatenate(parts, axis=0),
        ring_offsets=new_off,
        ring_kind=kinds[idx],
        ring_is_hole=holes[idx],
        ring_building=bidx[idx],
        center=np.asarray([cx, cy]),
        radius=np.asarray([args.radius]),
        swap_xy=np.asarray([0]),
    )
    zf = np.concatenate(parts)[:, 2]
    print(f"[Eiffel] wrote {OUT}  {len(idx)} rings, "
          f"{len(np.unique(bidx[idx]))} buildings, z {zf.min():.1f}..{zf.max():.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
