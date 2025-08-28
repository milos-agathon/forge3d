"""
City2Graph — end-to-end urban morphology + centrality analysis for any city.

Usage:
    python city2graph.py --city "Amsterdam, Netherlands" --radius 1200 --outdir outputs
"""

from __future__ import annotations
import os, math, warnings, argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
import networkx as nx
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim

# --- Robust import to avoid self-shadowing if user named the script city2graph.py
try:
    import city2graph as c2g  # real package
except Exception as e:
    raise SystemExit(
        "Could not import the 'city2graph' package. "
        "Make sure this file is NOT named 'city2graph.py' and install the package:\n"
        "    pip install city2graph geopandas shapely pyproj fiona contextily networkx geopy matplotlib\n\n"
        f"Original import error: {e}"
    )

# ----------------------------
# Helpers
# ----------------------------

def slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")

def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    north = lat >= 0
    return 32600 + zone if north else 32700 + zone

def geocode_city_bbox(city: str) -> tuple[float, float, float, float, float, float]:
    geocoder = Nominatim(user_agent="city2graph-demo")
    loc = geocoder.geocode(city, addressdetails=False, exactly_one=True, timeout=20)
    if not loc or not loc.raw.get("boundingbox"):
        raise RuntimeError(f"Could not geocode city name: {city}")
    south, north, west, east = map(float, loc.raw["boundingbox"])
    center_lon, center_lat = float(loc.longitude), float(loc.latitude)
    return west, south, east, north, center_lon, center_lat

def ensure_projected(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(epsg=epsg)

def add_segment_bearings(segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tmp = segments_gdf.to_crs(4326)
    def bearing_deg(ls: LineString) -> float:
        try:
            x0, y0 = ls.coords[0]
            x1, y1 = ls.coords[-1]
            # Initial bearing on a sphere (degrees)
            angle = math.degrees(math.atan2(
                math.sin(math.radians(x1 - x0)) * math.cos(math.radians(y1)),
                math.cos(math.radians(y0))*math.sin(math.radians(y1)) -
                math.sin(math.radians(y0))*math.cos(math.radians(y1))*math.cos(math.radians(x1 - x0))
            ))
            angle = (angle + 360) % 360
            return angle if angle <= 180 else angle - 180
        except Exception:
            return np.nan
    tmp["bearing"] = tmp.geometry.apply(bearing_deg)
    segments_gdf["bearing"] = tmp["bearing"].values
    return segments_gdf

# ----------------------------
# Data pipeline
# ----------------------------

def fetch_overture_data(bbox, outdir: Path, prefix: str):
    outdir.mkdir(parents=True, exist_ok=True)
    data = c2g.load_overture_data(
        area=[bbox[0], bbox[1], bbox[2], bbox[3]],
        types=["building", "segment", "connector"],
        output_dir=str(outdir),
        prefix=prefix,
        save_to_file=True,
        return_data=True,
    )
    return data["building"], data["segment"], data["connector"]

def prep_segments(segments_gdf: gpd.GeoDataFrame, connectors_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    seg = segments_gdf.copy()
    # Only filter if that schema exists and contains "road"
    if "subtype" in seg.columns and "road" in set(seg["subtype"].dropna().unique()):
        seg = seg[seg["subtype"] == "road"].copy()
    seg = c2g.process_overture_segments(
        segments_gdf=seg,
        get_barriers=True,
        connectors_gdf=connectors_gdf
    )
    return seg

def build_morphology(buildings_gdf, segments_gdf, center_lon, center_lat, epsg: int, radius_m: int):
    buildings = ensure_projected(buildings_gdf, epsg)
    segments = ensure_projected(segments_gdf, epsg)
    center_pt = gpd.GeoSeries([Point(center_lon, center_lat)], crs=4326).to_crs(epsg=epsg)

    nodes, edges = c2g.morphological_graph(
        buildings_gdf=buildings,
        segments_gdf=segments,
        center_point=center_pt,
        distance=float(radius_m),
        clipping_buffer=float(radius_m * 0.6),
        primary_barrier_col="barrier_geometry",
        contiguity="queen",
        keep_buildings=True
    )
    return nodes, edges, center_pt

# ----------------------------
# Analysis helpers
# ----------------------------

def pick_node_layer(nodes: dict, candidates: list[str]) -> gpd.GeoDataFrame:
    for c in candidates:
        if c in nodes:
            return nodes[c]
    # fallback: first layer
    return next(iter(nodes.values()))

def pick_public_edges(edges: dict) -> gpd.GeoDataFrame:
    # Prefer known tuple keys
    preferred = [
        ("public", "connected_to", "public"),
        ("public", "adjacent_to", "public"),
        ("street", "connected_to", "street"),
    ]
    for key in preferred:
        if key in edges:
            return edges[key]
    # Otherwise any key that looks like public↔public
    for key, gdf in edges.items():
        try:
            if isinstance(key, tuple) and all("public" in str(k) for k in key):
                return gdf
        except Exception:
            pass
    # Fallback: first edge layer
    return next(iter(edges.values()))

def compute_public_centrality(public_nodes: gpd.GeoDataFrame,
                              public_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    G = nx.Graph()
    # Add nodes
    for nid in public_nodes.index:
        G.add_node(nid)

    # Detect edge endpoints
    u_col, v_col = None, None
    for cand_u, cand_v in [
        ("from_public_id", "to_public_id"),
        ("from_id", "to_id"),
        ("u", "v"),
        ("from", "to"),
    ]:
        if cand_u in public_edges.columns and cand_v in public_edges.columns:
            u_col, v_col = cand_u, cand_v
            break

    if isinstance(public_edges.index, pd.MultiIndex) and len(public_edges.index.names) >= 2:
        for (u, v), _ in public_edges.iterrows():
            if u != v:
                G.add_edge(u, v)
    elif u_col and v_col:
        for _, row in public_edges.iterrows():
            u, v = row[u_col], row[v_col]
            if pd.notna(u) and pd.notna(v) and u != v:
                G.add_edge(u, v)
    else:
        # As a last resort, try geometry endpoints (not ideal but better than failing)
        for _, row in public_edges.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                u = hash(geom.coords[0])
                v = hash(geom.coords[-1])
                if u != v:
                    G.add_edge(u, v)

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    cls = nx.closeness_centrality(G)

    out = public_nodes.copy()
    out["deg_c"] = out.index.map(deg.get).astype(float)
    out["betw_c"] = out.index.map(btw.get).astype(float)
    out["close_c"] = out.index.map(cls.get).astype(float)
    for col in ["deg_c", "betw_c", "close_c"]:
        vals = out[col].fillna(0)
        out[col + "_z"] = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-9)
    return out

# ----------------------------
# Visualization
# ----------------------------

def plot_morphology_map(nodes: dict, edges: dict, center_pt_geom, epsg: int, title: str, out_path: Path):
    priv = pick_node_layer(nodes, ["private", "parcel", "plots", "cells"])
    pub  = pick_node_layer(nodes, ["public", "street", "segments"])
    e_pp = edges.get(("private", "touched_to", "private"), None)
    e_pu = edges.get(("private", "faced_to", "public"), None)
    e_uu = edges.get(("public", "connected_to", "public"), pick_public_edges(edges))

    fig, ax = plt.subplots(figsize=(12, 12))
    # base layers
    priv.plot(ax=ax, color="#bcd7ff", edgecolor="#99bde6", linewidth=0.2, alpha=0.25)
    if "building_geometry" in priv.columns:
        bg = gpd.GeoDataFrame(geometry=gpd.GeoSeries(priv["building_geometry"], crs=priv.crs))
        bg.plot(ax=ax, color="#dddddd", edgecolor="#bbbbbb", linewidth=0.3, alpha=0.8)
    pub.plot(ax=ax, color="#333333", linewidth=0.7, alpha=0.7)

    # graph edges (only if present)
    if e_pp is not None and len(e_pp):
        e_pp.plot(ax=ax, color="#b22222", linewidth=1.2, alpha=0.8)
    if e_uu is not None and len(e_uu):
        e_uu.plot(ax=ax, color="#0044ff", linewidth=0.9, alpha=0.8)
    if e_pu is not None and len(e_pu):
        e_pu.plot(ax=ax, color="#7B68EE", linewidth=0.9, alpha=0.7, linestyle="--")

    # center
    ax.scatter(center_pt_geom.x, center_pt_geom.y, s=160, c="black", marker="*", zorder=10)
    ax.set_title(title, fontsize=15)
    ax.set_axis_off()

    # make sure extent is set before adding basemap
    xmin, ymin, xmax, ymax = pub.total_bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, crs=f"EPSG:{epsg}", source=cx.providers.CartoDB.Positron, alpha=1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_centrality_map(public_nodes: gpd.GeoDataFrame, epsg: int, title: str, out_path: Path):
    g = public_nodes.to_crs(3857)
    fig, ax = plt.subplots(figsize=(12, 12))
    v = g["betw_c"].fillna(0)
    lw = 0.5 + 6 * (v / (v.max() + 1e-9))
    g.plot(ax=ax, linewidth=lw, color="#2b8cbe", alpha=0.9)
    ax.set_title(title, fontsize=15)
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = g.total_bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Positron, alpha=1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_orientation_hist(segments_gdf: gpd.GeoDataFrame, title: str, out_path: Path):
    if "bearing" not in segments_gdf.columns:
        return
    seg = segments_gdf.dropna(subset=["bearing"]).copy()
    bins = np.linspace(0, 180, 37)  # 5° bins
    counts, edges = np.histogram(seg["bearing"].values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(centers, counts, width=5, align="center")
    ax.set_xlabel("Street orientation (degrees, 0..180)")
    ax.set_ylabel("Count of segments")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# ----------------------------
# Save GIS outputs
# ----------------------------

def export_gpkg(nodes: dict, edges: dict, out_path: Path):
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()

    nodes["private"].to_file(out_path, layer="private_nodes", driver="GPKG")
    nodes["public"].to_file(out_path, layer="public_nodes", driver="GPKG")

    for key, gdf in edges.items():
        layer = f"edge_{str(key[0]).replace('-', '_')}_{str(key[1]).replace('-', '_')}_{str(key[2]).replace('-', '_')}"
        if isinstance(gdf.index, pd.MultiIndex):
            gdf = gdf.reset_index()
        gdf.to_file(out_path, layer=layer, driver="GPKG")

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="City2Graph morphological analysis for any city.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--city", type=str, default="Amsterdam, Netherlands",
                        help="City name (geocoded with Nominatim)")
    parser.add_argument("--radius", type=int, default=1000,
                        help="Radius in meters for graph cut around city center")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Output directory")
    args = parser.parse_args()

    city = args.city
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = slugify(city)

    print(f"🔎 Geocoding: {city}")
    minx, miny, maxx, maxy, clon, clat = geocode_city_bbox(city)
    bbox = (minx, miny, maxx, maxy)
    epsg = utm_epsg_from_lonlat(clon, clat)
    print(f"📦 BBox: {bbox}  | EPSG:{epsg}")

    print("⬇️  Downloading Overture Maps layers (building, segment, connector)…")
    buildings, segments, connectors = fetch_overture_data(bbox, outdir, prefix=f"{slug}_")

    print("🛣  Processing road segments and barriers…")
    segments = prep_segments(segments, connectors)

    print("🧭 Computing street bearings…")
    segments = add_segment_bearings(segments)

    print("🧱 Building morphological graph…")
    nodes, edges, center_pt = build_morphology(
        buildings, segments, clon, clat, epsg=epsg, radius_m=args.radius
    )
    print("✅ Morphological graph ready.")

    print("📈 Computing street centrality (dual graph)…")
    pub_nodes = pick_node_layer(nodes, ["public", "street", "segments"])
    pub_edges = pick_public_edges(edges)
    public_nodes = compute_public_centrality(pub_nodes, pub_edges)

    print("🖼  Rendering maps…")
    plot_morphology_map(
        nodes, edges, center_pt.iloc[0].geometry, epsg,
        title=f"{city} — Morphological Graph (r={args.radius} m)",
        out_path=outdir / f"{slug}_morphology.png"
    )
    plot_centrality_map(
        public_nodes,
        epsg,
        title=f"{city} — Street Betweenness Centrality",
        out_path=outdir / f"{slug}_centrality.png"
    )
    plot_orientation_hist(
        segments,
        title=f"{city} — Street Orientation Histogram (0..180°)",
        out_path=outdir / f"{slug}_orientation.png"
    )

    print("💾 Exporting GeoPackage…")
    export_gpkg(nodes, edges, outdir / f"{slug}_morphology.gpkg")

    print("\nDone ✅")
    print(f"• Morphology map:     {outdir / f'{slug}_morphology.png'}")
    print(f"• Centrality map:     {outdir / f'{slug}_centrality.png'}")
    print(f"• Orientation chart:  {outdir / f'{slug}_orientation.png'}")
    print(f"• GeoPackage:         {outdir / f'{slug}_morphology.gpkg'}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="The weights matrix is not fully connected")
    main()
