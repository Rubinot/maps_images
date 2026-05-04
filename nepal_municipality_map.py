#!/usr/bin/env python3
"""
Nepal Local Administrative Unit Boundary Mapper
Plots municipality/gaunpalika boundaries and calculates area.
"""

import json
import math
import os
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system("pip install matplotlib numpy --break-system-packages -q")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import numpy as np

# ── Load GeoJSON ─────────────────────────────────────────────────────────────
GEOJSON_PATH = "localboundries_updated.json"

def load_data():
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["features"]

# ── Geodesic area (Shoelace on spherical coords → km²) ───────────────────────
EARTH_RADIUS_KM = 6371.0

def polygon_area_km2(ring):
    """
    Approximate geodesic area of a polygon ring using the spherical excess
    formula (suitable for small polygons like municipalities).
    """
    n = len(ring)
    if n < 3:
        return 0.0
    # Convert to radians
    lons = [math.radians(p[0]) for p in ring]
    lats = [math.radians(p[1]) for p in ring]
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += (lons[j] - lons[i]) * (2 + math.sin(lats[i]) + math.sin(lats[j]))
    area = abs(total) * EARTH_RADIUS_KM ** 2 / 2.0
    return area

def geometry_area_km2(geometry):
    """Handle Polygon and MultiPolygon."""
    gtype = geometry["type"]
    total = 0.0
    if gtype == "Polygon":
        outer = geometry["coordinates"][0]
        total += polygon_area_km2(outer)
        for hole in geometry["coordinates"][1:]:
            total -= polygon_area_km2(hole)
    elif gtype == "MultiPolygon":
        for poly in geometry["coordinates"]:
            outer = poly[0]
            total += polygon_area_km2(outer)
            for hole in poly[1:]:
                total -= polygon_area_km2(hole)
    return abs(total)

# ── Menu helpers ──────────────────────────────────────────────────────────────
def numbered_menu(title, items):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    for i, item in enumerate(items, 1):
        print(f"  {i:>3}. {item}")
    print(f"{'─'*50}")
    while True:
        try:
            choice = int(input(f"Enter number (1-{len(items)}): ").strip())
            if 1 <= choice <= len(items):
                return items[choice - 1]
            print(f"  ✗  Please enter a number between 1 and {len(items)}.")
        except ValueError:
            print("  ✗  Invalid input. Enter a number.")

# ── Plotting ──────────────────────────────────────────────────────────────────
def get_all_coords(geometry):
    """Return flat list of (lon, lat) from any geometry."""
    coords = []
    gtype = geometry["type"]
    if gtype == "Polygon":
        for ring in geometry["coordinates"]:
            coords.extend(ring)
    elif gtype == "MultiPolygon":
        for poly in geometry["coordinates"]:
            for ring in poly:
                coords.extend(ring)
    return coords

def plot_municipality(feature, area_km2, output_path):
    geometry = feature["geometry"]
    props    = feature["properties"]
    name     = props["GaPa_NaPa"]
    kind     = props["Type_GN"]
    district = props["DISTRICT"].title()
    province = props["Province"]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_facecolor("#E8F4FD")

    # Collect patches
    patches = []
    all_lons, all_lats = [], []

    gtype = geometry["type"]
    rings = []
    if gtype == "Polygon":
        rings = [geometry["coordinates"][0]]   # outer ring only for display
    elif gtype == "MultiPolygon":
        rings = [poly[0] for poly in geometry["coordinates"]]

    for ring in rings:
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        all_lons.extend(lons)
        all_lats.extend(lats)
        xy = np.array(list(zip(lons, lats)))
        patches.append(MplPolygon(xy, closed=True))

    # Colour based on type
    colour_map = {
        "Gaunpalika":      ("#4CAF50", "#2E7D32"),
        "Nagarpalika":     ("#2196F3", "#1565C0"),
        "Sub-Metropolitan":("#FF9800", "#E65100"),
        "Metropolitan":    ("#F44336", "#B71C1C"),
    }
    fill_color, edge_color = colour_map.get(kind, ("#9C27B0", "#4A148C"))

    col = PatchCollection(patches, facecolor=fill_color, edgecolor=edge_color,
                          linewidth=1.8, alpha=0.75, zorder=2)
    ax.add_collection(col)

    # Extent with padding
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)
    pad_lon = max((lon_max - lon_min) * 0.15, 0.005)
    pad_lat = max((lat_max - lat_min) * 0.15, 0.005)
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)

    # Centroid marker
    cx = (lon_min + lon_max) / 2
    cy = (lat_min + lat_max) / 2
    ax.plot(cx, cy, "o", color=edge_color, markersize=7, zorder=3)

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.5, color="#BDBDBD", alpha=0.7, zorder=1)

    # Titles & labels
    kind_label = {
        "Gaunpalika": "Gaunpalika (Rural Municipality)",
        "Nagarpalika": "Nagarpalika (Municipality)",
        "Sub-Metropolitan": "Sub-Metropolitan City",
        "Metropolitan": "Metropolitan City",
    }.get(kind, kind)

    ax.set_title(f"{name}\n{kind_label}", fontsize=16, fontweight="bold",
                 color="#1A237E", pad=14)
    ax.set_xlabel("Longitude (°E)", fontsize=10, color="#37474F")
    ax.set_ylabel("Latitude (°N)",  fontsize=10, color="#37474F")
    ax.tick_params(labelsize=8, colors="#546E7A")
    for spine in ax.spines.values():
        spine.set_edgecolor("#90A4AE")

    # Info box
    info_lines = [
        f"Province : {province}",
        f"District : {district}",
        f"Type     : {kind_label}",
        f"Area     : {area_km2:.2f} km²",
    ]
    info_text = "\n".join(info_lines)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#90A4AE", alpha=0.9),
            color="#1A237E", zorder=4)

    # North arrow
    ax.annotate("N", xy=(0.97, 0.12), xytext=(0.97, 0.06),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=12, fontweight="bold", ha="center", color="#1A237E",
                arrowprops=dict(arrowstyle="-|>", color="#1A237E", lw=2))

    # Scale note
    ax.text(0.98, 0.01, "Coordinate Reference: WGS 84",
            transform=ax.transAxes, fontsize=7, ha="right",
            color="#78909C", style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✔  Map saved to: {output_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═"*50)
    print("  🗺   Nepal Local Administrative Unit Mapper")
    print("═"*50)

    features = load_data()

    # Build lookup structures
    prov_dist   = defaultdict(set)
    dist_mun    = defaultdict(set)
    lookup      = {}   # (province, district, mun) → feature

    for f in features:
        p  = f["properties"]["Province"]
        d  = f["properties"]["DISTRICT"]
        m  = f["properties"]["GaPa_NaPa"]
        prov_dist[p].add(d)
        dist_mun[(p, d)].add(m)
        lookup[(p, d, m)] = f

    # Step 1: Province
    provinces = sorted(prov_dist.keys())
    province  = numbered_menu("Select a Province", provinces)

    # Step 2: District
    districts = sorted(prov_dist[province])
    district  = numbered_menu(f"Select a District in {province}", districts)

    # Step 3: Municipality / Gaunpalika
    muns = sorted(dist_mun[(province, district)])
    mun  = numbered_menu(f"Select a Municipality / Gaunpalika in {district}", muns)

    # Fetch & process
    feature  = lookup[(province, district, mun)]
    area_km2 = geometry_area_km2(feature["geometry"])
    kind     = feature["properties"]["Type_GN"]

    print(f"\n{'─'*50}")
    print(f"  Selected  : {mun}")
    print(f"  Type      : {kind}")
    print(f"  District  : {district.title()}")
    print(f"  Province  : {province}")
    print(f"  Area      : {area_km2:.2f} km²")
    print(f"{'─'*50}")

    # Output path — save next to the script
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    safe_name   = mun.replace(" ", "_").replace("/", "-")
    output_path = os.path.join(script_dir, f"{safe_name}_boundary.png")

    plot_municipality(feature, area_km2, output_path)
    print("\n  Done! Open the image to view the boundary map.\n")

if __name__ == "__main__":
    main()
