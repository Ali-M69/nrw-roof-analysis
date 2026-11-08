"""Write results: GeoJSON (WGS84), CSV, a Markdown summary and an overview figure."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from rasterio.warp import transform_geom
from shapely.geometry import mapping

ROOF_COLOURS = {"flat": "#4c78a8", "gable": "#f58518", "hip": "#54a24b", "shed": "#b279a2",
                "complex": "#e45756", "unknown": "#9d9d9d"}


def write_geojson(records: list[dict], path: Path, src_crs: str) -> Path:
    feats = []
    for r in records:
        geom = transform_geom(src_crs, "EPSG:4326", mapping(r["geometry"]), precision=7)
        props = {k: v for k, v in r.items() if k not in ("geometry", "planes")}
        props["planes"] = r["planes"]
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1), encoding="utf-8")
    return path


CSV_FIELDS = ["id", "centroid_e", "centroid_n", "footprint_m2", "ground_elev_m", "height_max_m",
              "height_median_m", "roof_type", "n_planes", "main_slope_deg", "main_aspect_deg",
              "pv_area_m2", "unassigned_share"]


def write_csv(records: list[dict], path: Path) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    return path


def write_summary(records: list[dict], meta: dict, path: Path) -> Path:
    types: dict[str, int] = {}
    for r in records:
        types[r["roof_type"]] = types.get(r["roof_type"], 0) + 1
    lines = [f"# Results: {meta['name']}", "",
             f"- Area of interest: {meta['aoi']} ({meta['crs']})",
             f"- Points processed: {meta['n_points']:,}",
             f"- Ground classification: {meta['method']}",
             f"- Buildings found: {len(records)}",
             f"- Total footprint: {sum(r['footprint_m2'] for r in records):,.0f} m2",
             f"- PV-suitable roof area: {sum(r['pv_area_m2'] for r in records):,.0f} m2",
             f"- Run time: {meta['seconds']:.1f} s", "", "| Roof type | Buildings |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(types.items(), key=lambda kv: -kv[1])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def overview_figure(records: list[dict], ndsm: np.ndarray, extent: tuple, path: Path,
                    background: np.ndarray | None = None, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    minx, miny, maxx, maxy = extent
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    ax = axes[0]
    if background is not None:
        ax.imshow(background, extent=(minx, maxx, miny, maxy))
    else:
        ax.imshow(ndsm, extent=(minx, maxx, miny, maxy), cmap="Greys_r")
    for r in records:
        geoms = getattr(r["geometry"], "geoms", [r["geometry"]])
        for g in geoms:
            x, y = g.exterior.xy
            ax.fill(x, y, alpha=0.45, color=ROOF_COLOURS.get(r["roof_type"], "#999"))
            ax.plot(x, y, color="white", lw=0.6)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in ROOF_COLOURS.items()], loc="lower left", fontsize=8)
    ax.set_title("Buildings by roof type")
    im = axes[1].imshow(ndsm, extent=(minx, maxx, miny, maxy), cmap="viridis", vmin=0,
                        vmax=float(np.nanpercentile(ndsm, 99.5)) or 1.0)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="Height above ground (m)")
    axes[1].set_title("Normalised surface model (nDSM)")
    for a in axes:
        a.set_xlabel("E (m)")
        a.set_ylabel("N (m)")
        a.ticklabel_format(useOffset=False, style="plain")
    if title:
        fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
