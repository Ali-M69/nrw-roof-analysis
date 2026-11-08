"""End-to-end run: LiDAR + orthophoto -> buildings with roof planes, heights and PV area."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .buildings import build_surfaces, building_mask, label_raster, polygonize
from .config import Config
from .grid import Grid, sample, write_geotiff
from .pointcloud import load_points
from .report import overview_figure, write_csv, write_geojson, write_summary
from .roofs import analyse_roof


def run(cfg: Config, lidar: Path, ortho: Path | None = None, force_fallback: bool = False,
        save_chips: bool = False) -> dict:
    t0 = time.perf_counter()
    p = cfg.processing
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid(*cfg.aoi, res=p.resolution)

    points, method = load_points(lidar, grid, out_dir=out, force_fallback=force_fallback)
    print(f"  {len(points):,} points, ground by {method}")

    surf = build_surfaces(points, grid)
    for name, arr in (("dsm", surf.dsm), ("dtm", surf.dtm), ("ndsm", surf.ndsm)):
        write_geotiff(out / f"{name}.tif", arr, grid, cfg.crs)

    mask = building_mask(surf, grid, p.min_building_height, p.min_single_return_ratio)
    polys = polygonize(mask, grid, p.min_building_area)
    labels = label_raster(polys, grid)
    write_geotiff(out / "buildings.tif", labels.astype(np.int32), grid, cfg.crs, nodata=0)
    print(f"  {len(polys)} buildings")

    # Assign every roof point (above the minimum height) to its building through the label raster.
    row, col, ok = grid.cells(points.x, points.y)
    bid = np.zeros(len(points), np.int32)
    bid[ok] = labels[row[ok], col[ok]]
    roof_pt = (bid > 0) & (points.hag > p.min_building_height) & ~points.ground

    records = []
    for i, poly in enumerate(polys, start=1):
        idx = np.flatnonzero(roof_pt & (bid == i))
        bpts = points.subset(idx)
        roof = analyse_roof(bpts.xyz, poly.area, p.ransac_distance, p.ransac_iterations, p.max_planes,
                            p.flat_slope_deg, seed=p.seed + i)
        main = max(roof.planes, key=lambda q: q.projected_area, default=None)
        c = poly.centroid
        records.append({
            "id": i,
            "geometry": poly,
            "centroid_e": round(c.x, 2), "centroid_n": round(c.y, 2),
            "footprint_m2": round(poly.area, 1),
            "ground_elev_m": round(float(np.nanmedian(sample(grid, surf.dtm, np.array([c.x]), np.array([c.y])))), 2),
            "height_max_m": round(float(bpts.hag.max()), 2) if len(bpts) else None,
            "height_median_m": round(float(np.median(bpts.hag)), 2) if len(bpts) else None,
            "roof_type": roof.roof_type,
            "n_planes": len(roof.planes),
            "main_slope_deg": round(main.slope_deg, 1) if main else None,
            "main_aspect_deg": round(main.aspect_deg, 0) if main and main.aspect_deg is not None else None,
            "pv_area_m2": round(roof.pv_area, 1),
            "unassigned_share": round(roof.unassigned_share, 3),
            "planes": [{"slope_deg": round(q.slope_deg, 1),
                        "aspect_deg": None if q.aspect_deg is None else round(q.aspect_deg, 0),
                        "area_m2": round(q.true_area, 1)} for q in roof.planes],
        })

    background = None
    if ortho is not None and Path(ortho).exists():
        from .ortho import read_rgb, save_chips as _save_chips
        try:
            background = read_rgb(Path(ortho), grid.bounds)
            if save_chips:
                n = _save_chips(Path(ortho), polys, out / "chips")
                print(f"  {n} image chips")
        except Exception as e:  # e.g. GDAL without the JPEG 2000 driver
            print(f"  WARNING: orthophoto not used ({e}). LiDAR results are still written.")
            background = None

    write_geojson(records, out / "buildings.geojson", cfg.crs)
    write_csv(records, out / "buildings.csv")
    overview_figure(records, surf.ndsm, grid.bounds, out / "overview.png", background, title=cfg.name)
    meta = {"name": cfg.name, "aoi": list(cfg.aoi), "crs": cfg.crs, "n_points": len(points),
            "method": method, "seconds": time.perf_counter() - t0}
    write_summary(records, meta, out / "summary.md")
    print(f"  done in {meta['seconds']:.1f} s -> {out}")
    return {"records": records, "meta": meta, "surfaces": surf, "mask": mask}
