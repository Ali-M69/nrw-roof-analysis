"""Surface models and building footprint extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from .grid import Grid, fill_gaps, grid_stat
from .pointcloud import Points


@dataclass
class Surfaces:
    dsm: np.ndarray            # highest point per cell
    dtm: np.ndarray            # terrain
    ndsm: np.ndarray           # normalised DSM = object height above terrain
    single_ratio: np.ndarray   # share of single-return pulses per cell


def build_surfaces(points: Points, grid: Grid) -> Surfaces:
    dsm = fill_gaps(grid_stat(grid, points.x, points.y, points.z, "max"), max_cells=4)
    g = points.ground
    dtm = fill_gaps(grid_stat(grid, points.x[g], points.y[g], points.z[g], "mean"), max_cells=200)
    ndsm = np.clip(dsm - dtm, 0, None)
    total = grid_stat(grid, points.x, points.y, None, "count")
    single = grid_stat(grid, points.x[points.n_returns == 1], points.y[points.n_returns == 1], None, "count")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total > 0, single / total, np.nan)
    ratio = fill_gaps(ratio.astype(np.float32), max_cells=4)
    return Surfaces(dsm, dtm, ndsm.astype(np.float32), ratio)


def building_mask(s: Surfaces, grid: Grid, min_height: float, min_single_ratio: float) -> np.ndarray:
    """Raised objects with mostly single returns are roofs; trees give many multiple returns."""
    ratio = ndimage.uniform_filter(np.nan_to_num(s.single_ratio, nan=0.0), size=5)
    mask = (np.nan_to_num(s.ndsm) > min_height) & (ratio >= min_single_ratio)
    k = max(1, int(round(1.0 / grid.res)))
    mask = ndimage.binary_opening(mask, iterations=k)
    mask = ndimage.binary_closing(mask, iterations=k)
    return ndimage.binary_fill_holes(mask)


def polygonize(mask: np.ndarray, grid: Grid, min_area: float) -> list[BaseGeometry]:
    """Raster mask to simplified footprint polygons (GDAL polygonize through rasterio)."""
    polys = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform):
        if val != 1:
            continue
        p = shape(geom).buffer(0).simplify(grid.res * 0.75, preserve_topology=True)
        if p.area >= min_area:
            polys.append(p)
    polys.sort(key=lambda p: (-p.centroid.y, p.centroid.x))
    return polys


def label_raster(polys: list[BaseGeometry], grid: Grid) -> np.ndarray:
    """Raster with building id (1..n) per cell, 0 outside buildings."""
    if not polys:
        return np.zeros(grid.shape, np.int32)
    return features.rasterize(((p, i + 1) for i, p in enumerate(polys)), out_shape=grid.shape,
                              transform=grid.transform, fill=0, dtype="int32")
