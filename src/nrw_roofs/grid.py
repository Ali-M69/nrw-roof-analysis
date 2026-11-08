"""Regular raster grid helpers: bin points into cells and write GeoTIFFs with GDAL (rasterio)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.transform import Affine, from_origin


@dataclass(frozen=True)
class Grid:
    minx: float
    miny: float
    maxx: float
    maxy: float
    res: float

    @property
    def width(self) -> int:
        return int(np.ceil((self.maxx - self.minx) / self.res))

    @property
    def height(self) -> int:
        return int(np.ceil((self.maxy - self.miny) / self.res))

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def transform(self) -> Affine:
        return from_origin(self.minx, self.maxy, self.res, self.res)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.minx, self.miny, self.maxx, self.maxy

    def cells(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row, column and a validity mask for each point."""
        col = np.floor((x - self.minx) / self.res).astype(np.int64)
        row = np.floor((self.maxy - y) / self.res).astype(np.int64)
        ok = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return row, col, ok


def grid_stat(grid: Grid, x: np.ndarray, y: np.ndarray, v: np.ndarray | None, stat: str) -> np.ndarray:
    """Per-cell statistic of point values. Empty cells are NaN (0 for 'count')."""
    row, col, ok = grid.cells(x, y)
    idx = row[ok] * grid.width + col[ok]
    n = grid.width * grid.height
    if stat == "count":
        return np.bincount(idx, minlength=n).reshape(grid.shape).astype(np.float32)
    vals = np.asarray(v, dtype=np.float64)[ok]
    if stat == "max":
        out = np.full(n, -np.inf)
        np.maximum.at(out, idx, vals)
        out[np.isneginf(out)] = np.nan
    elif stat == "min":
        out = np.full(n, np.inf)
        np.minimum.at(out, idx, vals)
        out[np.isposinf(out)] = np.nan
    elif stat == "mean":
        s = np.bincount(idx, weights=vals, minlength=n)
        c = np.bincount(idx, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(c > 0, s / c, np.nan)
    else:
        raise ValueError(f"Unknown statistic: {stat}")
    return out.reshape(grid.shape).astype(np.float32)


def fill_gaps(arr: np.ndarray, max_cells: float = 100.0) -> np.ndarray:
    """Fill NaN cells by inverse-distance interpolation (GDAL FillNodata)."""
    valid = np.isfinite(arr)
    if valid.all() or not valid.any():
        return arr
    filled = fillnodata(np.where(valid, arr, 0).astype(np.float32), mask=valid.astype(np.uint8),
                        max_search_distance=max_cells, smoothing_iterations=0)
    still_empty = ~valid & (filled == 0)
    return np.where(still_empty, np.nan, filled).astype(np.float32)


def sample(grid: Grid, raster: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-cell value of a raster at point locations (NaN outside)."""
    row, col, ok = grid.cells(x, y)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[ok] = raster[row[ok], col[ok]]
    return out


def write_geotiff(path: Path, arr: np.ndarray, grid: Grid, crs: str, nodata: float | None = -9999.0) -> Path:
    """Write a tiled, compressed GeoTIFF with overviews (GDAL GTiff driver)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr)
    if nodata is not None and np.issubdtype(data.dtype, np.floating):
        data = np.where(np.isfinite(data), data, nodata)
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype=data.dtype.name,
                   crs=crs, transform=grid.transform, nodata=nodata, tiled=True, blockxsize=256,
                   blockysize=256, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        factors = [f for f in (2, 4, 8) if min(grid.width, grid.height) // f >= 16]
        if factors:
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
    return path
