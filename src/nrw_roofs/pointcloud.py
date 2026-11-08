"""Read the LiDAR tile, classify ground and compute height above ground.

Main path: a PDAL pipeline (crop, outlier removal, SMRF ground filter, height above ground,
DSM/DTM GeoTIFFs with writers.gdal).
Fallback path (when PDAL is not installed): laspy + a morphological ground filter in NumPy,
so the rest of the workflow and the tests run on any machine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .grid import Grid, fill_gaps, grid_stat, sample


@dataclass
class Points:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    n_returns: np.ndarray        # NumberOfReturns of the pulse
    ground: np.ndarray           # bool
    hag: np.ndarray              # height above ground, metres

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, idx: np.ndarray) -> "Points":
        return Points(self.x[idx], self.y[idx], self.z[idx], self.n_returns[idx], self.ground[idx], self.hag[idx])

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, self.z])


def pdal_available() -> bool:
    try:
        import pdal  # noqa: F401
        return True
    except ImportError:
        return False


def pdal_pipeline(laz: Path, grid: Grid, out_dir: Path | None = None) -> list[dict]:
    """PDAL pipeline as a list of stages. Writers are optional and pass the points through."""
    minx, miny, maxx, maxy = grid.bounds
    stages: list[dict] = [
        {"type": "readers.las", "filename": str(laz)},
        {"type": "filters.crop", "bounds": f"([{minx},{maxx}],[{miny},{maxy}])"},
        # Some NRW tiles hold a few points with ReturnNumber = 0, which SMRF refuses. Drop them.
        {"type": "filters.range", "limits": "ReturnNumber[1:], NumberOfReturns[1:]"},
        # Re-classify from scratch so the result does not depend on the vendor classes.
        {"type": "filters.assign", "value": ["Classification = 0"]},
        {"type": "filters.outlier", "method": "statistical", "mean_k": 12, "multiplier": 2.5},
        {"type": "filters.smrf", "ignore": "Classification[7:7]", "slope": 0.2, "window": 18.0,
         "threshold": 0.45, "scalar": 1.2},
        {"type": "filters.hag_nn"},
    ]
    if out_dir is not None:
        common = {"resolution": grid.res, "origin_x": minx, "origin_y": miny,
                  "width": grid.width, "height": grid.height, "data_type": "float32",
                  "gdalopts": "COMPRESS=DEFLATE,TILED=YES"}
        stages += [
            {"type": "writers.gdal", "filename": str(out_dir / "dsm_pdal.tif"), "output_type": "max",
             "where": "Classification != 7", **common},
            {"type": "writers.gdal", "filename": str(out_dir / "dtm_pdal.tif"), "output_type": "idw",
             "where": "Classification == 2", **common},
        ]
    return stages


def read_with_pdal(laz: Path, grid: Grid, out_dir: Path | None = None) -> Points:
    import pdal

    pipe = pdal.Pipeline(json.dumps(pdal_pipeline(laz, grid, out_dir)))
    pipe.execute()
    a = pipe.arrays[0]
    keep = a["Classification"] != 7  # drop outliers
    a = a[keep]
    return Points(
        x=a["X"].astype(np.float64), y=a["Y"].astype(np.float64), z=a["Z"].astype(np.float64),
        n_returns=a["NumberOfReturns"].astype(np.int16),
        ground=(a["Classification"] == 2),
        hag=a["HeightAboveGround"].astype(np.float32),
    )


def read_with_laspy(laz: Path, grid: Grid) -> Points:
    import laspy

    las = laspy.read(laz)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    minx, miny, maxx, maxy = grid.bounds
    m = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)
    n = len(x[m])
    return Points(x[m], y[m], z[m], np.asarray(las.number_of_returns)[m].astype(np.int16),
                  np.zeros(n, bool), np.zeros(n, np.float32))


def morphological_ground(points: Points, grid: Grid, window_m: float = 30.0,
                         threshold: float = 0.5) -> np.ndarray:
    """Simple ground filter: lowest point per cell, then a grey opening removes objects
    smaller than the window (buildings, trees). Updates points.ground and points.hag and
    returns the terrain surface."""
    zmin = fill_gaps(grid_stat(grid, points.x, points.y, points.z, "min"))
    size = max(3, int(round(window_m / grid.res)) | 1)
    terrain = ndimage.grey_opening(np.nan_to_num(zmin, nan=np.nanmin(zmin)), size=(size, size))
    terrain = ndimage.uniform_filter(terrain, size=5)
    t = sample(grid, terrain, points.x, points.y)
    points.ground = (points.z - t) < threshold
    points.hag = (points.z - t).astype(np.float32)
    return terrain.astype(np.float32)


def load_points(laz: Path, grid: Grid, out_dir: Path | None = None, force_fallback: bool = False) -> tuple[Points, str]:
    """Read points and classify ground. Returns the points and the method name used."""
    if not force_fallback and pdal_available():
        return read_with_pdal(laz, grid, out_dir), "pdal"
    pts = read_with_laspy(laz, grid)
    morphological_ground(pts, grid)
    return pts, "laspy+morphology"
