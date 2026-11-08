"""Synthetic aerial LiDAR scene with known answers, for tests and for a demo without downloads.

The scene has a gable roof, a hip roof, a flat roof and a tree on gently sloping terrain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORIGIN = (405000.0, 5758400.0)   # inside NRW tile 32_405_5758 (Münster)
SIZE = 120.0


@dataclass(frozen=True)
class Truth:
    name: str
    roof_type: str
    footprint: tuple[float, float, float, float]   # x0, y0, x1, y1 (local metres)
    eave: float
    slope_deg: float

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.footprint
        return (x1 - x0) * (y1 - y0)


BUILDINGS = (
    Truth("gable", "gable", (15, 80, 31, 90), 6.0, 35.0),
    Truth("hip", "hip", (60, 75, 76, 89), 6.5, 30.0),
    Truth("flat", "flat", (20, 20, 44, 38), 9.0, 0.0),
)
TREE = (85.0, 30.0, 5.0)  # centre x, y, radius


def roof_height(b: Truth, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = b.footprint
    t = np.tan(np.radians(b.slope_deg))
    if b.roof_type == "flat":
        return np.full_like(x, b.eave)
    if b.roof_type == "gable":          # ridge along x, slopes face north and south
        yc = (y0 + y1) / 2
        return b.eave + ((y1 - y0) / 2 - np.abs(y - yc)) * t
    if b.roof_type == "hip":            # four planes, height from the nearest eave line
        d = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
        return b.eave + d * t
    raise ValueError(b.roof_type)


def make_scene(density: float = 10.0, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(SIZE * SIZE * density)
    x = rng.uniform(0, SIZE, n)
    y = rng.uniform(0, SIZE, n)
    ground = 60.0 + 0.02 * x + 0.01 * y
    z = ground + rng.normal(0, 0.03, n)
    nret = np.ones(n, np.int16)

    for b in BUILDINGS:
        x0, y0, x1, y1 = b.footprint
        m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        base = 60.0 + 0.02 * x0 + 0.01 * y0
        z[m] = base + roof_height(b, x[m], y[m]) + rng.normal(0, 0.03, m.sum())

    cx, cy, r = TREE
    dist = np.hypot(x - cx, y - cy)
    m = dist < r
    canopy = m & (rng.random(n) < 0.75)                 # most pulses hit leaves
    top = 12.0 * np.sqrt(1 - (dist[canopy] / r) ** 2)
    z[canopy] = ground[canopy] + rng.uniform(0.4, 1.0, canopy.sum()) * top + 2.0
    nret[m] = rng.integers(2, 5, m.sum())               # vegetation gives multiple returns

    return {"x": x + ORIGIN[0], "y": y + ORIGIN[1], "z": z, "n_returns": nret}


def write_laz(path: Path, scene: dict[str, np.ndarray]) -> Path:
    import laspy

    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([ORIGIN[0], ORIGIN[1], 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = scene["x"], scene["y"], scene["z"]
    las.number_of_returns = scene["n_returns"].astype(np.uint8)
    las.return_number = np.ones(len(scene["x"]), np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)
    return path


def bounds() -> tuple[float, float, float, float]:
    return ORIGIN[0], ORIGIN[1], ORIGIN[0] + SIZE, ORIGIN[1] + SIZE
