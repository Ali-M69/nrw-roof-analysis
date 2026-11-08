"""Roof plane segmentation (sequential RANSAC) and roof type classification."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray          # unit normal with positive z
    d: float                    # plane: normal . p + d = 0
    inliers: np.ndarray         # indices into the building point array
    slope_deg: float = 0.0
    aspect_deg: float | None = None   # downslope direction, degrees clockwise from north
    projected_area: float = 0.0
    true_area: float = 0.0


@dataclass
class Roof:
    planes: list[Plane] = field(default_factory=list)
    roof_type: str = "unknown"
    pv_area: float = 0.0
    unassigned_share: float = 0.0


def _fit_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c)


def ransac_planes(pts: np.ndarray, dist: float = 0.15, iterations: int = 400, max_planes: int = 8,
                  min_inliers: int | None = None, seed: int = 0) -> list[Plane]:
    """Extract planes one after another; each plane takes the largest consensus set left."""
    rng = np.random.default_rng(seed)
    if min_inliers is None:
        min_inliers = max(30, int(0.03 * len(pts)))
    centred = pts - pts.mean(axis=0)
    remaining = np.arange(len(pts))
    planes: list[Plane] = []
    while len(remaining) >= min_inliers and len(planes) < max_planes:
        cur = centred[remaining]
        best_count, best_mask = 0, None
        for _ in range(iterations):
            a, b, c = cur[rng.choice(len(cur), 3, replace=False)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n /= norm
            m = np.abs(cur @ n - n @ a) < dist
            cnt = int(m.sum())
            if cnt > best_count:
                best_count, best_mask = cnt, m
        if best_mask is None or best_count < min_inliers:
            break
        n, d = _fit_plane(cur[best_mask])          # least-squares refinement
        m = np.abs(cur @ n + d) < dist
        if m.sum() < min_inliers:
            break
        n_world_d = d - n @ pts.mean(axis=0)       # convert offset back to world coordinates
        planes.append(Plane(normal=n, d=float(n_world_d), inliers=remaining[m]))
        remaining = remaining[~m]
    return planes


def plane_geometry(plane: Plane) -> tuple[float, float | None]:
    nx, ny, nz = plane.normal
    slope = float(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))))
    if slope < 2.0:
        return slope, None
    # z decreases along +(nx, ny), so (nx, ny) is the downslope (facing) direction.
    aspect = float(np.degrees(np.arctan2(nx, ny)) % 360.0)
    return slope, aspect


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def classify(planes: list[Plane], flat_slope: float = 10.0) -> str:
    if not planes:
        return "unknown"
    total = sum(p.projected_area for p in planes) or 1.0
    # Ignore small planes (chimneys, dormers, noise) when naming the roof type.
    planes = [p for p in planes if p.projected_area >= 0.05 * total]
    pitched = [p for p in planes if p.slope_deg >= flat_slope]
    flat_share = sum(p.projected_area for p in planes if p.slope_deg < flat_slope) / total
    if not pitched or flat_share > 0.8:
        return "flat"
    if flat_share > 0.3:
        return "complex"
    if len(pitched) == 1:
        return "shed"
    if len(pitched) == 2:
        return "gable" if _angle_diff(pitched[0].aspect_deg, pitched[1].aspect_deg) > 150 else "complex"
    sectors = {int(((p.aspect_deg + 45) % 360) // 90) for p in pitched}
    if len(pitched) <= 4 and len(sectors) >= 3:
        return "hip"
    return "complex"


def pv_suitable(p: Plane, flat_slope: float = 10.0) -> bool:
    """Flat roofs, or pitched planes up to 60 degrees that face east, south or west."""
    if p.slope_deg < flat_slope:
        return True
    return p.slope_deg <= 60.0 and p.aspect_deg is not None and 90.0 <= p.aspect_deg <= 270.0


def analyse_roof(pts: np.ndarray, footprint_area: float, dist: float = 0.15, iterations: int = 400,
                 max_planes: int = 8, flat_slope: float = 10.0, seed: int = 0) -> Roof:
    """Segment the roof points of one building and describe each plane.

    Aerial LiDAR samples the plan view almost uniformly, so the projected area of a plane
    is estimated from its share of the roof points times the footprint area."""
    if len(pts) < 30:
        return Roof()
    planes = ransac_planes(pts, dist, iterations, max_planes, seed=seed)
    for p in planes:
        p.slope_deg, p.aspect_deg = plane_geometry(p)
        p.projected_area = footprint_area * len(p.inliers) / len(pts)
        p.true_area = p.projected_area / max(np.cos(np.radians(p.slope_deg)), 0.2)
    assigned = sum(len(p.inliers) for p in planes)
    roof = Roof(planes=planes, roof_type=classify(planes, flat_slope),
                unassigned_share=1.0 - assigned / len(pts))
    roof.pv_area = float(sum(p.true_area for p in planes if pv_suitable(p, flat_slope)))
    return roof
