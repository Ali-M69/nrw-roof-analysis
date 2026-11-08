import json
from pathlib import Path

import numpy as np
import pytest

from nrw_roofs.config import load_config
from nrw_roofs.download import find_dop_name, lidar_name, parse_listing
from nrw_roofs.grid import Grid, grid_stat
from nrw_roofs.pointcloud import pdal_pipeline
from nrw_roofs.roofs import analyse_roof, classify, plane_geometry, Plane
from nrw_roofs.synthetic import BUILDINGS, roof_height

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- grid
def test_grid_stats_and_bounds():
    g = Grid(0, 0, 4, 4, 1.0)
    x = np.array([0.5, 0.6, 3.5, 10.0])   # last point is outside the grid
    y = np.array([3.5, 3.4, 0.5, 0.5])
    z = np.array([1.0, 3.0, 5.0, 99.0])
    assert grid_stat(g, x, y, z, "max")[0, 0] == 3.0
    assert grid_stat(g, x, y, z, "min")[0, 0] == 1.0
    assert grid_stat(g, x, y, z, "mean")[0, 0] == 2.0
    assert grid_stat(g, x, y, z, "count").sum() == 3
    assert np.isnan(grid_stat(g, x, y, z, "max")[1, 1])


# ---------------------------------------------------------------- download helpers
LISTING = """<opengeodata>
<file name="3dm_32_405_5758_1_nw.laz" size="100" timestamp="2023-01-26T10:53:48"/>
<file name="dop10rgbi_32_405_5758_1_nw_2022.jp2" size="10"/>
<file name="dop10rgbi_32_405_5758_1_nw_2024.jp2" size="11"/>
<file name="dop10rgbi_32_406_5758_1_nw_2025.jp2" size="12"/>
</opengeodata>"""


def test_listing_and_names():
    files = parse_listing(LISTING)
    assert files["3dm_32_405_5758_1_nw.laz"] == 100
    assert lidar_name("32_405_5758") == "3dm_32_405_5758_1_nw.laz"
    assert find_dop_name(files, "32_405_5758") == "dop10rgbi_32_405_5758_1_nw_2024.jp2"
    with pytest.raises(FileNotFoundError):
        find_dop_name(files, "32_999_9999")


# ---------------------------------------------------------------- config
def test_config_loads_and_rejects_bad_aoi(tmp_path):
    cfg = load_config(ROOT / "config" / "muenster_kreuzviertel.yaml")
    assert cfg.tile_key == "32_405_5758"
    bad = (ROOT / "config" / "muenster_kreuzviertel.yaml").read_text().replace(
        "aoi: [405000, 5758400, 405500, 5758900]", "aoi: [404000, 5758400, 405500, 5758900]")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_config(p)


# ---------------------------------------------------------------- PDAL pipeline definition
def test_pdal_pipeline_definition(tmp_path):
    g = Grid(405000, 5758400, 405500, 5758900, 0.5)
    stages = pdal_pipeline(Path("tile.laz"), g, out_dir=tmp_path)
    types = [s["type"] for s in stages]
    assert types[:7] == ["readers.las", "filters.crop", "filters.range", "filters.assign",
                         "filters.outlier", "filters.smrf", "filters.hag_nn"]
    writers = [s for s in stages if s["type"] == "writers.gdal"]
    assert len(writers) == 2 and all(w["width"] == 1000 and w["height"] == 1000 for w in writers)
    json.dumps(stages)  # must be valid JSON for pdal.Pipeline


# ---------------------------------------------------------------- roofs
def _roof_points(name, density=12.0, seed=3):
    b = next(t for t in BUILDINGS if t.name == name)
    x0, y0, x1, y1 = b.footprint
    rng = np.random.default_rng(seed)
    n = int(b.area * density)
    x = rng.uniform(x0, x1, n)
    y = rng.uniform(y0, y1, n)
    z = roof_height(b, x, y) + rng.normal(0, 0.03, n)
    return b, np.column_stack([x, y, z])


def test_gable_roof():
    b, pts = _roof_points("gable")
    roof = analyse_roof(pts, b.area, seed=1)
    assert roof.roof_type == "gable"
    slopes = sorted(round(p.slope_deg) for p in roof.planes[:2])
    assert all(abs(s - 35) <= 2 for s in slopes)
    aspects = sorted(round(p.aspect_deg) % 360 for p in roof.planes[:2])
    assert min(abs(aspects[0] - 0), abs(aspects[0] - 360)) <= 5 and abs(aspects[1] - 180) <= 5
    # Only the south-facing half is PV-suitable: about half of the true roof area.
    true_area = b.area / np.cos(np.radians(35))
    assert abs(roof.pv_area - true_area / 2) / (true_area / 2) < 0.1


def test_hip_roof():
    b, pts = _roof_points("hip")
    roof = analyse_roof(pts, b.area, seed=1)
    assert roof.roof_type == "hip"
    assert len([p for p in roof.planes if p.slope_deg > 10]) == 4


def test_flat_roof():
    b, pts = _roof_points("flat")
    roof = analyse_roof(pts, b.area, seed=1)
    assert roof.roof_type == "flat"
    assert abs(roof.pv_area - b.area) / b.area < 0.05


def test_aspect_convention():
    # A plane that falls towards the east (z decreases with x) faces 90 degrees.
    n = np.array([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))])
    slope, aspect = plane_geometry(Plane(normal=n, d=0.0, inliers=np.array([])))
    assert abs(slope - 30) < 1e-6 and abs(aspect - 90) < 1e-6


def test_classify_empty():
    assert classify([]) == "unknown"
