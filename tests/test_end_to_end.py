"""Full run on a synthetic LAZ tile with known buildings (no network, no PDAL needed)."""
import json

import pytest

from nrw_roofs.config import Config, Processing
from nrw_roofs.pipeline import run
from nrw_roofs.synthetic import BUILDINGS, bounds, make_scene, write_laz


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    laz = write_laz(tmp / "synthetic.laz", make_scene(density=10.0, seed=1))
    cfg = Config(name="synthetic", crs="EPSG:25832", east_km=405, north_km=5758, aoi=bounds(),
                 data_dir=tmp, out_dir=tmp / "out", processing=Processing(resolution=0.5))
    return run(cfg, laz, force_fallback=True), cfg


def test_finds_the_three_buildings_and_not_the_tree(result):
    res, _ = result
    assert len(res["records"]) == 3


def test_roof_types_areas_and_heights(result):
    res, _ = result
    by_type = {r["roof_type"]: r for r in res["records"]}
    assert set(by_type) == {"gable", "hip", "flat"}
    for truth in BUILDINGS:
        r = by_type[truth.roof_type]
        assert abs(r["footprint_m2"] - truth.area) / truth.area < 0.15, (truth.name, r["footprint_m2"])
    flat = by_type["flat"]
    # The roof is level but the terrain rises 2 cm/m in x and 1 cm/m in y under it,
    # so the median height above ground is 9.0 - 0.02*12 - 0.01*9 = 8.67 m.
    assert abs(flat["height_median_m"] - 8.67) < 0.15
    gable = by_type["gable"]
    assert abs(gable["main_slope_deg"] - 35) < 3


def test_outputs_are_written(result):
    _, cfg = result
    out = cfg.out_dir
    for f in ("dsm.tif", "dtm.tif", "ndsm.tif", "buildings.tif", "buildings.geojson",
              "buildings.csv", "summary.md", "overview.png"):
        assert (out / f).exists(), f
    gj = json.loads((out / "buildings.geojson").read_text())
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0][0]
    assert 7.5 < lon < 7.8 and 51.9 < lat < 52.1   # Münster, WGS84
