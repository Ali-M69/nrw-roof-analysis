"""Load and validate the YAML run configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Processing:
    resolution: float = 0.5
    min_building_height: float = 2.5
    min_single_return_ratio: float = 0.7
    min_building_area: float = 30.0
    ransac_distance: float = 0.15
    ransac_iterations: int = 400
    max_planes: int = 8
    flat_slope_deg: float = 10.0
    seed: int = 42


@dataclass
class Config:
    name: str
    crs: str
    east_km: int
    north_km: int
    aoi: tuple[float, float, float, float]
    data_dir: Path
    out_dir: Path
    processing: Processing = field(default_factory=Processing)

    @property
    def tile_key(self) -> str:
        """NRW tile key, for example '32_405_5758'."""
        return f"32_{self.east_km}_{self.north_km}"

    def validate(self) -> None:
        minx, miny, maxx, maxy = self.aoi
        if not (minx < maxx and miny < maxy):
            raise ValueError(f"AOI is not a valid box: {self.aoi}")
        tminx, tminy = self.east_km * 1000, self.north_km * 1000
        if not (tminx <= minx and maxx <= tminx + 1000 and tminy <= miny and maxy <= tminy + 1000):
            raise ValueError(f"AOI {self.aoi} is not inside tile {self.tile_key}")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        name=raw["name"],
        crs=raw.get("crs", "EPSG:25832"),
        east_km=int(raw["tile"]["east_km"]),
        north_km=int(raw["tile"]["north_km"]),
        aoi=tuple(float(v) for v in raw["aoi"]),
        data_dir=Path(raw.get("data_dir", "data/raw")),
        out_dir=Path(raw.get("out_dir", "outputs")),
        processing=Processing(**raw.get("processing", {})),
    )
    cfg.validate()
    return cfg
