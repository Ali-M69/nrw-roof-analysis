"""Download NRW open data tiles (LiDAR point cloud and 10 cm orthophoto).

Source: Geobasis NRW, https://www.opengeodata.nrw.de (licence: Datenlizenz Deutschland Zero 2.0).
Each product folder returns an XML listing of <file name=... size=... timestamp=...> entries.
"""
from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path

BASE = "https://www.opengeodata.nrw.de/produkte/geobasis"
LIDAR_DIR = f"{BASE}/hm/3dm_l_las/3dm_l_las/"
DOP_DIR = f"{BASE}/lusat/akt/dop/dop_jp2_f10/"

_FILE_RE = re.compile(r'<file\s+name="([^"]+)"(?:\s+size="(\d+)")?')


def parse_listing(xml_text: str) -> dict[str, int | None]:
    """Return {file name: size in bytes} from an opengeodata.nrw.de folder listing."""
    return {m.group(1): (int(m.group(2)) if m.group(2) else None) for m in _FILE_RE.finditer(xml_text)}


def lidar_name(tile_key: str) -> str:
    return f"3dm_{tile_key}_1_nw.laz"


def find_dop_name(listing: dict[str, int | None], tile_key: str) -> str:
    """Orthophoto names include the flight year, so pick the newest match for the tile."""
    prefix = f"dop10rgbi_{tile_key}_1_nw_"
    matches = sorted(n for n in listing if n.startswith(prefix) and n.endswith(".jp2"))
    if not matches:
        raise FileNotFoundError(f"No orthophoto for tile {tile_key} in the listing")
    return matches[-1]


def _fetch_text(url: str, timeout: int = 120) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https host)
        return r.read().decode("utf-8", errors="replace")


def _download(url: str, dest: Path, expected_size: int | None) -> Path:
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  already present: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f, length=1 << 20)
    if expected_size is not None and part.stat().st_size != expected_size:
        raise IOError(f"Size mismatch for {dest.name}: {part.stat().st_size} != {expected_size}")
    part.rename(dest)
    return dest


def download_tile(tile_key: str, data_dir: Path, with_ortho: bool = True) -> dict[str, Path]:
    """Download the LiDAR tile and (optionally) the newest orthophoto for one 1 km tile."""
    out: dict[str, Path] = {}
    lidar_listing = parse_listing(_fetch_text(LIDAR_DIR))
    name = lidar_name(tile_key)
    if name not in lidar_listing:
        raise FileNotFoundError(f"{name} is not in the NRW LiDAR listing")
    out["lidar"] = _download(LIDAR_DIR + name, data_dir / name, lidar_listing[name])

    if with_ortho:
        dop_listing = parse_listing(_fetch_text(DOP_DIR))
        dop = find_dop_name(dop_listing, tile_key)
        out["ortho"] = _download(DOP_DIR + dop, data_dir / dop, dop_listing[dop])
    return out


def local_files(tile_key: str, data_dir: Path) -> dict[str, Path]:
    """Find already downloaded files for a tile."""
    out: dict[str, Path] = {}
    laz = data_dir / lidar_name(tile_key)
    if laz.exists():
        out["lidar"] = laz
    dops = sorted(data_dir.glob(f"dop10rgbi_{tile_key}_1_nw_*.jp2"))
    if dops:
        out["ortho"] = dops[-1]
    return out
