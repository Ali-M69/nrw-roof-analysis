"""Command line: `nrw-roofs download CONFIG` and `nrw-roofs run CONFIG`."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .download import download_tile, local_files


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nrw-roofs", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download the LiDAR tile and orthophoto for the config")
    d.add_argument("config")
    d.add_argument("--no-ortho", action="store_true")

    r = sub.add_parser("run", help="process the downloaded tile")
    r.add_argument("config")
    r.add_argument("--lidar", type=Path, help="LAZ file (default: from data_dir)")
    r.add_argument("--ortho", type=Path, help="orthophoto JP2 (default: from data_dir)")
    r.add_argument("--fallback", action="store_true", help="use laspy + NumPy ground filter, not PDAL")
    r.add_argument("--chips", action="store_true", help="save one orthophoto chip per building")

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "download":
        files = download_tile(cfg.tile_key, cfg.data_dir, with_ortho=not a.no_ortho)
        for k, v in files.items():
            print(f"{k}: {v}")
        return

    from .pipeline import run

    files = local_files(cfg.tile_key, cfg.data_dir)
    lidar = a.lidar or files.get("lidar")
    if lidar is None:
        raise SystemExit("No LiDAR file. Run `nrw-roofs download` first or pass --lidar.")
    run(cfg, lidar, a.ortho or files.get("ortho"), force_fallback=a.fallback, save_chips=a.chips)


if __name__ == "__main__":
    main()
