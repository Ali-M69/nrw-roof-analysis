"""Read the 10 cm NRW orthophoto (JPEG 2000, RGB + infrared) for the AOI with GDAL."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


def read_rgb(path: Path, bounds: tuple[float, float, float, float], max_size: int = 2000) -> np.ndarray:
    """RGB image (H, W, 3, uint8) of the bounds, downsampled so the long side is <= max_size."""
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        scale = max(1.0, max(win.width, win.height) / max_size)
        out_shape = (3, int(round(win.height / scale)), int(round(win.width / scale)))
        rgb = src.read([1, 2, 3], window=win, out_shape=out_shape, resampling=Resampling.average,
                       boundless=True, fill_value=0)
    return np.moveaxis(rgb, 0, -1).astype(np.uint8)


def save_chips(path: Path, polys, out_dir: Path, pad: float = 2.0, max_size: int = 512) -> int:
    """Save one RGB image chip per building, as training data for a later image model."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, p in enumerate(polys, start=1):
        minx, miny, maxx, maxy = p.bounds
        img = read_rgb(path, (minx - pad, miny - pad, maxx + pad, maxy + pad), max_size=max_size)
        if img.size:
            plt.imsave(out_dir / f"building_{i:04d}.png", img)
            n += 1
    return n
