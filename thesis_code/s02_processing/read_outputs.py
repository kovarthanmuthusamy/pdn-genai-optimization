"""Read eCADSTAR outputs: the spectrum CSV and the map file.

What: a spectrum CSV (``*PIPinZ*.csv``, ';'-separated, 4 header rows) holds the impedance at the 231
    sweep frequencies in its second column. A map file (``Z_<MHz>MHz.map``) lists triangles: a line
    "3" followed by three "x y z" vertex lines. create_heatmaps() averages z over repeated vertices
    (z floored at 0.01 ohm), interpolates it onto a 64 x 64 grid with a thin-plate-spline RBF
    (cubic + nearest fill if the RBF system is singular) and multiplies by the board mask.
Usage (import only):
    read_impedance_file(path) -> (231,) float32, or None if unreadable / wrong length
    create_heatmaps(map_path, mask_board) -> (2, 64, 64): map on the board, and the mask itself
    load_mask_board(path) -> mask array (cached)
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata

EXPECTED_IMP_LENGTH = 231
HEATMAP_GRID_SIZE = 64


def read_impedance_file(filepath):
    """Impedance values (231,) of a spectrum CSV, or None if unreadable or of another length."""
    try:
        data = pd.read_csv(filepath, delimiter=";", skiprows=4, usecols=[1], dtype=np.float32, engine="c")
        imp = data.values.flatten().astype(np.float32)
        if len(imp) != EXPECTED_IMP_LENGTH:
            return None
        return imp
    except Exception:
        return None


@lru_cache(maxsize=4)
def load_mask_board(path):
    return np.load(path)


def create_heatmaps(file_path, mask_board=None, grid_size=HEATMAP_GRID_SIZE):
    """(2, grid, grid) array: interpolated map times mask, and the mask (ones if none given)."""
    with open(file_path) as f:
        lines = f.readlines()
    data = []
    for i, line in enumerate(lines):
        if line.strip() == "3" and i + 3 < len(lines):
            try:
                for k in range(1, 4):
                    px, py, pz = map(float, lines[i + k].split())
                    data.append((px, py, max(pz, 0.01)))
            except ValueError:
                pass
    if not data:
        raise RuntimeError("No data loaded from MAP file.")

    data_arr = np.array(data, dtype=np.float32)
    x, y, z = data_arr[:, 0], data_arr[:, 1], data_arr[:, 2]
    scale, x_min, y_min = 1e-5, x.min(), y.min()
    x, y = (x - x_min) * scale, (y - y_min) * scale
    xi, yi = np.linspace(x.min(), x.max(), grid_size), np.linspace(y.min(), y.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    points = np.column_stack((x, y))
    # Triangles share vertices; the RBF needs unique points, so z is averaged at repeated (x, y).
    pts_unique, inverse = np.unique(points, axis=0, return_inverse=True)
    z_unique = np.array([z[inverse == i].mean() for i in range(len(pts_unique))], dtype=np.float32)
    try:
        rbf = RBFInterpolator(pts_unique, z_unique, kernel="thin_plate_spline", smoothing=0)
        Zi = rbf(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(grid_size, grid_size)
    except np.linalg.LinAlgError:
        Zi = griddata(points, z, (Xi, Yi), method="cubic")
        nan_mask = np.isnan(Zi)
        if np.any(nan_mask):
            Zi[nan_mask] = griddata(points, z, (Xi, Yi), method="nearest")[nan_mask]

    mask_board = np.ones_like(Zi, dtype=np.float32) if mask_board is None else mask_board.astype(np.float32)
    return np.stack([Zi * mask_board, mask_board], axis=0)
