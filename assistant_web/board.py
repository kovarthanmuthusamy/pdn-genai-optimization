"""Board geometry for the web page: slot positions, IC port, H-board outline and the map grid, in mm.

What: everything the page needs to draw the board of the thesis (Ch2, board_cad.png) and to lay the
    64 x 64 map under it. Coordinates are board millimetres with the origin at the lower left corner
    (y up), the same frame as ``common/inputs/slot_xy_mm.json`` and the eCADSTAR map mesh.
Usage: ``from assistant_web.board import geometry`` -> dict (JSON-ready). Import-only.
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.common.board import INPUTS

# The 64 x 64 map grid: every eCADSTAR .map file of this board has the same mesh, x from -1.761 to
# 301.765 mm and y from 0.097 to 319.598 mm over 64 evenly spread points (thesis_code s12 common.py).
MAP_X0, MAP_X1 = -1.761, 301.765
MAP_Y0, MAP_Y1 = 0.097, 319.598
N_PIX = 64
PIX_X = (MAP_X1 - MAP_X0) / (N_PIX - 1)
PIX_Y = (MAP_Y1 - MAP_Y0) / (N_PIX - 1)


def board_mask() -> np.ndarray:
    """(64, 64) bool, row 0 = bottom of the board."""
    return np.load(INPUTS / "binary_mask.npy").astype(bool)


def _edge_x(col: float) -> float:
    return MAP_X0 + col * PIX_X


def _edge_y(row: float) -> float:
    return MAP_Y0 + row * PIX_Y


def outline_mm(mask: np.ndarray, board_w: float, board_h: float) -> list[list[float]]:
    """The H-shaped outline as a closed polygon: the board rectangle minus the two notches of the mask
    (top and bottom, same columns). Notch edges sit halfway between an on-board and an off-board pixel."""
    full = np.flatnonzero(mask.all(1))
    top_row, bot_row = int(full.max()), int(full.min())        # last full rows above / below the notches
    notch = np.flatnonzero(~mask[top_row + 1])
    c0, c1 = int(notch.min()), int(notch.max())
    x0, x1 = _edge_x(c0 - 0.5), _edge_x(c1 + 0.5)
    y_top = _edge_y(top_row + 0.5)
    y_bot = _edge_y(bot_row - 0.5)
    w, h = float(board_w), float(board_h)
    pts = [(0, 0), (x0, 0), (x0, y_bot), (x1, y_bot), (x1, 0), (w, 0),
           (w, h), (x1, h), (x1, y_top), (x0, y_top), (x0, h), (0, h)]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]


def geometry() -> dict:
    j = json.loads((INPUTS / "slot_xy_mm.json").read_text())
    w, h = (float(v) for v in j["board_mm"])
    mask = board_mask()
    return {
        "board_mm": [w, h],
        "outline": outline_mm(mask, w, h),
        "ic_port": [float(v) for v in j["ic_port"]],
        "slots": [[float(x), float(y)] for x, y in j["slots"]],
        # the image of the map covers the pixel cells, i.e. half a pixel beyond the first/last centre
        "map_extent": [MAP_X0 - PIX_X / 2, MAP_X1 + PIX_X / 2, MAP_Y0 - PIX_Y / 2, MAP_Y1 + PIX_Y / 2],
        "mask": mask.astype(np.uint8).tolist(),
    }
