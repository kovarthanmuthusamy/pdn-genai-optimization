---
title: ic_inverse_mask_creator
type: code
path: Rules/ic_inverse_mask_creator.py
group: Rules
loc: 132
tags: [code, Rules]
---

# ic_inverse_mask_creator

**Source:** `Rules/ic_inverse_mask_creator.py` · 132 lines

## Constants

| Name | Value |
|------|-------|
| `IC_X` | `27.595` |
| `IC_Y` | `101.975` |
| `PAD_RADIUS` | `0` |
| `GRID_SIZE` | `64` |
| `OUTPUT_FILE` | `os.path.join(os.path.dirname(__file__), '../configs/ic_inverse_mask.npy')` |

## Functions

- **`make_ic_inverse_mask(ic_x, ic_y, pad_radius=PAD_RADIUS, grid_size=GRID_SIZE, output_file=OUTPUT_FILE)`** — Create an inverse mask (all zeros) with a 1 at the grid cell
- **`visualise(inverse_mask, row, col, board_mask_path='../configs/binary_mask.npy')`** — Overlay the inverse mask on the board mask for a quick sanity check.

## External dependencies

`coord_ext`, `matplotlib`, `numpy`
