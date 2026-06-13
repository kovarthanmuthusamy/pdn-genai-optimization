import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from coord_ext import coordinate_extractor

# ==============================================================
# CONFIG
# ==============================================================
IC_X         = 27.595       # IC component X location from XML
IC_Y         = 101.975      # IC component Y location from XML
PAD_RADIUS   = 0            # extra grid cells around the IC cell (0 = single pixel)
GRID_SIZE    = 64           # must match the board mask grid size
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "../configs/ic_inverse_mask.npy")
# ==============================================================


def make_ic_inverse_mask(ic_x, ic_y, pad_radius=PAD_RADIUS,
                          grid_size=GRID_SIZE, output_file=OUTPUT_FILE):
    """
    Create an inverse mask (all zeros) with a 1 at the grid cell
    corresponding to the given IC component location.

    The coordinate normalization is identical to mask_creator.py so the
    inverse mask aligns pixel-perfectly with configs/binary_mask.npy.

    Parameters
    ----------
    ic_x, ic_y   : physical coordinates from the XML <Location> tag (mm)
    pad_radius   : number of extra cells to mark around the IC cell
    grid_size    : grid resolution (must match board mask)
    output_file  : path where the resulting .npy mask is saved
    """
    # ------------------------------------------------------------------
    # Step 1: get the same coordinate bounds used by mask_creator.py
    # ------------------------------------------------------------------
    os.chdir(os.path.dirname(__file__))   # coord_ext opens ProKI_design.xml locally
    x_coords, y_coords = coordinate_extractor()
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # ------------------------------------------------------------------
    # Step 2: normalise IC location using the same transform
    # ------------------------------------------------------------------
    x_norm = (ic_x - x_min) / (x_max - x_min)
    y_norm = (ic_y - y_min) / (y_max - y_min)

    print(f"IC physical : ({ic_x}, {ic_y})")
    print(f"IC normalised: ({x_norm:.4f}, {y_norm:.4f})")

    # ------------------------------------------------------------------
    # Step 3: map to grid indices
    # meshgrid uses 'xy' indexing: rows -> y, cols -> x
    # linspace(0,1,grid_size) -> index = round(norm * (grid_size-1))
    # ------------------------------------------------------------------
    col = int(round(x_norm * (grid_size - 1)))   # x -> column
    row = int(round(y_norm * (grid_size - 1)))   # y -> row

    col = np.clip(col, 0, grid_size - 1)
    row = np.clip(row, 0, grid_size - 1)

    print(f"IC grid cell : row={row}, col={col}")

    # ------------------------------------------------------------------
    # Step 4: build the inverse mask (zeros everywhere, 1 at IC location)
    # ------------------------------------------------------------------
    inverse_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)

    r_lo = max(0, row - pad_radius)
    r_hi = min(grid_size - 1, row + pad_radius)
    c_lo = max(0, col - pad_radius)
    c_hi = min(grid_size - 1, col + pad_radius)

    inverse_mask[r_lo:r_hi + 1, c_lo:c_hi + 1] = 1

    print(f"Cells marked : {inverse_mask.sum()}")

    # ------------------------------------------------------------------
    # Step 5: save
    # ------------------------------------------------------------------
    out_path = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, inverse_mask)
    print(f"Inverse mask saved to {out_path}")

    return inverse_mask, row, col


def visualise(inverse_mask, row, col, board_mask_path="../configs/binary_mask.npy"):
    """Overlay the inverse mask on the board mask for a quick sanity check."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Board mask
    if os.path.exists(board_mask_path):
        board = np.load(board_mask_path).astype(float)
        axes[0].imshow(board, cmap="gray", origin="lower")
        axes[0].set_title("Board mask (binary_mask.npy)")

        # Overlay
        overlay = board.copy()
        axes[2].imshow(board, cmap="gray", origin="lower", alpha=0.6)
        axes[2].imshow(inverse_mask, cmap="Reds", origin="lower", alpha=0.8)
        axes[2].set_title("Overlay (IC on board)")
    else:
        axes[0].set_visible(False)
        axes[2].set_visible(False)

    # Inverse mask alone
    axes[1].imshow(inverse_mask, cmap="gray", origin="lower")
    axes[1].scatter([col], [row], c="red", s=40, zorder=5)
    axes[1].set_title(f"Inverse mask  row={row}, col={col}")

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../configs/ic_inverse_mask_overlay.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Visualisation saved to {out_png}")


if __name__ == "__main__":
    inverse_mask, row, col = make_ic_inverse_mask(IC_X, IC_Y)
    visualise(inverse_mask, row, col,
              board_mask_path=os.path.join(os.path.dirname(__file__),
                                           "../configs/binary_mask.npy"))
