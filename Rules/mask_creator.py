import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from coord_ext import coordinate_extractor
import os
#
Grid_size = 64

def Mask():
    """
    Create or load PCB mask. If mask file exists, load it instead of recreating.
    This prevents redundant mask generation on repeated calls.
    """
    mask_file = "pcb_mask_128.npy"
    
    # ----------------------------
    # Check if mask already exists
    # ----------------------------
    if os.path.exists(mask_file):
        print(f"Loading existing mask from {mask_file}")
        mask = np.load(mask_file)
        return mask
    
    print(f"Creating new mask...")
    # ----------------------------
    # Step 1: Define your polygon points
    # ----------------------------
    x_coords, y_coords = coordinate_extractor()

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # ----------------------------
    # Step 2: Normalize coordinates to [0,1]
    # ----------------------------
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

    # ----------------------------
    # Step 3: Create 256x256 grid
    # ----------------------------
    grid_size = Grid_size
    xv, yv = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))

    # ----------------------------
    # Step 4: Rasterize polygon
    # ----------------------------
    polygon = np.column_stack((x_norm, y_norm))
    path = mpath.Path(polygon)

    # Flatten grid points and check which are inside polygon
    points = np.vstack((xv.flatten(), yv.flatten())).T
    mask = path.contains_points(points).reshape(grid_size, grid_size)
    mask = mask.astype(np.uint8)  # 1 inside, 0 outside

    # ----------------------------
    # Step 5: Display mask
    # ----------------------------
    #plt.imshow(mask, cmap="gray", origin="lower")
    #plt.title("PCB Mask ({} x {})".format(Grid_size,Grid_size))
    #plt.show()

    # ----------------------------
    # Save mask for future use
    # ----------------------------
    np.save(mask_file, mask)
    print(f"Mask saved to {mask_file}")
    
    return mask

#Mask()