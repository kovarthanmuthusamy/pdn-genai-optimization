# IC Inverse Mask Creator

Creates a 64×64 binary mask that is **0 everywhere** and **1 at the grid cell corresponding to a given IC component's physical location** on the PCB.

The result is the spatial inverse of `configs/binary_mask.npy` (board outline mask) in the sense that it marks a single point of interest rather than the valid board area.

---

## Output

| File | Description |
|------|-------------|
| `configs/ic_inverse_mask.npy` | 64×64 `uint8` array — 0 everywhere, 1 at IC location |
| `configs/ic_inverse_mask_overlay.png` | Visualisation: board mask, inverse mask, overlay |

---

## Configuration (top of script)

```python
IC_X       = 27.595    # physical X coordinate from XML <Location> tag (mm)
IC_Y       = 101.975   # physical Y coordinate from XML <Location> tag (mm)
PAD_RADIUS = 0         # extra cells to mark around the IC cell (0 = single pixel)
GRID_SIZE  = 64        # must match board mask and heatmap grid size
OUTPUT_FILE = "../configs/ic_inverse_mask.npy"
```

To use a different IC component, change `IC_X` / `IC_Y` to the values from its `<Location>` tag in `ProKI_design.xml`.

---

## Coordinate Mapping

The mapping is **identical** to `mask_creator.py` and `heatmap.py`, ensuring pixel-perfect alignment across all three:

1. Polygon bounds are read from `ProKI_design.xml` via `coord_ext.py`  
   (x: 0 – 300 mm, y: 0 – 320 mm)

2. Physical IC location is normalised to [0, 1]:

$$x_{norm} = \frac{x_{IC} - x_{min}}{x_{max} - x_{min}}, \quad y_{norm} = \frac{y_{IC} - y_{min}}{y_{max} - y_{min}}$$

3. Mapped to 64×64 grid indices (row = y-axis, col = x-axis):

$$\text{col} = \operatorname{round}(x_{norm} \times 63), \quad \text{row} = \operatorname{round}(y_{norm} \times 63)$$

**Example** — `IC1_Port1 pin 1` at `(27.595, 101.975)`:

| | Value |
|---|---|
| Normalised | (0.0920, 0.3187) |
| Grid cell | row = 20, col = 6 |

---

## Usage

```bash
python Rules/ic_inverse_mask_creator.py
```

Run from any directory — the script changes into its own directory internally so `ProKI_design.xml` is found correctly.

---

## XML Reference

The IC location is taken from the `<Location>` tag in `ProKI_design.xml`:

```xml
<LayerPad layerRef="Conductor-1">
  <Xform rotation="90"/>
  <Location x="27.595" y="101.975"/>
  <StandardPrimitiveRef id="r207_66"/>
  <PinRef componentRef="IC1_Port1" pin="1"/>
</LayerPad>
```

---

## Consistency with other files

| File | Grid | Coordinate bounds | Row/Col convention |
|------|------|-------------------|--------------------|
| `Rules/mask_creator.py` | 64×64 | polygon min/max from `coord_ext.py` | row = y, col = x |
| `pipelines/data/heatmap.py` | 64×64 | data min/max (same physical space) | row = y, col = x |
| `Rules/ic_inverse_mask_creator.py` | 64×64 | polygon min/max from `coord_ext.py` | row = y, col = x |
