# Heatmap decoder architecture v3

Targets: localized peaks, H-shape fidelity, layout-conditioned cross-freq decode.

## Changes (on top of log1p train space + upsample decoder)

### 1. U-Net skip connections (`use_heatmap_unet_skips: true`)

Encoder stores **16×16 / 32ch** (`s16`) and **8×8 / 128ch** (`s8`) features.  
Decoder fuses `s16` after each upsample (8→16→32).  
When decoding from **layout z only** (no heatmap encode), skips are `None` — `OptionalSkipFuse` falls back to conv-only path (matches sweep inference).

### 2. Occupancy spatial prior (`use_occ_spatial_decoder: true`)

52-d decap vector → **7×8 board grid** → bilinear **32×32** map, concatenated to decoder features.  
Gives the decoder explicit **where capacitors sit** — critical for layout-z cross-freq path.

`decode(z, K, PI_freq, occupancy=occ)` now accepts occupancy; training/sweep pass it through.

### 3. Full-resolution refine

`ResidualConv2d` at **64×64** before output head — sharper peaks, less blur.

## Config

```json
"use_heatmap_unet_skips": true,
"use_occ_spatial_decoder": true
```

## Fresh restart

Incompatible with prior checkpoints (new decoder modules).

## What we did NOT add (yet)

| Idea | Why deferred |
|------|----------------|
| Larger 12×12 bottleneck | +40% decoder params; try v3 first |
| Heatmap←imp cross-attention | Heavier; occ spatial is cheaper layout hint |
| Per-MHz decoder heads | 16× output layers; freq PoE + FiLM may suffice |
| Fixed H-mask multiply | Could help BG; risks killing off-mask learning |

## Verify

```bash
.venv/bin/python scratch/_smoke_exp043_struct.py
```
