---
title: active-learning-pi-heatmaps (archived)
type: archive
source: docs/_archive/active-learning-pi-heatmaps.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

## Active learning loop for PI-distribution heatmaps

This document describes a practical active-learning loop to improve **PI-distribution heatmap** accuracy at **off-anchor PI frequencies** (e.g. 80 / 250 MHz) with minimal ECADStar simulations.

### Flowchart

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 70}, 'theme': 'base', 'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
  %% Swimlanes
  subgraph L1["Model / Inference (cheap)"]
    A["Generate candidates<br/>(decap combos × MHz grid)"]
    B["Predict heatmaps<br/>+ robust stats (p95/p99/top‑K)"]
    C["Score candidates<br/>(uncertainty / disagreement)"]
    D["Select batch<br/>(top‑N + random + diversity)"]
  end

  subgraph L2["ECADStar Simulation (expensive labels)"]
    E["Build PEB / run list"]
    F["Run ECADStar batch"]
    G["Export real heatmaps"]
  end

  subgraph L3["Data + Training (update model)"]
    H["Ingest labels<br/>(PI_freq + occupancy + heatmap)"]
    I["Fine‑tune model<br/>(layout‑heavy + cross‑freq + peak loss)"]
    J["Evaluate<br/>(off‑anchor 80/250 + anchors)"]
  end

  A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> A
```

### Recommended acquisition signals

- **MC Dropout variance of robust peaks**: compute variance of foreground p99 (or top-K mean) across N stochastic passes.
- **Ensemble disagreement**: disagreement across multiple trained checkpoints/seeds (best uncertainty quality, higher cost).
- **Layout-path disagreement**: prioritize candidates where layout-mode decoding is unstable vs alternate decoding (targets your deployment path).
- **Frequency gap prior**: oversample between training anchors (and explicitly include known-bad MHz like 80/250).

### Metrics to track (robust to outliers)

- Avoid `max()` as a primary metric. Prefer **foreground percentiles** (`p95`, `p99`, `p99.9`) and **top-K mean**.
- Track off-anchor performance for your generation path: **`layout_cross`** error at 80/250 MHz (and compare to `encode_cross`).

### Automated pipeline implementation

See **`active_learning_pi/README.md`**. Default flow:

1. **Infer all** candidates (model-only, one-by-one scoring).
2. **Select worst** (high uncertainty / bad predictions) — skip good ones.
3. **One ECADStar batch** (single `.peb`) for labels only on bad cases.
4. **Ingest + fine-tune**.

Command: `python active_learning_pi/run_pipeline.py cycle`
