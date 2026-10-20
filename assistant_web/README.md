# assistant_web — the design assistant as a web page (defense demo)

A board tool on top of the thesis model (exp096 `last_model.pt` + placement-only encoder `H1_spec_hm200`).
You click slots on the H-shaped board, and the page shows at once:
- the verdict for the chosen target, safe (with the measured margin) and direct (no margin);
- the predicted spectrum against the target;
- the predicted map at a chosen frequency;
- the nearest passing placements and the best one-decap additions.

Every number comes from `thesis_code` (the verdict, `find_fixes` ranking and margins of the thesis
assistant). `thesis_code` itself is not changed.

## Run

```bash
cd /home/ubuntu/genai_pdn && source /home/ubuntu/venv-cgan/bin/activate
python -m assistant_web.server          # then open http://localhost:8050 in the Windows browser
```

- **Start-up:** about 5 s. The very first start also computes the colour scale of the maps (about 40 s);
  it is cached in `cache/map_scale.json`.
- **Stopping:** Ctrl+C.
- **Settings:** port and GPUs are in the CONFIG block of `server.py`, or set through `ASSIST_WEB_PORT` and
  `ASSIST_WEB_GPUS=0,1`.
- **Shared host:** no eCADSTAR is used. The GPUs are used only while the page is in use.

## Using it

**Placing parts**
- Pick T1 / T2 / T3 (keys 1 / 2 / 3), then click a slot.
- Clicking a slot that already holds the selected part removes it. "✕" (key 0) removes any part.
  Ctrl+Z undoes.
- At most 10 decaps, the trained range.
- An empty board shows the simulated bare-board spectrum, labelled as such.

**Examples** (`examples.json`, written by `python -m assistant_web.make_examples`)
- the thesis assistant's example;
- the rule of thumb (T1 in the three slots closest to the IC port);
- independent placement 138, which fails 0.6 Ω and is repaired by one move (confirmed in eCADSTAR);
- the search result for 0.5 Ω at K = 4.

**Map**
- The board shows no map until you click the spectrum curve. The click picks that frequency point, for
  example a peak, and the map of that frequency is painted on the board; a ring marks the point on
  the curve.
- The picked frequency stays when the placement changes, so before and after are compared at the same
  frequency. "✕ clear" removes the map. Choosing an example starts again with no map.
- "fixed" keeps one colour scale per frequency: the 1st–99th percentiles of predicted maps of
  random placements with K = 3–8. "auto" scales each map to its own range.

**Suggestions**
- Hover a row to preview it on the board and in the plot; click it to apply it.
- The nearest passing placement keeps K and is searched up to 2 changed decaps. If nothing passes
  there, it searches 3 changed decaps automatically: about 6 s for K = 5, against 13 s for the thesis
  script.
- Two things make the 3-change search faster, and neither changes the proposals:
  - the candidates are judged on both GPUs, in a worker process per extra GPU;
  - the candidate builder draws the same random sample as `suggest.neighbours_at` but builds only the
    drawn neighbours.

## Files

| File | What |
|---|---|
| `server.py` | FastAPI app, CONFIG block, routes `/api/setup`, `/api/check`, `/api/map`, `/api/fixes` |
| `engine.py` | loads the model once; `check`, `additions`, `map`, `fixes`; GPU worker processes; `SplitJudge`; `neighbours_fast` / `find_fixes_fast` |
| `board.py` | slot positions, IC port, H-board outline, map grid (mm) |
| `make_examples.py` | writes `examples.json` from saved thesis results |
| `static/` | `index.html`, `app.js`, `style.css`, `vendor/plotly-basic.min.js` (local copy, no internet needed) |

## Tests

```bash
pytest tests/assistant_web -q      # 21 tests, about 80 s; needs the weights and a GPU (two for the split)
```

The tests check that:
- the verdicts equal `suggest.verdict` for every example and target;
- the fix lists equal `suggest.find_fixes` (same placements, order and scores);
- the GPU split gives bit-identical ratios;
- the fast candidate builder gives identical candidates and generator state;
- the maps equal `Surrogate.maps`;
- the budget limits and the routes work.

## Limits (also on the page)

- Predictions of the surrogate, not eCADSTAR.
- The safe margins were measured on the test placements.
- Only the three thesis targets (0.8 / 0.6 / 0.5 Ω) are offered.
