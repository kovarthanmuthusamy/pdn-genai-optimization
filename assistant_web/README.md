# assistant_web — the design assistant as a web page (defense demo)

A board tool on top of the thesis model (exp096 `last_model.pt` + placement-only encoder `H1_spec_hm200`).
You click slots on the H-shaped board, and the page shows at once:
- the verdict for the chosen target, safe (with the measured margin, or one set with the slider) and
  direct (no margin);
- the predicted spectrum against the target;
- the predicted map at a chosen frequency;
- the nearest passing placements and the best one-decap additions;
- on request, an eCADSTAR simulation of the placement, compared with the model (eCADSTAR comparison view).

Every number comes from `thesis_code` (the verdict, `find_fixes` and margins of the thesis assistant).
`thesis_code` itself is not changed. One difference from the thesis assistant: the fix search may also
remove decaps (the thesis code's `ALLOW_REMOVE`, off in the thesis), see Suggestions.

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

The eCADSTAR note of an example shows as the hover text of the example list.

**Margin**
- The slider in the verdict row sets the safe margin m (0–60 %). Safe pass needs |Z| ≤ (1 − m) × target
  at every frequency.
- Default: the thesis margin of the target (0.8 Ω: 25 %, 0.6 Ω: 37 %, 0.5 Ω: 27 %), marked by a tick.
  Another value moves the dashed safe bar, the badge (then labelled "MARGIN 20 %" instead of "SAFE"),
  the safe chips under "Add one decap" and the safe "Nearest pass" search. Direct ignores it.
- "↺" goes back to the thesis margin. Changing the target or choosing an example also does; a
  placement change keeps the chosen margin.

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
- "Nearest pass" searches up to 2 changes (type change, move or remove), then 3 changes automatically
  when nothing passes: about 6 s for K = 5, against 13 s for the thesis script.
- Ranking (author, 2026-10-22): fewer changes first; with the same number of changes, fewer decaps
  first ("remove C19" before "C19: T3 → T1"); then type changes before moves, then headroom. A row
  with another K than the board shows a "K 3" tag.
- A placement that already passes gets a switch "fewer decaps | improve":
  - fewer decaps: the nearest placements with fewer decaps that still pass ("Passes · 2 with fewer
    decaps"). "Already passes." means no smaller budget passes within 3 changes;
  - improve (author, 2026-10-22): the same K, every placement 1 or 2 changes away (type change or
    move), those with a lower ρ̂ than now, the lowest first ("ρ̂ 0.72 → 0.63"); about 0.2–0.8 s.
    A lower ρ̂ than a passing placement's also passes. Model predictions: apply a row and use the
    eCADSTAR comparison to check it.
- For a failing placement a removal rarely helps (fewer decaps raise the impedance). Checked 2026-10-22:
  #138 at 0.6 Ω gives the thesis list unchanged (34 found, 1 change); the thesis example at 0.8 Ω needs
  the sampled 3-change level, whose random sample now also draws removals, so its list differs slightly
  (517 found instead of 523, all K = 5).
- Two things make the 3-change search faster, and neither changes the proposals:
  - the candidates are judged on both GPUs, in a worker process per extra GPU;
  - the candidate builder draws the same random sample as `suggest.neighbours_at` but builds only the
    drawn neighbours.

**eCADSTAR comparison** (second view: tabs "Design | eCADSTAR comparison" under the header)
- The view fills the window: the button and the table on the left, a large spectrum plot (target, model,
  eCADSTAR) on the right, and below the model map and the eCADSTAR map side by
  side on the board, with the decaps, the IC port and the hot-spot rings. Clicking the curve picks the
  map frequency here too; the picked frequency is shared with the design view. The part palette is
  hidden in this view (the board is edited in the design view); target, examples, undo and clear work.
- "▶ Simulate in eCADSTAR" simulates the selected placement: its spectrum, plus the map at the frequency
  picked on the curve if one is picked. One placement with one map took 19 s (2026-10-22).
- It runs through `thesis_code.s12_evaluation.ecad`, as the Chapter 4 checks did (batches start with a
  C1 group, runs resume). Results are saved in `cache/ecad/`, so a placement (and a map frequency) is
  simulated once; a placement simulated before shows its result at once.
- eCADSTAR is a shared host: the button refuses while engineer.exe or a dataset / queue job runs, and
  never waits in line. One check runs at a time.
- The table compares model and eCADSTAR: ρ, its frequency, pass (ρ ≤ 1, no margin, for both) and the
  hot spot at the picked frequency. The two maps share one colour scale ("fixed" / "auto", the same
  switch as in the design view).
- Check of the pipeline: #138 at 0.6 Ω gives eCADSTAR ρ = 1.27, the value of its earlier simulation in
  the independent placements; the model says 1.40. Hot spot at 120 MHz: model 7.04 Ω, eCADSTAR
  6.72 Ω, same pixel.

**On a projector**
- Set the laptop output to 1920 × 1080 if the projector takes it: from 1600 px wide the page switches to
  larger text, numbers and plot lines for reading from the back of the room. Lower outputs (1280 × 720,
  1366 × 768) still fit on one screen with the normal sizes.
- Full screen with F11 gives the page the whole height; Ctrl + / Ctrl − zoom still works, but the
  one-screen layout is only checked at 100 %.
- The page needs no internet (local plot library, system fonts).

## Files

| File | What |
|---|---|
| `server.py` | FastAPI app, CONFIG block, routes `/api/setup`, `/api/check`, `/api/map`, `/api/fixes` |
| `engine.py` | loads the model once; `check`, `additions`, `map`, `fixes`; GPU worker processes; `SplitJudge`; `neighbours_fast` / `find_fixes_fast` |
| `board.py` | slot positions, IC port, H-board outline, map grid (mm) |
| `ecad_check.py` | the eCADSTAR comparison: `EcadRunner` (one job at a time in a thread, busy check, thesis `ecad` store) and `compare` |
| `make_examples.py` | writes `examples.json` from saved thesis results |
| `static/` | `index.html`, `app.js`, `style.css`, `vendor/plotly-basic.min.js` (local copy, no internet needed) |

## Tests

```bash
pytest tests/assistant_web -q      # 45 tests, about 2.5 min; needs the weights and a GPU (two for the split)
```

The tests check that:
- the verdicts equal `suggest.verdict` for every example and target;
- the fix lists equal `suggest.find_fixes` (same placements, order and scores): with removals off the
  thesis lists, with removals on `find_fixes` with `ALLOW_REMOVE` on, ranked as above; for a passing
  placement, `find_fixes` with its K range capped at K − 1;
- the GPU split gives bit-identical ratios;
- the fast candidate builder gives identical candidates and generator state;
- the maps equal `Surrogate.maps`;
- improve lists every same-K placement 1 or 2 changes away with a lower ratio, lowest first;
- a margin from the slider gives `suggest.verdict` and `suggest.find_fixes` with that margin, and no
  margin gives the thesis margin;
- the eCADSTAR comparison reads saved results correctly, refuses while eCADSTAR is busy, calls the thesis
  simulation with `ECADSTAR_RUN` on only during the job, and never simulates a saved placement again
  (the eCADSTAR calls are replaced in the tests; the real run was checked by hand);
- the budget limits and the routes work.

## Limits (in the hover texts of the page)

- Predictions of the surrogate; eCADSTAR results only in the eCADSTAR comparison (needs the Windows host).
- The thesis margins were measured on the test placements; a margin set with the slider was not.
- Only the three thesis targets (0.8 / 0.6 / 0.5 Ω) are offered.
