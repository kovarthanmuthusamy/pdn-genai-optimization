# Design assistant web page (defense demo) — design

Date: 2026-10-20. Approved by the author in an interview the same day ("proceed to build").

## Purpose

A live demo of the thesis design assistant (Ch4, Section 4.4) for the thesis defense. Only the author
operates it, on the WSL machine with both GPUs, opened in the Windows browser at `http://localhost:8050`.
It is a board tool, not a chatbot: the author clicks slots on the H-shaped board and the thesis model
answers at once.

## Decisions (interview 2026-10-20)

| Topic | Decision |
|---|---|
| Audience | thesis defense demo, operated by the author |
| Interaction | board tool: click slots, see verdict, spectrum, map, suggestions |
| Functions | check (always) + maps + fix suggestions; no design from a target, no eCADSTAR overlay |
| Runtime | this PC, live model; no static backup |
| Verdict | both modes: safe (with the measured margin) as the main badge, direct (no margin) below |
| Targets | only the three thesis targets 0.8 / 0.6 / 0.5 ohm (margins 0.25 / 0.37 / 0.27) |
| Start | empty board + a list of prepared examples |
| Close-call label | no |
| Layout | map painted under the board; verdict, spectrum and suggestions on the right; one screen |
| Build | FastAPI server + hand-written page; plot library stored locally (no internet at the defense) |
| Fix search | up to 3 changed decaps, as in the thesis; judging split over both GPUs |

## Architecture

`assistant_web/` (new; `thesis_code/` is imported, never changed):

- `engine.py` — loads the thesis model once (exp096 `last_model.pt` + head `H1_spec_hm200`) through
  `thesis_code`: the full surrogate on GPU 0 (spectrum + maps), one more copy of the placement-only
  encoder per extra GPU. Answers: `check`, `map`, `fixes`, `additions`. Verdicts use
  `thesis_code.s11_assistant.suggest.verdict`; the fix search is `suggest.find_fixes` with a judge whose
  spectrum call is split over the GPUs, so proposals are the thesis assistant's.
- `board.py` — slot positions, IC port, H-board outline (notches from the board mask), map grid extent.
- `make_examples.py` — writes `examples.json` from saved thesis results (CONFIG-only script).
- `server.py` — FastAPI app (CONFIG block: host, port, GPUs); `python -m assistant_web.server`.
- `static/` — `index.html`, `app.js`, `style.css`, `vendor/plotly-basic.min.js`.

## Behaviour

- Empty board: the simulated bare-board spectrum, labelled as simulated (the model needs K >= 1).
- An 11th decap is refused (the model was trained for K = 1..10).
- Map: predicted at the frequency of the slider, which jumps to the worst frequency after each change;
  clicking the spectrum plot also sets it. Colour scale fixed per frequency (percentiles of predicted
  maps of reference placements, cached), jet colours on a log scale as in the thesis figures.
- Spectrum plot: prediction, target, safe bar (dashed), previous placement (grey dashed), hovered
  suggestion (dotted).
- Suggestions: nearest passing placements (safe/direct switch, 5 shown), searched automatically up to
  2 changes, then up to 3 when nothing passes ("searching 3 changes ..."); a newer request cancels an
  older search. Additions: the 5 best one-decap additions when K <= 9. Hover previews, click applies.
- Footer: predictions of the surrogate, not eCADSTAR; margins measured on the test placements.

## Testing

`tests/assistant_web/`: verdict and ratio equal `suggest.verdict` for the examples and all targets;
fix list identical (order and codes) to `suggest.find_fixes` with the one-GPU thesis judge; maps equal
`Surrogate.maps`; K = 0 and K = 11 handled; API smoke test. Then a browser check with a screenshot.

## Added while building (2026-10-20)

- Fix search speed (author: "we can use two gpus for this process"): a worker process per extra GPU
  (threads gave only 1.2x, processes run fully in parallel), and `neighbours_fast`, which makes the
  same random draws as `suggest.neighbours_at` but builds only the drawn neighbours (identical
  candidates and generator state, tested). 3-change search for K = 5: 13.3 s -> about 6 s.
- Map controls: a "this map" colour scale next to the fixed one, and a "follow worst frequency" switch
  (off = compare before and after at one frequency).
- The hot-spot value is shown next to the frequency, not on the board (it covered slot labels).
- Change requested by the author after the first build: no slider. The board shows no map until the
  spectrum curve is clicked; the click picks that frequency (e.g. a peak), which is kept across
  placement changes until "clear map" or a new example. The "follow worst frequency" switch was removed
  with the slider.
- Wording cut on request (2026-10-20): short labels, details moved into hover tooltips.
- Verdict sentence "Passes/Fails <target> with a <m> margin" removed on request (2026-10-22); the badge,
  the "needs ρ̂ ≤ 1 − m" line on a fail, the direct chip and ρ̂ remain.
- Margin slider (author, 2026-10-22, interview: "Verdict + fixes too", "Slider + reset"): 0–60 % in 1 %
  steps in the verdict row, a tick at the thesis margin and a "thesis" reset button. The chosen margin
  drives the safe bar, the badge ("MARGIN m %" when it differs from the thesis value), the safe chips of
  the additions and the safe fix search; the API takes an optional `margin` (null = thesis margin), so
  the default answers stay the thesis's (tests: 28 pass). The margin resets on a target or example
  change and is kept across placement changes.
- Text cut (author, 2026-10-22): the reset button is an icon (↺, thesis value in its tooltip); the
  "needs ρ̂ ≤" line, the example notes (now the example list's tooltip) and the footer are gone; the
  "Click the curve" hints stay.
- Budget in the fix search (author, 2026-10-22, interview: "Remove decaps too", "Fewer changes first",
  passing placement: "give a suggestion like removing this also passes"): the search also removes decaps
  (thesis `ALLOW_REMOVE` semantics, a removal = one change, still up to 3 changes); ranked by changes,
  then K, then the thesis order; a passing placement gets the nearest placements with fewer decaps that
  still pass. `remove=False` in the API gives the thesis assistant. Tests: 40 pass.
- eCADSTAR tab (author, 2026-10-22, reversing "no eCADSTAR overlay"; interview: "Spectrum +
  picked-frequency map", "Tab beside suggestions", "Short: verdict only", one test run allowed):
  `ecad_check.py` runs one job at a time through thesis_code `s12_evaluation.ecad` (store
  `cache/ecad/`), refuses while `ecad.busy()`; table rows ρ / worst frequency / passes (ρ ≤ 1) / hot
  spot, model vs eCADSTAR; map pair + green dashed simulated spectrum. Test run 2026-10-22: #138,
  spectrum + 120 MHz map in 19 s, ρ 1.27 (matches its earlier simulation). Tests: 44 pass.
- Same day, on request: the comparison moved out of the small tab into a full second view ("Design |
  eCADSTAR comparison" under the header): button + table, large spectrum plot (target, model,
  eCADSTAR), large model / eCADSTAR maps on the board. The suggestions panel is back to one pane; the
  design view's plot no longer shows the eCADSTAR curve.
- Design plot trimmed (author, 2026-10-22): no "previous placement" curve and no vertical line at the map
  frequency (the ring on the curve stays); the comparison plot keeps its line. eCADSTAR curve there: dotted.
- No hot-spot ring on the design board (author, 2026-10-22); "hot spot … Ω" under the board stays, and the
  comparison maps keep their rings.
- No "worst" × markers in either plot (author, 2026-10-22); the worst frequency stays in the verdict row
  and in the comparison table.
- Board panel narrower (40 / 60 instead of 46 / 54) and "improve" for a passing placement (author,
  2026-10-22, interview: "Lowest ρ̂", "Up to 2 changes", "Biggest drop first", "Switch in Nearest pass"):
  `Engine.improve` enumerates every same-K placement 1-2 changes away, keeps those below the current ρ̂,
  lowest first (ties: fewer changes, type changes before moves); `/api/improve`; switch "fewer decaps |
  improve" shown only when the placement passes. Fixes now start after the verdict arrives. Tests: 45.
- Visual polish (author, 2026-10-22: "make it likable and easy"), CSS-led, no behaviour change: light
  theme kept for the projector (dark text on light, stronger pass/fail colours), cool neutral background,
  white cards with soft shadows, one accent blue for the active tab / primary button / slider, brand mark
  (H-board logo), icons on Undo / Clear / view tabs, segmented switches as pills, numbered suggestion
  rows with hover highlight and part names in their colours, map controls in a toolbar, legend swatches
  in the comparison table and map titles, board drop shadow, one plot style for both plots, a start hint
  while the board is empty, fail chips without the ✗ sign. Both views fit 1920×1080 and 1366×768.
- Projector sizing (author, 2026-10-22: shown on a large screen): grey text darkened to ~5:1 contrast; from
  1600 px wide (a 1080p projector) text, controls, numbers 15-30 % larger and plot text 15-16 px with
  thicker lines (app.js PS); board labels larger; ≤1320 px / ≤760 px tall keep one screen (1280×720).
  Checked: 1280×720, 1280×800, 1366×768, 1920×1080, 2560×1440, both views, no scrolling.
