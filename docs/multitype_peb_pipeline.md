# Multi-Type Decap PEB Pipeline & Combined Sampler

Extends the single-type (0/1) combinations→ECADStar flow to **multiple decap
types** encoded as integer type codes per slot (`0` empty, `1` type-1, `2`
type-2). The generated PEB now sets per-component `C`, `ESR`, `ESL` so ECADStar
places the correct physical part on each slot.

## 📝 Summary of Changes

**Added**
- `scrap/generation/generate_peb_multitype.py` — type-aware PEB writer. Emits the
  4-line block per component (`Component/Name`, `Component/Value`,
  `PI Decap/ESR`, `PI Decap/ESL`) matching `configs/Batch_Decap_example.peb`.
- `pipelines/dataset_sim/combinations_multitype.py` — `load_multitype_csv()`
  loader that accepts type codes `{0,1,2}`.
- `pipelines/dataset_sim/peb_multitype.py` — `build_impedance_peb_multitype()`
  and `build_distribution_peb_multitype()` (mirror `peb.py`).
- `pipelines/dataset_sim/run_multitype_sim_pipeline.py` — full pipeline, an exact
  mirror of `run_combinations_sim_pipeline.py` but type-aware (impedance +
  per-MHz PI-Distribution, ECADStar staging, output move, resume).

**Modified**
- `pipelines/heatmaps/sample_multitype_combinations.py` — new **combined mode**
  producing a single CSV of 15000 unique random rows (type1-only + type2-only +
  mixed). Legacy two-file split is still available via `WRITE_LEGACY_SPLIT`.

## 🚀 Implementation Details

### Type catalog (SI units)
Defined once in `generate_peb_multitype.py::TYPE_CATALOG`:

| Type | Code | C | ESL | ESR |
|------|------|-----------|--------|---------|
| Type1 | `1` | 100 nF = `1e-07` F | 222 pH = `2.22e-10` H | 8.9 mΩ = `0.0089` Ω |
| Type2 | `2` | 47 nF = `4.7e-08` F | 154 pH = `1.54e-10` H | 21.4 mΩ = `0.0214` Ω |

Empty slots (`0`) write `false` and `0` for Value/ESR/ESL.

### PEB per-component block (verified against reference)
```
<Edit Table="Component" Name="C14" Column="Name" Value="true" />
<Edit Table="Component" Name="C14" Column="Value" Value="1e-07"/>
<Edit Table="PI Decap" Name="C14" Column="ESR" Value="0.0089"/>
<Edit Table="PI Decap" Name="C14" Column="ESL" Value="2.22e-10"/>
```

### Pipeline
`run_multitype_sim_pipeline.py` keeps every ECADStar behavior of the original
(one PEB per phase, PI-k in CSV order, `sim_progress_multitype.json` resume,
`SKIP_*` flags). Only the CSV loader and PEB builders are swapped for the
type-aware versions. Default input:
`data/heatmaps/combinations_multitype_combined_15000.csv`.
PEB names: `multitype_impedance.peb`, `multitype_dist_{MHz}MHz.peb`.

### Combined sampler
`build_combined()` produces `TOTAL_COMBINED` (15000) unique rows with a K-shape
peaking at K=2:

- **K=1**: all `C(52,1)` single-type layouts (52 type-1 + 52 type-2), if
  `INCLUDE_K1_SINGLE`.
- **K=2**: **fully enumerated** — every layout of all three families
  (`1326 + 1326 + 2652 = 5304`).
- **K=3..K_MAX**: totals **decrease** with K (weights `n..1`), each K split across
  the three families, sampled uniquely at random.

The K=3..K_MAX budget is `TOTAL_COMBINED − fixed(K≤2)`, so the grand total is
exactly `TOTAL_COMBINED`. Verified per-K: `{1:104, 2:5304, 3:4797, 4:3197,
5:1598}` (sum 15000, all unique). Families cannot collide (distinct nonzero value
sets); result is shuffled.

## 🛠️ Verification & Execution Results

- `py_compile` OK for all 4 new/edited modules.
- Sampler run → `combinations_multitype_combined_15000.csv`: **15000 unique rows**,
  per-K `{2:3750, 3:3750, 4:3750, 5:3750}`; families `type1_only/type2_only/mixed`
  = 5000 each, each `{2:1250,3:1250,4:1250,5:1250}`; nonzero type counts
  `{1:26258, 2:26242}`.
- PEB smoke test on real CSV rows: type-1 block → `1e-07 / 0.0089 / 2.22e-10`;
  type-2 block → `4.7e-08 / 0.0214 / 1.54e-10`; distribution PEB contains
  `PI-Distribution` loop with `170e6`; impedance PEB contains `CreatePISpectrum`.
  Group counts match layout count.

## Run

```bash
# 1) build the combined 15k CSV
python pipelines/heatmaps/sample_multitype_combinations.py

# 2) simulate (edit CONFIG at top for MHz / ECADStar paths / SKIP flags)
python pipelines/dataset_sim/run_multitype_sim_pipeline.py
```
