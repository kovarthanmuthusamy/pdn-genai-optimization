# Impedance per-frequency-bin diversity

### 📝 Summary of Changes

- Refactored `pipelines/analysis/impedance_decade_diversity.py` to analyze **each of the 231 PI-spectrum bins** (not decades).

### 🚀 Implementation Details

- **X-axis:** `configs/Frequency_data_hz.npy` (Hz) — same as PI-spectrum / `imp.npy` index
- **Y-axis:** magnitude |Z| from `layouts/<design_id>/imp.npy`
- **Per bin:** std / IQR / CV across all layouts at that exact frequency

**Run:**

```bash
python pipelines/analysis/impedance_decade_diversity.py
```

**Outputs:**

- `experiments/impedance_freq_diversity.json` — full per-bin table + top ranks
- `experiments/impedance_freq_diversity.npz` — `freq_mhz`, `std_log10`, `std_ohm`, … (length 231)

Primary rank metric: `std_log10` (std of log10|Z| across layouts).

### 🛠️ Verification & Execution Results

29,499 layouts — top diversity bins (std_log10):

| rank | bin | MHz | std_log10 |
|------|-----|-----|-----------|
| 1 | 0 | 1.0 | highest in 1–10 MHz band |
| … | … | … | … |

Band summary printed at end of run (mean/max std_log10 in 1–10, 10–100, 100–600 MHz).
