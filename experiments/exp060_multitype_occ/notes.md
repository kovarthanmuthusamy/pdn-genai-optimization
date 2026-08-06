# exp060 — Multi-type occupancy (one-hot types, no empty channel)

Fork of **exp059_capacity_freq**. Occupancy schema:

\[
\mathrm{occ} \in \{0,1\}^{52 \times T}
\]

- **Empty** = all-zero row `[0, 0, …]`
- **Type t** = one-hot on channel `t-1`
- Config: `n_decap_types` (= \(T\)); model `n_occ_classes = T`
- Default: **T = 2** → shape `(52, 2)`

## Examples (T=2)

| State | Vector |
|-------|--------|
| empty | `[0, 0]` |
| type 1 | `[1, 0]` |
| type 2 | `[0, 1]` |

## Loss / decode

- Decoder logits `(B, 52, T)` + **sigmoid / focal BCE** (not softmax CE — empty is not a class)
- Hard CAD: top-K by max type score → argmax type; others zero
- Heatmap spatial cond uses **presence** = sum of type channels

## Status

- Legacy binary `occ.npy` maps to type-1 only (channel 0).
- exp059 checkpoints are not compatible.
