# EXP055 (hard_occ): Epoch time regression fixed (80s → 150s)

### 📝 Summary of Changes

- **Root cause found**: the new occupancy binarization (`occupancy_binary.py`) ran a **Python
  for-loop over the batch with per-row `.item()`** inside `model.decode()`, which is on the training
  hot path (called ~3× per step, ~134 steps/epoch). Each `.item()` and the `is_binary_occupancy`
  `bool(torch.all(...))` forces a **GPU→CPU sync**, serializing the GPU and adding huge, jittery
  overhead.
- **Fix**: rewrote `topk_occ_binary` + `occupancy_for_heatmap_decode` to be **fully vectorized and
  sync-free** (descending-sort threshold for per-row K). Removed the redundant `is_binary_occupancy`
  check from the decode path.

### 🚀 Implementation Details

**Before** (per decode call, B=224):
```python
for i in range(B):
    ki = int(k_per[i].item())          # GPU->CPU sync ×224
    idx = occ_prob[i].topk(ki).indices  # 224 tiny kernels
    out[i].scatter_(0, idx, 1.0)        # 224 tiny kernels
```
Plus `is_binary_occupancy` → `bool(torch.all(...))` → another sync per decode.

**After** (vectorized, no loop, no sync):
```python
k_per = k.long().clamp(0, N)
sorted_vals, _ = occ_prob.sort(dim=-1, descending=True)
thr = sorted_vals.gather(1, (k_per - 1).clamp(min=0).unsqueeze(1))
out = (occ_prob >= thr).float().masked_fill((k_per == 0).unsqueeze(1), 0.0)
```
For binary GT the k-th largest value is exactly `1.0`, so the result has **exactly K ones** and is
**bit-identical** to the reference top-K.

### 🛠️ Verification & Execution Results

**Benchmark** (402 decodes ≈ one epoch of decode calls, on the shared GPU):
```
current occupancy binarize: 75.32s   (187.37 ms/decode)
vectorized binarize:         0.06s   (  0.14 ms/decode)
speedup: 1364.6x    epoch overhead saved ~75s
match on binary GT: True    sums equal K: True
```

**Correctness** (CPU unit tests):
```
binary GT sums==K: True        binary GT preserved: True
soft scalar K=5 sums: 5,5,5,5  K=0 rows zero: [0,3,0,7]
STE grad flows: True           STE forward binary: True
```

This accounts for the observed ~80s → ~150s per-epoch increase (and the large variance, since GPU
syncs stall on any concurrent GPU work).

### ⚠️ Action required

The **currently running** training (`pid 27848`) imported the old slow module at startup and will
**not** pick up this fix until restarted. It is a fresh run (metrics already cleared), so restart to
get the speedup:
```bash
# stop the current run, then:
./experiments/exp055_hard_occ/run_train_gpu1.sh
```
(Not stopping it automatically — killing a training process is destructive; restart when ready.)
