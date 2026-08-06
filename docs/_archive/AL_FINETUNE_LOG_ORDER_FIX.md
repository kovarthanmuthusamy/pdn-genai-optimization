# AL Fine-Tune Log Order Fix

### 📝 Summary of Changes

- **`active_learning_pi/al/finetune_run.py`**: Added `prepare_line_buffered_logging()`; set `PYTHONUNBUFFERED=1` for training subprocess; flushed schedule print.
- **`pipelines/active_learning/finetune_exp057.py`**: Call `prepare_line_buffered_logging()` at start; `flush=True` on wrapper prints.
- **`active_learning_pi/al/pipeline.py`**: Same buffering in `cmd_finetune`.

### 🚀 Implementation Details

**What looked like a second fine-tune**

When stdout is redirected (e.g. `python finetune_exp057.py > /tmp/finetune_exp057_run.log`), Python block-buffers the parent process. `finetune_exp057.py` prints:

1. `=== Build AL overlay ===`
2. `=== Launch exp057 AL fine-tune ===`
3. then runs `train_vae_simple.py` (child writes immediately)

The child’s ~20h of logs appeared first; the parent’s buffered lines flushed **only after training finished**, so `tail -f` showed overlay/launch messages after `TRAINING COMPLETE` — as if fine-tune restarted. **There was no second run or auto-retrigger.**

### 🛠️ Verification & Execution Results

- Reproduced block-buffering with redirected stdout (child before parent).
- After fix, preamble lines appear before training output in redirected logs.
- No duplicate finetune hook exists in `train_vae_simple.py` or `train_core.py`.
