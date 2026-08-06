# AL Full-Cycle Post-Finetune Evaluation

### 📝 Summary of Changes

- **`active_learning_pi/al/evaluate_report.py`**: Generates `CYCLE_EVAL_REPORT.md` with cycle overview, fine-tune metrics, ECAD pre/post eval, training off-anchor table, per-layout detail.
- **`evaluate_cycle.py`**: Calls report writer automatically at end of step 8.
- **Config** `evaluation.write_markdown_report: true`
- **Command** `evaluate-report` — regenerate report from existing JSON.

### 🚀 Implementation Details

Report sections:
1. Cycle overview (candidates, ECAD batch, overlay)
2. Fine-tune (epochs, val/train loss, overlay weight)
3. ECAD p99 MAE pre vs post + per-layout table
4. Training `layout_cross` off-anchor at final epoch
5. Artifact paths

Outputs:
- `iter_XXXX/CYCLE_EVAL_REPORT.md`
- `runs/<run_name>/LATEST_CYCLE_EVAL_REPORT_iterXXXX.md`

### 🛠️ Verification & Execution Results

Generated for iter 2:
`active_learning_pi/runs/al_exp057_001/iter_0002/CYCLE_EVAL_REPORT.md`

Regenerate without re-infer:
```python
COMMAND = "evaluate-report"
ITERATION = 2
```
