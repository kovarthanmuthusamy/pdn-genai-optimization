"""Read/write exp041 metrics CSVs (dedupe by epoch, sorted).

This experiment can produce duplicate epoch rows when training is resumed or a
run restarts mid-epoch. These helpers rewrite CSVs keeping the *last* row per
epoch and sorting by epoch.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src_vae.others.vae_logger import CSV_HEADER, LATENT_CSV_HEADER

EPOCH_TIMING_HEADER = [
    "epoch",
    "train_sec",
    "val_sec",
    "total_sec",
    "val_ran",
    "train_loss",
    "val_loss",
]


def dedupe_csv_by_epoch(
    path: Path,
    header: list[str],
    *,
    epoch_col: str = "epoch",
) -> tuple[int, int]:
    """Keep last row per epoch; rewrite file sorted. Returns (rows_before, rows_after)."""
    if not path.exists():
        return 0, 0
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_before = len(rows)
    if not rows:
        return 0, 0
    by_ep: dict[int, dict] = {}
    for r in rows:
        try:
            ep = int(float(r[epoch_col]))
        except (ValueError, KeyError):
            continue
        by_ep[ep] = r
    ordered = [by_ep[ep] for ep in sorted(by_ep)]
    n_after = len(ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow({h: r.get(h, "") for h in header})
    return n_before, n_after


def insert_epoch_row(path: Path, header: list[str], row: list) -> bool:
    """Insert one row; dedupe file first; skip if epoch already exists. Returns True if inserted."""
    dedupe_csv_by_epoch(path, header)
    ep = int(row[0])
    rows: list[list] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    if int(float(r["epoch"])) == ep:
                        print(f"  {path.name}: epoch {ep} already present — skip")
                        return False
                    rows.append([r.get(h, "") for h in header])
                except (ValueError, KeyError):
                    pass
    rows.append([str(x) for x in row])
    rows.sort(key=lambda r: int(float(r[0])))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  updated {path.name} ({len(rows)} rows, epochs {int(float(rows[0][0]))}–{int(float(rows[-1][0]))})")
    return True


def update_metrics_loss_csv(metrics_dir: Path, *, backup: bool = True) -> dict:
    """Dedupe metrics/loss.csv and optional backup."""
    loss_csv = metrics_dir / "loss.csv"
    if backup and loss_csv.exists():
        bak = metrics_dir / "loss.csv.bak"
        if not bak.exists():
            bak.write_bytes(loss_csv.read_bytes())
            print(f"  backup → {bak.name}")
    n0, n1 = dedupe_csv_by_epoch(loss_csv, CSV_HEADER)
    if n0:
        print(f"  loss.csv: {n0} → {n1} rows (unique epochs)")
    missing_ep1 = n1 > 0
    if loss_csv.exists():
        with loss_csv.open(newline="", encoding="utf-8") as f:
            eps = {int(float(r["epoch"])) for r in csv.DictReader(f) if r.get("epoch")}
        missing_ep1 = 1 not in eps
    return {"loss_rows": n1, "needs_epoch1": missing_ep1}


def update_epoch_timing_csv(metrics_dir: Path, *, backup: bool = True) -> dict:
    """Dedupe metrics/epoch_timing.csv and optional backup."""
    timing_csv = metrics_dir / "epoch_timing.csv"
    if backup and timing_csv.exists():
        bak = metrics_dir / "epoch_timing.csv.bak"
        if not bak.exists():
            bak.write_bytes(timing_csv.read_bytes())
            print(f"  backup → {bak.name}")
    n0, n1 = dedupe_csv_by_epoch(timing_csv, EPOCH_TIMING_HEADER)
    if n0:
        print(f"  epoch_timing.csv: {n0} → {n1} rows (unique epochs)")
    return {"epoch_timing_rows": n1}


def update_all_metrics_csv(metrics_dir: Path, *, backup: bool = True) -> dict:
    """Dedupe all exp041 metrics CSVs that are epoch-indexed."""
    out: dict = {}
    out.update(update_metrics_loss_csv(metrics_dir, backup=backup))
    out.update(update_epoch_timing_csv(metrics_dir, backup=backup))
    return out
