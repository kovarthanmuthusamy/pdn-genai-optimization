---
title: decision_report
type: code
path: active_learning_pi/al/decision_report.py
group: active_learning_pi/al
loc: 659
tags: [code, active_learning_pi, uncommitted]
---

# decision_report

> Decision ledger: collect numerical evals → PASS/FAIL/UNCERTAIN claims.

**Source:** `active_learning_pi/al/decision_report.py` · 659 lines
**Git:** uncommitted — not yet tracked

## Purpose

```text
Decision ledger: collect numerical evals → PASS/FAIL/UNCERTAIN claims.

Every AL workflow step that produces numbers should feed this ledger. At the end
of a cycle (or after acquisition A/B), write a decision report that answers:

- Did acquisition / uncertainty scoring point in the right direction?
- Did fine-tune improve physical (ECAD) metrics?
- Is the equal-budget A/B claim (GP vs random / MC) supported?
- Are sample sizes / budgets sufficient to trust the claim?
```

## Constants

| Name | Value |
|------|-------|
| `DECISION_LEDGER` | `'decision_ledger.json'` |
| `DECISION_REPORT_MD` | `'DECISION_REPORT.md'` |
| `CYCLE_SUMMARY` | `'eval_cycle_summary.json'` |
| `ACQUISITION_RANK` | `'acquisition_rank_quality.json'` |
| `DEFAULT_AB_LATEST` | `'active_learning_pi/runs/acquisition_ab_validation_exp059/LATEST_acquisition_ab.json'` |
| `DEFAULT_HOLE_FINDING` | `'active_learning_pi/runs/hole_finding_exp059/LATEST_hole_finding.json'` |

## Functions

- **`_status(ok: bool | None, *, missing: bool=False)`**
- **`_fmt(v: Any, *, digits: int=4)`**
- **`_load_optional(path: Path)`**
- **`_claim(claim_id: str, statement: str, status: str, *, evidence: dict[str, Any] | None=None, note: str='')`**
- **`build_claims_from_cycle(summary: dict[str, Any], *, acquisition_ab: dict[str, Any] | None=None, hole_finding: dict[str, Any] | None=None, min_rank_n: int=30, min_ecad_n: int=20)`** — Turn one cycle's numericals into explicit decision claims.
- **`aggregate_verdict(claims: list[dict[str, Any]])`** — Roll-up: blocking FAILs vs soft UNCERTAIN gaps.
- **`resolve_acquisition_ab(groot: Path, cfg: dict, *, explicit_path: str | Path | None=None)`** — Find latest equal-budget A/B JSON (config path → default exp059 path).
- **`resolve_hole_finding(groot: Path, cfg: dict)`**
- **`build_decision_ledger(cfg: dict, groot: Path, iteration: int, *, summary: dict[str, Any] | None=None, acquisition_ab: dict[str, Any] | None=None, ab_path: str | None=None, hole_finding: dict[str, Any] | None=None)`**
- **`render_decision_markdown(ledger: dict[str, Any])`**
- **`write_decision_report(cfg: dict, groot: Path, iteration: int, *, summary: dict[str, Any] | None=None)`** — Write decision_ledger.json + DECISION_REPORT.md under the run dir (and iter copy).
- **`cmd_write_decision_report(cfg: dict, groot: Path, iteration: int)`**
- **`register_acquisition_ab(cfg: dict, groot: Path, ab_report: dict[str, Any], *, src_path: Path | None=None)`** — Copy A/B JSON into the AL run dir so the next decision report can link it.

## Imports

- [[active_learning_pi.al.paths]]
- [[config]]

## Imported by

- [[evaluate_cycle]]
- [[evaluate_report]]
- [[pipeline]]
- [[validate_acquisition_ab]]
