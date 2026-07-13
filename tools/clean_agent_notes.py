#!/usr/bin/env python3
"""Remove CLI/argparse wording from module docstrings and user-facing messages."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (pattern, replacement) — applied to file text
DOC_REPLACEMENTS: list[tuple[str, str]] = [
    (r"- Type: CLI dataset maintenance script\.", "- Type: CONFIG entry script — extract layouts between multifreq folders."),
    (r"- Type: CLI analysis script\.", "- Type: CONFIG entry script — dataset statistics and quality scoring."),
    (r"- Type: CLI data-pipeline script \(legacy single-freq; prefer ``processing_multifreq.py``\)\.",
     "- Type: CONFIG entry script — legacy single-frequency builder (prefer ``processing_multifreq.py``)."),
    (r"- Type: CLI thin wrapper \(prefer run_all_k\.py for current experiments\)",
     "- Type: CONFIG wrapper — single experiment path (prefer ``run_all_k.py`` for sweeps)"),
    (r"- Type: CLI generator \(single K; legacy — see run_all_k\.py\)",
     "- Type: CONFIG generator — one K at a time (legacy; see ``run_all_k.py``)"),
    (r"- Type: CLI single-K visualization", "- Type: CONFIG script — single-K occupancy visualization"),
    (r"- Type: CLI single-K comparison \(superseded in part by comparison/compare\.py\)",
     "- Type: CONFIG script — single-K comparison (see also ``comparison/compare.py``)"),
    (r"- Type: CLI evaluation script\.", "- Type: CONFIG evaluation script."),
    (r"- Type: CLI utility script \(top-level constants or argparse\)\.",
     "- Type: CONFIG utility — edit top-level constants before running."),
    (r"- Type: CLI utility script \(top-level constants\)\.", "- Type: CONFIG utility — edit top-level constants before running."),
    (r"- Type: training entrypoint \(no argparse; config-driven\)\.", "- Type: CONFIG training entrypoint — reads ``config.yaml`` and env overrides."),
    (r"- Type: inference script \(config via top-level constants, not argparse\)\.",
     "- Type: inference module — top-level CONFIG when run as ``__main__``; imported by AL and latent pipelines."),
    (r"Run pipeline from fixed config \(no argparse\)\. Called by",
     "Run pipeline from CONFIG. Called by"),
    (r"Import only — config path passed via ``--config`` on ``pipelines/active_learning/run\.py``\.",
     "Import only — path set via ``CONFIG_PATH`` in ``pipelines/active_learning/run.py``."),
    (r"# CONFIGURATION \(standalone CLI defaults\)", "# CONFIGURATION — edit before running as __main__"),
    (r"same format as CLI\)", "same format as ``vae_novelty_report.py`` main)"),
    (r"Active-learning CLI \(run\.py\)", "Active-learning entry (``run.py``)"),
    (r"\*\*CLI entry\*\*", "**CONFIG entry**"),
    (r"AL CLI entry", "AL CONFIG entry"),
]

MSG_REPLACEMENTS: list[tuple[str, str]] = [
    ("Pass --overwrite to replace, or --resume to add missing files.",
     "Set OVERWRITE=True to replace, or RESUME=True to add missing files."),
    ("Use --execute --overwrite to replace or --execute --resume to fill gaps.",
     "Set EXECUTE=True with OVERWRITE=True to replace, or EXECUTE=True with RESUME=True to fill gaps."),
    ("Dry-run only. Set EXECUTE=True or pass --execute to create the folder.",
     "Dry-run only. Set EXECUTE=True in the CONFIG block to create the folder."),
    ("Use only one of --overwrite or --resume, not both.",
     "Set only one of OVERWRITE or RESUME, not both."),
    ("run without --plots-only to backfill epoch 1.",
     "set PLOTS_ONLY=False to backfill epoch 1."),
    ("No K folders were scored (check --out-root and K range)",
     "No K folders were scored (check OUT_ROOT and K range in CONFIG)"),
    ("Use `--max-train` to cap dataset size per K for speed.",
     "Set MAX_TRAIN in CONFIG to cap dataset size per K for speed."),
]

# Strip Usage/Example blocks that show CLI flags from top docstrings
USAGE_BLOCK = re.compile(
    r"\n(?:Usage|Example|Run\n---)\n(?:.*\n)*?(?=\n(?:Outputs|Dependencies|Agent notes|\"\"\"|\Z))",
    re.MULTILINE,
)


def _clean_docstring_usage(text: str) -> str:
    """Remove multi-line Usage/Example sections that only show --flags."""
    if '"""' not in text[:2000]:
        return text
    m = re.match(r'(?s)(.*?""".*?)(?=\nfrom |\nimport )', text)
    if not m:
        return text
    head = m.group(1)
    if "--" not in head and "python " not in head.lower():
        return text
    # Remove lines with --flags inside first docstring only
    lines = head.splitlines()
    out: list[str] = []
    in_doc = False
    for line in lines:
        if '"""' in line and not in_doc:
            in_doc = True
            out.append(line)
            continue
        if in_doc and '"""' in line and line.strip().endswith('"""'):
            out.append(line)
            break
        if in_doc and re.search(r"\s--[a-z]", line):
            continue
        if in_doc and line.strip().startswith("python ") and " --" in line:
            # keep simple run line without flags
            simple = line.split(" --")[0]
            if simple.strip() not in {l.strip() for l in out}:
                out.append(simple)
            continue
        out.append(line)
    new_head = "\n".join(out)
    return text.replace(head, new_head, 1)


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for pat, repl in DOC_REPLACEMENTS:
        text = re.sub(pat, repl, text)
    for old, new in MSG_REPLACEMENTS:
        text = text.replace(old, new)
    text = _clean_docstring_usage(text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    roots = [
        REPO / "pipelines",
        REPO / "scrap",
        REPO / "libs",
        REPO / "experiments",
        REPO / "evaluation",
        REPO / "visualization",
        REPO / "active_learning_pi",
        REPO / "src_vae" / "others" / "model_to_config.py",
    ]
    changed = 0
    for root in roots:
        if root.is_file():
            files = [root]
        else:
            files = list(root.rglob("*.py"))
        for p in files:
            if "__pycache__" in str(p):
                continue
            if process_file(p):
                print("updated", p.relative_to(REPO))
                changed += 1
    print(f"Done — {changed} files updated.")


if __name__ == "__main__":
    main()
