#!/usr/bin/env python3
"""Build an Obsidian knowledge-graph vault from this repository.

Purpose:
    Convert the codebase (Python modules, docs, experiment notes, AL run
    results, dataset references) into a linked Obsidian vault at
    ``thesis_vault/`` so the thesis can be written against a navigable graph
    instead of raw source trees.

Run:
    python tools/build_vault.py

Agent notes:
    What
        Walks every tracked ``*.py`` file, parses it with ``ast``, and emits one
        note per module carrying its docstring, top-level classes/functions,
        CONFIG constants, and internal imports rendered as ``[[wikilinks]]``.
        Also mirrors ``docs/*.md`` as concept notes, builds one note per
        ``experiments/*`` (config + notes.md + lineage), one per
        ``active_learning_pi/runs/*`` (decision ledger), and one per referenced
        dataset directory.
    Usage
        Edit the CONFIG block below, then ``python tools/build_vault.py``.
        Re-run after code changes; generated folders are rewritten wholesale.
    Config keys
        VAULT_DIR       — vault output directory (repo-relative)
        OWNED_DIRS      — folders the generator rewrites; everything else is
                          left alone so hand-written notes survive re-runs
        LINEAGE_HINT    — pins order only for non-numeric experiment names
        CURRENT_TRACK   — experiments marked as the active model track
        CONCEPT_CODE    — curated concept-doc -> source-path links
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT  # noqa: E402

# =============================================================================
# CONFIGURATION — edit these before running: python tools/build_vault.py
# =============================================================================

VAULT_DIR = "thesis_vault"

# Obsidian cannot open a vault over \\wsl.localhost — its file watcher fails with
# EISDIR because WSL's 9P filesystem emits no Windows change notifications. So the
# vault is mirrored onto the Windows filesystem, and Obsidian opens the mirror.
# Set to None to disable mirroring.
WINDOWS_MIRROR = "/mnt/c/Users/muthusamy/thesis_vault"

# Folders the generator owns and rewrites on every run. Anything else in the
# vault (00-Thesis/, 15-Ideas/, your own notes) is never touched.
OWNED_DIRS = ("10-Concepts", "12-Archive", "20-Experiments", "30-Code", "40-Datasets", "50-Results")

# Hand-written content. Pulled back from the mirror before each build so notes
# written in Obsidian land in the repo (and therefore in git), not overwritten.
HANDWRITTEN = ("00-Thesis", "05-Literature", "15-Ideas", "README.md")

# Copied to a fresh mirror once, then never touched again. `.obsidian/` holds the
# window layout, open tabs, and UI state that Obsidian rewrites as you work —
# re-pushing it on every build would reset your workspace mid-session.
MIRROR_SEED_ONCE = (".obsidian",)

# Per-machine sync bookkeeping; lives in the repo vault, never mirrored.
SYNC_STATE_NAME = ".vault_sync.json"

# Experiment lineage is derived automatically from the numeric prefix of each directory
# under experiments/, so newly copied-in experiments (exp001..exp036) slot into the chain
# with no edit here. This list only pins ordering for names that do not sort naturally.
LINEAGE_HINT = [
    "exp029_heat_private", "exp037_lat_change", "exp038_true_multi", "exp039_improved_heatmap",
    "exp040", "exp041", "exp042", "exp043", "exp044", "exp045", "exp046", "exp047", "exp048",
    "exp049", "exp050", "exp051_new_datas_appended", "exp052_unbounded_pearson",
    "exp053_peak_log1p_losses", "exp054_K_30", "exp055_hard_occ", "exp056_graph_vae",
    "exp057_structured_graph", "exp058_asymmetric_kl", "exp059_capacity_freq",
    "exp060_multitype_occ",
]

# Experiments older than this numeric prefix get a SUMMARY note only — no per-module
# code notes. The early GAN-era and prototype runs matter to the thesis as a narrative
# (why the approach changed), not as source. Set to 0 to document every experiment fully.
SUMMARY_ONLY_BELOW = 37

# Status labels shown on experiment notes.
CURRENT_TRACK = {"exp059_capacity_freq": "current", "exp060_multitype_occ": "exploratory"}
LEGACY = {"exp057_structured_graph", "exp058_asymmetric_kl"}

# Curated links from a concept doc (docs/<name>.md) to source files it describes.
# Paths that do not exist are skipped and reported at the end.
CONCEPT_CODE = {
    "framework-overview": [
        "pipelines/latent/optimize.py",
        "pipelines/active_learning/run.py",
        "experiments/exp059_capacity_freq/codes/train_vae_simple.py",
    ],
    "model-architecture": [
        "experiments/exp059_capacity_freq/codes/vae_poe_freq.py",
        "experiments/exp059_capacity_freq/codes/vae_multi_input_simple.py",
        "experiments/exp059_capacity_freq/codes/exp059_common.py",
    ],
    "gnn-rationale": [
        "experiments/exp059_capacity_freq/codes/graph_occ.py",
        "experiments/exp059_capacity_freq/codes/graph_imp.py",
    ],
    "latent-optimization": [
        "pipelines/latent/optimize.py",
        "pipelines/latent/find_feasible.py",
        "pipelines/latent/optimization_loader.py",
        "pipelines/latent/generate_run_report.py",
    ],
    "impedance-surrogate": [
        "pipelines/latent/optimize.py",
    ],
    "active-learning": [
        "active_learning_pi/al/pipeline.py",
        "active_learning_pi/al/inference_pool.py",
        "active_learning_pi/al/acquisition.py",
        "active_learning_pi/al/candidates.py",
        "active_learning_pi/al/build_overlay.py",
        "active_learning_pi/al/finetune_run.py",
        "pipelines/active_learning/run.py",
    ],
    "gp-error-surrogate": [
        "active_learning_pi/al/gp_error_surrogate.py",
        "active_learning_pi/al/validate_acquisition_ab.py",
        "active_learning_pi/al/per_k_acquire.py",
    ],
    "normalization-and-losses": [
        "active_learning_pi/al/robust_normalize.py",
        "active_learning_pi/al/robust_stats.py",
        "experiments/exp059_capacity_freq/codes/impedance_spectrum_loss.py",
        "experiments/exp059_capacity_freq/codes/heatmap_peak_losses.py",
        "experiments/exp059_capacity_freq/codes/physics_loss.py",
    ],
    "training-procedure": [
        "experiments/exp059_capacity_freq/codes/train_core.py",
        "experiments/exp059_capacity_freq/codes/train_vae_simple.py",
        "experiments/exp059_capacity_freq/codes/distributed_train.py",
        "experiments/exp059_capacity_freq/codes/training_guard.py",
    ],
    "dataset": [
        "pipelines/data/processing_multifreq.py",
        "libs/dataset_meta.py",
        "experiments/exp059_capacity_freq/codes/dataloader_multifreq.py",
    ],
    "data-pipeline": [
        "pipelines/data/processing_multifreq.py",
        "active_learning_pi/al/ingest_labels.py",
        "active_learning_pi/al/ecadstar.py",
        "active_learning_pi/al/peb_batch.py",
    ],
    "evaluation-metrics": [
        "experiments/exp059_capacity_freq/codes/spatial_metrics.py",
        "experiments/exp059_capacity_freq/codes/eval_spatial_metrics.py",
        "experiments/exp059_capacity_freq/codes/eval_off_anchor.py",
        "active_learning_pi/al/evaluate_cycle.py",
    ],
    "evaluation-suite": [
        "active_learning_pi/al/evaluate_report.py",
        "active_learning_pi/al/evaluate_off_anchor.py",
        "active_learning_pi/al/evaluate_hole_finding.py",
    ],
    "limitations-and-validity": [
        "active_learning_pi/al/decision_report.py",
    ],
    "multitype_peb_pipeline": [
        "active_learning_pi/al/peb_batch.py",
        "experiments/exp060_multitype_occ/codes/occupancy_binary.py",
    ],
    "cursor-handoff": [
        "pipelines/active_learning/run.py",
        "active_learning_pi/al/validate_acquisition_ab.py",
        "active_learning_pi/al/decision_report.py",
    ],
}

# Hyperparameters surfaced in the experiment note table (order preserved).
KEY_HPARAMS = [
    "latent_dim", "heatmap_private_dim", "cond_dim", "freq_fourier_features",
    "use_multiscale_film", "graph_hidden_dim", "graph_num_layers",
    "num_epochs", "batch_size", "learning_rate", "train_split", "split_by_design",
    "layout_train_prob", "occ_only_encode_prob", "occupancy_binary_decode",
    "heatmap_weight", "impedance_weight", "occupancy_weight", "cross_freq_weight",
    "data_dir", "al_overlay_data_dir", "resume_checkpoint",
]

# =============================================================================

VAULT = REPO_ROOT / VAULT_DIR
STDLIB_HINT = {
    "os", "sys", "json", "math", "re", "csv", "time", "shutil", "pathlib", "typing",
    "dataclasses", "collections", "itertools", "functools", "subprocess", "random",
    "argparse", "logging", "warnings", "copy", "glob", "ast", "abc", "enum", "io",
    "datetime", "hashlib", "traceback", "tempfile", "textwrap", "contextlib", "__future__",
}


# ── source model ─────────────────────────────────────────────────────────────
@dataclass
class ModuleInfo:
    path: str                       # repo-relative posix path
    dotted: str                     # importable dotted path
    stem: str
    note: str = ""                  # unique vault note name
    doc: str = ""
    summary: str = ""
    loc: int = 0
    classes: list[tuple[str, str, str]] = field(default_factory=list)   # name, bases, doc
    functions: list[tuple[str, str, str]] = field(default_factory=list)  # name, sig, doc
    constants: list[tuple[str, str]] = field(default_factory=list)
    imports_internal: set[str] = field(default_factory=set)   # dotted targets
    imports_external: set[str] = field(default_factory=set)
    has_config_block: bool = False
    parse_error: str = ""
    tracked: bool = True

    @property
    def group(self) -> str:
        parts = self.path.split("/")
        return "/".join(parts[:-1]) if len(parts) > 1 else "(root)"

    @property
    def experiment(self) -> str:
        """Owning experiment directory, empty for loose files under experiments/."""
        parts = self.path.split("/")
        return parts[1] if parts[0] == "experiments" and len(parts) > 2 else ""


def python_files() -> tuple[list[str], set[str]]:
    """Every .py file git would consider part of the project.

    Union of tracked files and untracked-but-not-ignored files, so work in
    progress (e.g. the residual-GP modules) still lands in the graph. Returns
    (paths, untracked_set).
    """
    def _git(*args: str) -> list[str]:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        return [f for f in out.splitlines() if f.strip()]

    try:
        tracked = _git("ls-files", "*.py")
        untracked = _git("ls-files", "--others", "--exclude-standard", "*.py")
    except (subprocess.CalledProcessError, FileNotFoundError):
        tracked = [
            p.relative_to(REPO_ROOT).as_posix()
            for p in REPO_ROOT.rglob("*.py")
            if "__pycache__" not in p.parts and ".venv" not in p.parts
        ]
        untracked = []
    files = sorted({*tracked, *untracked})
    return [f for f in files if (REPO_ROOT / f).is_file()], set(untracked)


def _no_links(text: str) -> str:
    """Neutralise stray ``[[`` in prose so docstrings cannot forge graph edges."""
    return text.replace("[[", "[ [").replace("]]", "] ]")


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        if line.strip():
            return _no_links(line.strip())
    return ""


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    return f"{node.name}({args})"


def _const_value(node: ast.AST, limit: int = 90) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        return "…"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def parse_module(rel: str) -> ModuleInfo:
    path = REPO_ROOT / rel
    src = path.read_text(encoding="utf-8", errors="replace")
    dotted = rel[:-3].replace("/", ".")
    info = ModuleInfo(path=rel, dotted=dotted, stem=Path(rel).stem, loc=src.count("\n") + 1)
    info.has_config_block = "CONFIGURATION" in src

    try:
        tree = ast.parse(src, filename=rel)
    except SyntaxError as exc:
        info.parse_error = f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})"
        return info

    info.doc = ast.get_docstring(tree) or ""
    info.summary = _first_line(info.doc)
    pkg = dotted.rsplit(".", 1)[0] if "." in dotted else ""

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(_const_value(b, 40) for b in node.bases)
            info.classes.append((node.name, bases, _first_line(ast.get_docstring(node))))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("__"):
                info.functions.append((node.name, _sig(node), _first_line(ast.get_docstring(node))))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and len(target.id) > 2:
                    info.constants.append((target.id, _const_value(node.value)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info.imports_external.add(alias.name.split(".")[0])
                info.imports_internal.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = pkg
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0] if "." in base else ""
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
                info.imports_external.add(mod.split(".")[0])
            if mod:
                info.imports_internal.add(mod)
                for alias in node.names:
                    info.imports_internal.add(f"{mod}.{alias.name}")
    return info


def assign_note_names(modules: list[ModuleInfo]) -> None:
    """Short stem when globally unique, else the full dotted path."""
    by_stem: dict[str, list[ModuleInfo]] = defaultdict(list)
    for m in modules:
        by_stem[m.stem].append(m)
    for stem, group in by_stem.items():
        if len(group) == 1 and stem not in {"README", "Home"}:
            group[0].note = stem
        else:
            for m in group:
                m.note = m.dotted


def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|#^\[\]]', "-", name).strip()


def yaml_list(values) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def write_note(rel_path: str, content: str) -> None:
    dest = VAULT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")


def load_jsonlike(path: Path) -> dict:
    """Experiment config.yaml files are JSON with '#' comment lines."""
    try:
        lines = [
            ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        return json.loads("\n".join(lines))
    except Exception:
        return {}


# ── note builders ────────────────────────────────────────────────────────────
def build_code_notes(modules: list[ModuleInfo], index: dict[str, ModuleInfo]) -> None:
    imported_by: dict[str, set[str]] = defaultdict(set)
    for m in modules:
        for target in m.imports_internal:
            hit = index.get(target)
            if hit and hit.note != m.note:
                imported_by[hit.note].add(m.note)

    for m in modules:
        links: set[str] = set()
        for target in sorted(m.imports_internal):
            hit = index.get(target)
            if hit and hit.note != m.note:
                links.add(hit.note)

        tags = ["code"]
        if m.experiment:
            tags.append(m.experiment)
        else:
            tags.append(m.path.split("/")[0].replace("-", "_"))
        if m.has_config_block:
            tags.append("runnable")
        if not m.tracked:
            tags.append("uncommitted")
        is_stub = not (m.doc or m.classes or m.functions or m.constants)
        if is_stub:
            tags.append("stub")

        lines = [
            "---",
            f"title: {m.stem}",
            "type: code",
            f"path: {m.path}",
            f"group: {m.group}",
        ]
        if m.experiment:
            lines.append(f"experiment: {m.experiment}")
        lines += [
            f"loc: {m.loc}",
            f"tags: {yaml_list(tags)}",
            "---",
            "",
            f"# {m.stem}",
            "",
        ]
        if m.summary:
            lines += [f"> {m.summary}", ""]
        lines += [f"**Source:** `{m.path}` · {m.loc} lines"]
        if not m.tracked:
            lines.append("**Git:** uncommitted — not yet tracked")
        if m.experiment:
            lines.append(f"**Experiment:** [[{m.experiment}]]")
        if m.has_config_block:
            lines.append("**Runnable:** CONFIG-only script — edit constants at top, then `python " + m.path + "`")
        lines.append("")

        if m.parse_error:
            lines += ["> [!warning] Could not parse", f"> {m.parse_error}", ""]

        if m.doc and len(m.doc.strip().splitlines()) > 1:
            lines += ["## Purpose", "", "```text", m.doc.strip(), "```", ""]

        if m.constants:
            lines += ["## Constants", "", "| Name | Value |", "|------|-------|"]
            for name, val in m.constants[:60]:
                safe = val.replace("|", "\\|")
                lines.append(f"| `{name}` | `{safe}` |")
            if len(m.constants) > 60:
                lines.append(f"| … | {len(m.constants) - 60} more |")
            lines.append("")

        if m.classes:
            lines += ["## Classes", ""]
            for name, bases, doc in m.classes:
                sig = f"`{name}({bases})`" if bases else f"`{name}`"
                lines.append(f"- **{sig}**" + (f" — {doc}" if doc else ""))
            lines.append("")

        if m.functions:
            lines += ["## Functions", ""]
            for name, sig, doc in m.functions[:80]:
                lines.append(f"- **`{sig}`**" + (f" — {doc}" if doc else ""))
            if len(m.functions) > 80:
                lines.append(f"- … {len(m.functions) - 80} more")
            lines.append("")

        if links:
            lines += ["## Imports", ""] + [f"- [[{n}]]" for n in sorted(links)] + [""]
        if imported_by.get(m.note):
            lines += ["## Imported by", ""] + [f"- [[{n}]]" for n in sorted(imported_by[m.note])] + [""]

        ext = sorted(x for x in m.imports_external if x and x not in STDLIB_HINT)
        ext = [x for x in ext if x not in {p.split("/")[0] for p in [m.path]}]
        if ext:
            lines += ["## External dependencies", "", ", ".join(f"`{e}`" for e in ext[:20]), ""]

        folder = "30-Code/" + (m.group if m.group != "(root)" else "_root")
        write_note(f"{folder}/{sanitize(m.note)}.md", "\n".join(lines))



# ── experiment lineage and era ───────────────────────────────────────────────
_GAN_RE = re.compile(r"discriminator|adversarial|wgan|gan_loss|generator_loss|net_?[dg]\b", re.I)
_VAE_RE = re.compile(r"\bvae\b|reparameter|logvar|kl_div|elbo|posterior", re.I)


def experiment_dirs() -> list[str]:
    """All experiment directories, ordered oldest -> newest by numeric prefix.

    Auto-derived so experiments copied in later (exp001..exp036) join the chain with
    no code edit. LINEAGE_HINT only pins order for names that do not sort naturally.
    """
    root = REPO_ROOT / "experiments"
    if not root.is_dir():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir()]

    def sort_key(n: str):
        hint = LINEAGE_HINT.index(n) if n in LINEAGE_HINT else None
        m = re.match(r"exp(\d+)", n)
        if m:
            return (0, int(m.group(1)), hint if hint is not None else 0, n)
        return (1, 0, 0, n)  # non-numeric names (exp_simple, …) sort last

    return sorted(names, key=sort_key)


def is_summary_only(exp: str) -> bool:
    """True when an experiment should get a summary note but no per-module notes."""
    m = re.match(r"exp(\d+)", exp)
    return bool(m) and int(m.group(1)) < SUMMARY_ONLY_BELOW


def detect_era(exp: str) -> tuple[str, int]:
    """Classify an experiment as gan / vae / unknown.

    Code is the strongest signal, but early experiments are often copied in as notes
    and checkpoints before their source, so ``notes.md`` and configs are used as a
    fallback. Returns (era, python_file_count); no code and no notes gives ("empty", 0).
    """
    edir = REPO_ROOT / "experiments" / exp
    gan = vae = files = 0
    for path in edir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        files += 1
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if _GAN_RE.search(src):
            gan += 1
        if _VAE_RE.search(src):
            vae += 1

    if gan or vae:
        return ("gan" if gan >= vae else "vae"), files

    # No code signal — fall back to prose and config.
    prose = ""
    for name in ("notes.md", "README.md", "config.yaml"):
        f = edir / name
        if f.is_file():
            prose += f.read_text(encoding="utf-8", errors="replace") + "\n"
    if prose.strip():
        g = len(_GAN_RE.findall(prose)) + len(re.findall(r"\bGAN\b", prose))
        v = len(_VAE_RE.findall(prose)) + len(re.findall(r"\bVAE\b", prose))
        if g or v:
            return ("gan" if g >= v else "vae"), files
        return "unknown", files
    return ("empty" if not files else "unknown"), files


def build_area_indexes(modules: list[ModuleInfo]) -> list[str]:
    """One index note per top-level code area, so no module is unreachable.

    Experiment modules are reachable from their experiment note already, so those
    areas are summarised rather than enumerated.
    """
    by_area: dict[str, list[ModuleInfo]] = defaultdict(list)
    for m in modules:
        by_area[m.path.split("/")[0] if "/" in m.path else "_root"].append(m)

    names: list[str] = []
    for area, mods in sorted(by_area.items()):
        if area == "experiments":
            continue
        title = f"{area} (code index)"
        by_sub: dict[str, list[ModuleInfo]] = defaultdict(list)
        for m in mods:
            by_sub[m.group].append(m)

        lines = [
            "---",
            f"title: {title}",
            "type: index",
            f"tags: {yaml_list(['index', 'code', area.replace('-', '_')])}",
            "---",
            "",
            f"# {area}/ — code index",
            "",
            f"{len(mods)} modules.",
            "",
        ]
        for sub, items in sorted(by_sub.items()):
            lines += [f"## `{sub}/`", ""]
            for m in sorted(items, key=lambda x: x.stem):
                bits = [f"- [[{m.note}]]"]
                if m.has_config_block:
                    bits.append("*(runnable)*")
                if m.summary:
                    bits.append(f"— {m.summary}")
                lines.append(" ".join(bits))
            lines.append("")
        write_note(f"30-Code/{sanitize(title)}.md", "\n".join(lines))
        names.append(title)

    exp_mods = by_area.get("experiments", [])
    if exp_mods:
        by_exp: dict[str, list[ModuleInfo]] = defaultdict(list)
        for m in exp_mods:
            by_exp[m.experiment or "(loose files)"].append(m)
        lines = [
            "---",
            "title: experiments (code index)",
            "type: index",
            "tags: [index, code, experiments]",
            "---",
            "",
            "# experiments/ — code index",
            "",
            f"{len(exp_mods)} modules across {len(by_exp)} experiments. "
            "Each experiment directory is a self-contained fork of the training stack.",
            "",
            "| Experiment | Modules |",
            "|------------|---------|",
        ]
        for exp, items in sorted(by_exp.items()):
            link = f"[[{exp}]]" if exp != "(loose files)" else exp
            lines.append(f"| {link} | {len(items)} |")
        lines.append("")
        loose = by_exp.get("(loose files)", [])
        if loose:
            lines += ["## Loose files under `experiments/`", ""]
            lines += [f"- [[{m.note}]]" for m in sorted(loose, key=lambda x: x.stem)]
            lines.append("")
        write_note("30-Code/experiments (code index).md", "\n".join(lines))
        names.append("experiments (code index)")
    return names


def build_archive_index() -> None:
    archive = REPO_ROOT / "docs" / "_archive"
    if not archive.is_dir():
        return
    docs = sorted(p.stem for p in archive.glob("*.md"))
    lines = [
        "---",
        "title: Archive Index",
        "type: index",
        "tags: [index, archive]",
        "---",
        "",
        "# Archive Index",
        "",
        f"{len(docs)} archived implementation notes mirrored from `docs/_archive/`.",
        "These record how specific fixes were made; per `docs/README.md` they are **not**",
        "thesis-citable. Kept in the graph because they often explain *why* a config value is",
        "what it is.",
        "",
    ]
    lines += [f"- [[{archive_note_name(d)}]]" for d in docs]
    lines.append("")
    write_note("12-Archive/Archive Index.md", "\n".join(lines))


def concept_note_name(stem: str) -> str:
    """docs/README.md would collide with the vault's own README — rename it."""
    return "Docs Index" if stem == "README" else stem


def archive_note_name(stem: str) -> str:
    """Archive mirrors several filenames that also exist in docs/; suffix them."""
    return f"{stem} (archived)"


def build_concept_notes(index_by_path: dict[str, ModuleInfo]) -> list[str]:
    missing: list[str] = []
    docs_dir = REPO_ROOT / "docs"
    concept_names = {p.stem for p in docs_dir.glob("*.md")}

    def convert_links(text: str) -> str:
        def repl(match: re.Match) -> str:
            label, target = match.group(1), match.group(2)
            clean = target.split("#")[0].strip()
            if clean.endswith(".md"):
                stem = Path(clean).stem
                if stem in concept_names or stem == "README":
                    return f"[[{concept_note_name(stem)}|{label}]]"
            if clean.endswith(".py"):
                norm = clean.lstrip("./").replace("../", "")
                mod = index_by_path.get(norm)
                if mod:
                    return f"[[{mod.note}|{label}]]"
            return f"{label} (`{target}`)"
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, text)

    for doc in sorted(docs_dir.glob("*.md")):
        body = doc.read_text(encoding="utf-8")
        related = []
        for src in CONCEPT_CODE.get(doc.stem, []):
            mod = index_by_path.get(src)
            if mod:
                related.append(f"- [[{mod.note}]] — `{src}`")
            else:
                missing.append(f"{doc.stem} -> {src}")
        lines = [
            "---",
            f"title: {concept_note_name(doc.stem)}",
            "type: concept",
            f"source: docs/{doc.name}",
            "tags: [concept, thesis]",
            "---",
            "",
            f"> [!info] Mirror of `docs/{doc.name}` — edit the source file, then re-run `tools/build_vault.py`.",
            "",
            convert_links(body).strip(),
            "",
        ]
        if related:
            lines += ["", "## Implemented by", ""] + related + [""]
        write_note(f"10-Concepts/{sanitize(concept_note_name(doc.stem))}.md", "\n".join(lines))

    archive = docs_dir / "_archive"
    if archive.is_dir():
        for doc in sorted(archive.glob("*.md")):
            body = doc.read_text(encoding="utf-8")
            lines = [
                "---",
                f"title: {archive_note_name(doc.stem)}",
                "type: archive",
                f"source: docs/_archive/{doc.name}",
                "tags: [archive]",
                "---",
                "",
                "> [!caution] Archived implementation note — not for thesis citation.",
                "",
                convert_links(body).strip(),
                "",
            ]
            write_note(f"12-Archive/{sanitize(archive_note_name(doc.stem))}.md", "\n".join(lines))
    return missing


def build_experiment_notes(modules: list[ModuleInfo]) -> list[str]:
    by_exp: dict[str, list[ModuleInfo]] = defaultdict(list)
    for m in modules:
        if m.experiment:
            by_exp[m.experiment].append(m)

    order = experiment_dirs()
    names: list[str] = []
    for exp in order:
        edir = REPO_ROOT / "experiments" / exp
        cfg = load_jsonlike(edir / "config.yaml")
        notes_md = edir / "notes.md"
        idx = order.index(exp)
        prev_exp = order[idx - 1] if idx > 0 else ""
        next_exp = order[idx + 1] if idx < len(order) - 1 else ""
        status = CURRENT_TRACK.get(exp, "legacy" if exp in LEGACY else "historical")
        era, nfiles = detect_era(exp)

        tags = ["experiment", exp, status, f"era-{era}"]
        lines = [
            "---",
            f"title: {exp}",
            "type: experiment",
            f"status: {status}",
            f"era: {era}",
            f"tags: {yaml_list(tags)}",
            "---",
            "",
            f"# {exp}",
            "",
        ]
        chain = []
        if prev_exp:
            chain.append(f"[[{prev_exp}]]")
        chain.append(f"**{exp}**")
        if next_exp:
            chain.append(f"[[{next_exp}]]")
        era_label = {"gan": "GAN era", "vae": "VAE era", "empty": "no code on disk",
                     "unknown": "era undetermined"}[era]
        lines += [f"**Lineage:** {' → '.join(chain)}",
                  f"**Status:** {status} · **Era:** {era_label} ({nfiles} Python files)", ""]

        if cfg:
            rows = [(k, cfg[k]) for k in KEY_HPARAMS if k in cfg]
            if rows:
                lines += ["## Key hyperparameters", "", "| Key | Value |", "|-----|-------|"]
                lines += [f"| `{k}` | `{v}` |" for k, v in rows]
                lines += ["", f"*Full config: `experiments/{exp}/config.yaml` ({len(cfg)} keys)*", ""]

        variants = sorted(p.name for p in edir.glob("config*.yaml") if p.name != "config.yaml")
        if variants:
            lines += ["## Config variants", ""] + [f"- `{v}`" for v in variants] + [""]

        if notes_md.is_file():
            lines += [f"## Notes (from `experiments/{exp}/notes.md`)", "",
                      notes_md.read_text(encoding="utf-8").strip(), ""]

        mods = sorted(by_exp.get(exp, []), key=lambda m: m.path)
        if mods and is_summary_only(exp):
            # Summary-only era: inventory the source inline rather than minting a note per
            # file, so the early arc stays legible without flooding the graph.
            total_loc = sum(m.loc for m in mods)
            lines += ["## Source inventory", "",
                      f"*Summary-only experiment: {len(mods)} Python files, {total_loc} lines. "
                      f"No per-module notes are generated for experiments before "
                      f"`exp{SUMMARY_ONLY_BELOW:03d}` — read the source directly at "
                      f"`experiments/{exp}/`.*", "",
                      "| File | Lines | Purpose |", "|------|-------|---------|"]
            for m in mods:
                rel = m.path.split(f"{exp}/", 1)[-1]
                lines.append(f"| `{rel}` | {m.loc} | {m.summary or ''} |")
            lines.append("")
        elif mods:
            lines += ["## Code modules", ""]
            lines += [f"- [[{m.note}]]" + (f" — {m.summary}" if m.summary else "") for m in mods]
            lines.append("")

        metrics = edir / "metrics"
        if metrics.is_dir():
            files = sorted(p.name for p in metrics.iterdir() if p.is_file())[:15]
            if files:
                lines += ["## Metrics artifacts", "",
                          f"`experiments/{exp}/metrics/`", ""] + [f"- `{f}`" for f in files] + [""]

        ckpt = edir / "checkpoints"
        if ckpt.is_dir():
            found = sorted(p.name for p in ckpt.glob("*.pt"))
            lines += ["## Checkpoints", "",
                      (", ".join(f"`{c}`" for c in found) if found else "*none on disk (untracked)*"), ""]

        write_note(f"20-Experiments/{sanitize(exp)}.md", "\n".join(lines))
        names.append(exp)
    return names



def build_lineage_note(order: list[str]) -> None:
    """Narrative index of the whole experiment arc, grouped by detected era.

    This is where the GAN -> VAE transition becomes visible as a story rather than
    a directory listing. Thesis-relevant: it is the empirical history behind the
    architectural choice, which Chapter 2 has to justify.
    """
    rows = []
    for exp in order:
        era, nfiles = detect_era(exp)
        edir = REPO_ROOT / "experiments" / exp
        note = (edir / "notes.md")
        headline = ""
        if note.is_file():
            for line in note.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    headline = s[:110]
                    break
        rows.append((exp, era, nfiles, headline))

    eras = [r[1] for r in rows]
    switch = ""
    for i in range(1, len(rows)):
        if eras[i - 1] == "gan" and eras[i] == "vae":
            switch = rows[i][0]
            break

    lines = [
        "---", "title: Experiment Lineage", "type: moc", "tags: [moc, experiment, lineage]", "---",
        "", "# Experiment Lineage", "",
        f"{len(rows)} experiment directories under `experiments/`, ordered by numeric prefix.",
        "Era is detected from the source: GAN markers (discriminator, adversarial, WGAN) versus",
        "VAE markers (reparameterisation, logvar, KL, posterior).", "",
    ]
    counts = {e: eras.count(e) for e in ("gan", "vae", "unknown", "empty")}
    lines += [f"**Era split:** {counts['gan']} GAN · {counts['vae']} VAE · "
              f"{counts['unknown']} undetermined · {counts['empty']} empty", ""]
    if switch:
        lines += [f"**GAN → VAE transition detected at [[{switch}]].** This is the methodological",
                  "turning point of the thesis: the empirical reason the framework is generative-",
                  "surrogate rather than adversarial. Chapter 2 should argue it from these runs, not",
                  "from first principles alone.", ""]
    else:
        lines += ["> [!note] No GAN → VAE transition detected yet.",
                  "> Either the early experiments are not copied in, or their code carries no GAN",
                  "> markers. Re-run `tools/build_vault.py` after copying exp001–exp036.", ""]

    lines += ["## The arc", "", "| # | Experiment | Era | Files | Headline |",
              "|---|-----------|-----|-------|----------|"]
    for i, (exp, era, nfiles, headline) in enumerate(rows, 1):
        badge = {"gan": "GAN", "vae": "VAE", "unknown": "?", "empty": "—"}[era]
        lines.append(f"| {i} | [[{exp}]] | {badge} | {nfiles} | {headline} |")
    lines += ["", "## Related", "", "- [[experiment-lineage]] — the curated thesis narrative in `docs/`",
              "- [[Thesis Outline]] — where the arc is argued", ""]
    write_note("20-Experiments/Experiment Lineage.md", "\n".join(lines))


def build_result_notes() -> list[str]:
    runs_dir = REPO_ROOT / "active_learning_pi" / "runs"
    names: list[str] = []
    if not runs_dir.is_dir():
        return names
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        exp_guess = next((e for e in experiment_dirs() if e.split("_")[0] in run.name), "")
        iters = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name.startswith("iter_"))
        lines = [
            "---",
            f"title: {run.name}",
            "type: result",
            f"tags: {yaml_list(['result', 'active-learning'] + ([exp_guess] if exp_guess else []))}",
            "---",
            "",
            f"# {run.name}",
            "",
            f"**Path:** `active_learning_pi/runs/{run.name}/`",
        ]
        if exp_guess:
            lines.append(f"**Model:** [[{exp_guess}]]")
        lines += [f"**Iterations:** {len(iters)}" + (f" ({iters[0]} … {iters[-1]})" if iters else ""), ""]

        state = run / "state.json"
        if state.is_file():
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
                lines += ["## State", "", "| Key | Value |", "|-----|-------|"]
                for k, v in list(data.items())[:20]:
                    if not isinstance(v, (dict, list)):
                        lines.append(f"| `{k}` | `{v}` |")
                lines.append("")
            except Exception:
                pass

        ledger = run / "decision_ledger.json"
        if ledger.is_file():
            try:
                data = json.loads(ledger.read_text(encoding="utf-8"))
                claims = data.get("claims") or data.get("entries") or []
                if isinstance(claims, list) and claims:
                    lines += ["## Decision ledger", "", "| Claim | Verdict |", "|-------|---------|"]
                    for c in claims[:25]:
                        if isinstance(c, dict):
                            name = c.get("claim") or c.get("name") or c.get("id") or "?"
                            verdict = c.get("verdict") or c.get("status") or "?"
                            lines.append(f"| {name} | **{verdict}** |")
                    lines.append("")
                elif isinstance(claims, dict):
                    lines += ["## Decision ledger", "", "| Claim | Verdict |", "|-------|---------|"]
                    for name, c in list(claims.items())[:25]:
                        verdict = c.get("verdict") if isinstance(c, dict) else c
                        lines.append(f"| {name} | **{verdict}** |")
                    lines.append("")
            except Exception:
                pass

        report = run / "DECISION_REPORT.md"
        if report.is_file():
            lines += ["## Decision report", "",
                      f"*Mirror of `active_learning_pi/runs/{run.name}/DECISION_REPORT.md`*", "",
                      report.read_text(encoding="utf-8").strip(), ""]

        reports = sorted(p.name for p in run.glob("LATEST_*.md"))
        if reports:
            lines += ["## Cycle reports on disk", ""] + [f"- `{r}`" for r in reports] + [""]

        write_note(f"50-Results/{sanitize(run.name)}.md", "\n".join(lines))
        names.append(run.name)
    return names


def build_dataset_notes() -> list[str]:
    """Datasets are untracked; derive notes from every config that references them."""
    refs: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r"datasets/[A-Za-z0-9_\-./]+")
    search_roots = [
        REPO_ROOT / "active_learning_pi" / "config",
        REPO_ROOT / "experiments",
        REPO_ROOT / "configs",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".json", ".yaml", ".yml"} or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for hit in pattern.findall(text):
                parts = hit.split("/")
                if len(parts) >= 2:
                    refs["/".join(parts[:2])].add(path.relative_to(REPO_ROOT).as_posix())

    names: list[str] = []
    for dataset, sources in sorted(refs.items()):
        name = dataset.split("/")[1]
        on_disk = (REPO_ROOT / dataset).is_dir()
        lines = [
            "---",
            f"title: {name}",
            "type: dataset",
            f"path: {dataset}",
            f"tags: {yaml_list(['dataset'])}",
            "---",
            "",
            f"# {name}",
            "",
            f"**Path:** `{dataset}/` — {'present on disk' if on_disk else 'not on disk'} (untracked by git)",
            "",
        ]
        meta = REPO_ROOT / dataset / "dataset_meta.json"
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                lines += ["## dataset_meta.json", "", "| Key | Value |", "|-----|-------|"]
                for k, v in list(data.items())[:20]:
                    if isinstance(v, (dict, list)):
                        v = json.dumps(v)[:80]
                    lines.append(f"| `{k}` | `{v}` |")
                lines.append("")
            except Exception:
                pass
        lines += ["## Referenced by", ""] + [f"- `{s}`" for s in sorted(sources)] + [""]
        write_note(f"40-Datasets/{sanitize(name)}.md", "\n".join(lines))
        names.append(name)
    return names


def build_home(modules: list[ModuleInfo], experiments: list[str],
               results: list[str], datasets: list[str], areas: list[str]) -> None:
    groups: dict[str, int] = defaultdict(int)
    for m in modules:
        groups[m.path.split("/")[0] if "/" in m.path else "_root"] += 1
    concepts = sorted(p.stem for p in (REPO_ROOT / "docs").glob("*.md"))
    current = [e for e in experiments if CURRENT_TRACK.get(e)]

    lines = [
        "---",
        "title: Home",
        "type: moc",
        "tags: [moc]",
        "---",
        "",
        "# PDN Generative-Surrogate Thesis Vault",
        "",
        "Knowledge graph generated from the `genai_pdn` codebase by `tools/build_vault.py`.",
        "Re-run that script after code changes — it rewrites `10-Concepts`, `12-Archive`,",
        "`20-Experiments`, `30-Code`, `40-Datasets`, `50-Results` and leaves `00-Thesis/`",
        "and `15-Ideas/` (your own writing) untouched.",
        "",
        "## Start here",
        "",
        "- [[Thesis Outline]] — chapter map, what to write where",
        "- [[Claim Ledger]] — every claim with its evidence artifact",
        "- [[Reading Paths]] — guided tours through the graph",
        "- [[Literature Review]] — curated papers, mapped to chapters",
        "- [[Experiment Lineage]] — the full arc, GAN era through to the current model",
        "- [[framework-overview]] — the framework in one page",
        "",
        "## Current model track",
        "",
    ]
    lines += [f"- [[{e}]] — **{CURRENT_TRACK[e]}**" for e in current]
    lines += [
        "",
        "## Concepts",
        "",
    ]
    lines += [f"- [[{concept_note_name(c)}]]" for c in concepts if c != "README"]
    lines += [
        "",
        "## Experiments",
        "",
        ", ".join(f"[[{e}]]" for e in experiments),
        "",
        "## Results",
        "",
    ]
    lines += [f"- [[{r}]]" for r in results]
    lines += ["", "## Datasets", ""]
    lines += [f"- [[{d}]]" for d in datasets]
    lines += [
        "",
        "## Code by area",
        "",
        "| Area | Modules | Index |",
        "|------|---------|-------|",
    ]
    area_lookup = {a.split(" ")[0]: a for a in areas}
    for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        idx = f"[[{area_lookup[g]}]]" if g in area_lookup else "—"
        lines.append(f"| `{g}/` | {n} | {idx} |")
    lines += ["", "## Archive", "", "- [[Archive Index]] — historical implementation notes (not citable)"]
    lines += [
        "",
        f"**Totals:** {len(modules)} modules · {len(concepts)} concepts · "
        f"{len(experiments)} experiments · {len(results)} AL runs · {len(datasets)} datasets",
        "",
    ]
    write_note("Home.md", "\n".join(lines))


def write_obsidian_config() -> None:
    """Graph colour groups so the vault is readable on first open."""
    cfg = {
        "collapse-filter": False, "search": "", "showTags": True, "showAttachments": False,
        "hideUnresolved": True, "showOrphans": True,
        "collapse-color-groups": False,
        "colorGroups": [
            {"query": "tag:#concept", "color": {"a": 1, "rgb": 5431378}},
            {"query": "tag:#experiment", "color": {"a": 1, "rgb": 14701138}},
            {"query": "tag:#result", "color": {"a": 1, "rgb": 11621088}},
            {"query": "tag:#dataset", "color": {"a": 1, "rgb": 5419488}},
            {"query": "tag:#moc", "color": {"a": 1, "rgb": 16217871}},
            {"query": "tag:#archive", "color": {"a": 1, "rgb": 8355711}},
            {"query": "tag:#stub", "color": {"a": 1, "rgb": 5592405}},
        ],
        "collapse-display": False, "showArrow": False, "textFadeMultiplier": 0,
        "nodeSizeMultiplier": 1.2, "lineSizeMultiplier": 1,
        "collapse-forces": False, "centerStrength": 0.42, "repelStrength": 12,
        "linkStrength": 0.6, "linkDistance": 190, "scale": 0.55,
    }
    write_note(".obsidian/graph.json", json.dumps(cfg, indent=2))
    write_note(".obsidian/app.json", json.dumps(
        {"attachmentFolderPath": "_attachments", "newLinkFormat": "shortest",
         "useMarkdownLinks": False, "alwaysUpdateLinks": True}, indent=2))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handwritten_files(root: Path) -> dict[str, Path]:
    """Map vault-relative path -> file, for every hand-written note under *root*."""
    found: dict[str, Path] = {}
    for item in HANDWRITTEN:
        src = root / item
        if src.is_dir():
            for path in src.rglob("*.md"):
                found[path.relative_to(root).as_posix()] = path
        elif src.is_file():
            found[item] = src
    return found


def sync_handwritten(mirror: Path) -> tuple[list[str], list[str]]:
    """Three-way sync of hand-written notes between repo and mirror.

    Obsidian edits land in the mirror; edits made here land in the repo. Comparing
    both against the hashes recorded at the last sync tells us which side actually
    changed, so neither can silently clobber the other. If both changed, the
    mirror's version is preserved beside the repo's as a ``.conflict-*`` file
    rather than either being dropped. Returns (pulled, conflicts).
    """
    state_path = VAULT / SYNC_STATE_NAME
    try:
        last: dict[str, str] = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        last = {}

    repo_files = _handwritten_files(VAULT)
    mirror_files = _handwritten_files(mirror) if mirror.is_dir() else {}
    pulled: list[str] = []
    conflicts: list[str] = []

    for rel in sorted(set(repo_files) | set(mirror_files)):
        repo_path, mirror_path = repo_files.get(rel), mirror_files.get(rel)
        repo_hash = _sha(repo_path) if repo_path else None
        mirror_hash = _sha(mirror_path) if mirror_path else None
        if repo_hash == mirror_hash:
            continue

        prev = last.get(rel)
        repo_changed = repo_hash != prev
        mirror_changed = mirror_hash != prev

        if mirror_changed and not repo_changed and mirror_path:
            dest = VAULT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mirror_path, dest)
            pulled.append(rel)
        elif repo_changed and mirror_changed and mirror_path and repo_path:
            keep = VAULT / f"{rel.rsplit('.md', 1)[0]}.conflict-from-obsidian.md"
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mirror_path, keep)
            conflicts.append(rel)
        # repo-only change (or mirror file deleted): push_to_mirror carries it out.

    return pulled, conflicts


def save_sync_state() -> None:
    """Record hashes of hand-written notes so the next run can detect real edits."""
    state = {rel: _sha(path) for rel, path in _handwritten_files(VAULT).items()}
    (VAULT / SYNC_STATE_NAME).write_text(json.dumps(state, indent=2), encoding="utf-8")


def push_to_mirror(mirror: Path) -> tuple[int, list[str]]:
    """Mirror the repo vault onto the Windows filesystem for Obsidian.

    Generated folders are replaced wholesale; hand-written folders were already
    reconciled by pull_handwritten_from_mirror, so a plain copy is safe. Entries
    in MIRROR_SEED_ONCE are written only when absent, leaving Obsidian's own
    state alone on every later run. Returns (files_copied, seeded_names).
    """
    mirror.mkdir(parents=True, exist_ok=True)

    seeded: list[str] = []
    for name in MIRROR_SEED_ONCE:
        src, dest = VAULT / name, mirror / name
        if dest.exists() or not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        seeded.append(name)

    for owned in OWNED_DIRS:
        target = mirror / owned
        if target.exists():
            shutil.rmtree(target)

    count = 0
    for path in VAULT.rglob("*"):
        rel = path.relative_to(VAULT)
        if rel.parts[0] in MIRROR_SEED_ONCE or rel.name == SYNC_STATE_NAME:
            continue
        target = mirror / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != path.read_bytes():
            shutil.copy2(path, target)
        count += 1
    return count, seeded


def main() -> int:
    print(f"Repo: {REPO_ROOT}")
    files, untracked = python_files()
    print(f"Parsing {len(files)} Python files ({len(untracked)} uncommitted)…")

    mirror = Path(WINDOWS_MIRROR) if WINDOWS_MIRROR else None
    if mirror:
        pulled, conflicts = sync_handwritten(mirror)
        for item in pulled:
            print(f"  < pulled from Obsidian: {item}")
        for item in conflicts:
            print(f"  ! CONFLICT (edited on both sides): {item}")
            print("    kept the Obsidian version beside it as .conflict-from-obsidian.md")

    modules = [parse_module(f) for f in files]
    for m in modules:
        m.tracked = m.path not in untracked
    assign_note_names(modules)

    index: dict[str, ModuleInfo] = {}
    for m in modules:
        index[m.dotted] = m
        if m.stem == "__init__":
            index[m.dotted.rsplit(".", 1)[0]] = m
    index_by_path = {m.path: m for m in modules}

    for owned in OWNED_DIRS:
        target = VAULT / owned
        if target.exists():
            shutil.rmtree(target)

    documented = [m for m in modules if not (m.experiment and is_summary_only(m.experiment))]
    summarised = len(modules) - len(documented)
    build_code_notes(documented, index)
    areas = build_area_indexes(documented)
    build_archive_index()
    missing = build_concept_notes(index_by_path)
    experiments = build_experiment_notes(modules)
    build_lineage_note(experiment_dirs())
    results = build_result_notes()
    datasets = build_dataset_notes()
    build_home(modules, experiments, results, datasets, areas)
    write_obsidian_config()

    errors = [m for m in modules if m.parse_error]
    stubs = [m for m in modules if not (m.doc or m.classes or m.functions or m.constants)]
    total = sum(1 for _ in VAULT.rglob("*.md"))
    print(f"  code notes      : {len(documented)}"
          + (f"  (+{summarised} files summarised only, pre-exp{SUMMARY_ONLY_BELOW:03d})" if summarised else ""))
    print(f"  concept notes   : {len(list((REPO_ROOT / 'docs').glob('*.md')))}")
    print(f"  experiment notes: {len(experiments)}")
    print(f"  result notes    : {len(results)}")
    print(f"  dataset notes   : {len(datasets)}")
    print(f"  total .md       : {total}")
    print(f"  parse errors    : {len(errors)}")
    for m in errors[:10]:
        print(f"    ! {m.path}: {m.parse_error}")
    print(f"  stub modules    : {len(stubs)} (tagged #stub)")
    if missing:
        print(f"  unresolved concept->code links: {len(missing)}")
        for item in missing[:10]:
            print(f"    ? {item}")
    print(f"\nVault ready: {VAULT}")

    if mirror:
        copied, seeded = push_to_mirror(mirror)
        win = "C:" + str(mirror).replace("/mnt/c", "").replace("/", "\\")
        print(f"  mirrored {copied} files -> {mirror}")
        if seeded:
            print(f"  seeded (first run only): {', '.join(seeded)}")
        print(f"  left untouched: {', '.join(MIRROR_SEED_ONCE)} (Obsidian's own state)")
        print(f"  open this path in Obsidian: {win}")
        save_sync_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
