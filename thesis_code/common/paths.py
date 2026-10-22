"""Output folders and checked input paths.

What: out_dir(stage) is the only place an output folder is made; need() stops with the name of the
    setting to edit when an input is missing; result_path() finds a Chapter 4 result and plan_file()
    a file of the sampling plan, new output first. A Chapter 4 result read from the saved results
    is announced, since it belongs to the thesis model, not to other weights set in settings.py.
"""
from pathlib import Path

from thesis_code.common import settings


def out_dir(stage: str) -> Path:
    p = Path(settings.OUTPUT_DIR) / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


def need(path: Path, setting: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{setting} = {path} does not exist -- edit thesis_code/common/settings.py")
    return path


def result_path(name: str) -> Path:
    new = Path(settings.OUTPUT_DIR) / "s12_evaluation" / name
    if new.exists():
        return new
    saved = Path(settings.SAVED_RESULTS_DIR) / name
    print(f"note: {name} was not made in this output folder; reading the saved result {saved}")
    return saved


def plan_file(name: str) -> Path:
    new = Path(settings.OUTPUT_DIR) / "s01_generation" / name
    return new if new.is_file() else Path(settings.PLAN_DIR) / name
