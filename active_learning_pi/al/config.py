"""Load/save active-learning pipeline configuration (JSON or YAML).

Run:
    Import only — path set via ``CONFIG_PATH`` in ``pipelines/active_learning/run.py``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from active_learning_pi.al.paths import al_root, gan_root


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        cfg_dir = al_root() / "config"
        json_path = cfg_dir / "default.json"
        yaml_path = cfg_dir / "default.yaml"
        path = json_path if json_path.is_file() else yaml_path
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML required for .yaml config. Use config/default.json or pip install pyyaml"
                ) from exc
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    groot = gan_root()
    if not cfg.get("gan_root"):
        cfg["gan_root"] = str(groot)
    else:
        cfg["gan_root"] = str(Path(cfg["gan_root"]).resolve())
    return cfg


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
