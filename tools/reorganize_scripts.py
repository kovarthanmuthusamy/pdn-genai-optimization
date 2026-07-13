#!/usr/bin/env python3
"""One-time layout migration: move scripts into pipelines/ + libs/, leave compat shims."""
from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (old_relative, new_relative)
MOVES: list[tuple[str, str]] = [
    # libs — shared data-creation modules
    ("Data_Creation/heatmap.py", "libs/data_creation/heatmap.py"),
    ("Data_Creation/impedance.py", "libs/data_creation/impedance.py"),
    ("Data_Creation/occupancy.py", "libs/data_creation/occupancy.py"),
    ("Data_Creation/csv_to_occupancy.py", "libs/data_creation/csv_to_occupancy.py"),
    # pipelines — data creation CLIs
    ("Data_Creation/Data_processing.py", "pipelines/data/processing_single.py"),
    ("Data_Creation/Data_processing_multifreq.py", "pipelines/data/processing_multifreq.py"),
    ("Data_Creation/Data_processing_eval.py", "pipelines/data/processing_eval.py"),
    ("Data_Creation/verify_layout_store.py", "pipelines/data/verify_layout_store.py"),
    ("Data_Creation/visualize_sample.py", "pipelines/data/visualize_sample.py"),
    ("Data_Creation/data_check_stats.py", "pipelines/data/data_check_stats.py"),
    ("Data_Creation/occ_grid.py", "pipelines/data/occ_grid.py"),
    # pipelines — dataset ops
    ("datasets/build_multifreq_gmax_dataset.py", "pipelines/dataset/build_gmax.py"),
    ("datasets/extract_multifreq_subset.py", "pipelines/dataset/extract_subset.py"),
    ("datasets/remove_freq_from_multifreq.py", "pipelines/dataset/remove_freq.py"),
    ("datasets/subsample_multifreq_inverse_k.py", "pipelines/dataset/subsample_inverse_k.py"),
    # pipelines — normalization
    ("scripts/Normalization.py", "pipelines/normalize/multifreq.py"),
    ("scripts/calculate_normalization_stats.py", "pipelines/normalize/compute_stats.py"),
    ("scripts/normalize_dataset_with_stats.py", "pipelines/normalize/apply_stats.py"),
    ("scripts/verify_normalization.py", "pipelines/normalize/verify.py"),
    ("scripts/precompute_decap_profiles.py", "pipelines/normalize/precompute_decap.py"),
    # pipelines — analysis
    ("scripts/latent_traversal.py", "pipelines/analysis/latent_traversal.py"),
    ("scripts/quick_traversal_test.py", "pipelines/analysis/quick_traversal.py"),
    ("scripts/compute_latent_stats.py", "pipelines/analysis/latent_stats.py"),
    ("scripts/dis_con.py", "pipelines/analysis/dis_con.py"),
    ("scripts/check_mask_shape.py", "pipelines/analysis/check_mask.py"),
    # pipelines — visualize
    ("scripts/Heatmap_visual.py", "pipelines/visualize/heatmap.py"),
    ("scripts/impedance_visuals.py", "pipelines/visualize/impedance.py"),
    # libs — peb
    ("New_heatmaps/peb_frequency.py", "libs/peb/frequency.py"),
    # pipelines — heatmaps / PEB
    ("New_heatmaps/change_frequency.py", "pipelines/heatmaps/change_frequency.py"),
    ("New_heatmaps/regenerate_mhz_pebs.py", "pipelines/heatmaps/regenerate_mhz_pebs.py"),
    ("New_heatmaps/filter_combinations_from_manifest.py", "pipelines/heatmaps/filter_combinations.py"),
    # pipelines — latent optimization
    ("Latent_opm/_optimization_loader.py", "pipelines/latent/optimization_loader.py"),
    ("Latent_opm/latent_scrap_pipeline.py", "pipelines/latent/scrap_pipeline.py"),
    ("Latent_opm/generate_run_report.py", "pipelines/latent/generate_run_report.py"),
    ("Latent_opm/latent_optimization_impedance.py", "pipelines/latent/optimize.py"),
    ("Latent_opm/find_feasible_configs.py", "pipelines/latent/find_feasible.py"),
    ("Latent_opm/build_latent_opt_report.py", "pipelines/latent/build_report.py"),
    ("Latent_opm/latent_run_export_peb.py", "pipelines/latent/export_peb.py"),
    ("Latent_opm/latent_run_compare_report.py", "pipelines/latent/compare_report.py"),
    ("Latent_opm/plot_latent_opt_results.py", "pipelines/latent/plot_results.py"),
    # pipelines — active learning entry
    ("active_learning_pi/run_pipeline.py", "pipelines/active_learning/run.py"),
    # scrap — orchestration layer
    ("scrap/run_multifreq_sweep_pipeline.py", "scrap/orchestration/run_multifreq_sweep_pipeline.py"),
    ("scrap/multifreq_move_and_compare.py", "scrap/orchestration/multifreq_move_and_compare.py"),
    ("scrap/move_pi_to_real.py", "scrap/orchestration/move_pi_to_real.py"),
    ("scrap/build_comparison_report.py", "scrap/orchestration/build_comparison_report.py"),
    ("scrap/_compare_multifreq_datasets.py", "scrap/orchestration/compare_multifreq_datasets.py"),
]

IMPORT_REPLACEMENTS = [
    (r"\bfrom heatmap import\b", "from libs.data_creation.heatmap import"),
    (r"\bfrom impedance import\b", "from libs.data_creation.impedance import"),
    (r"\bfrom occupancy import\b", "from libs.data_creation.occupancy import"),
    (r"\bfrom csv_to_occupancy import\b", "from libs.data_creation.csv_to_occupancy import"),
    (r"\bfrom New_heatmaps\.peb_frequency import\b", "from libs.peb.frequency import"),
    (r"\bfrom Latent_opm\._optimization_loader import\b", "from pipelines.latent.optimization_loader import"),
    (r"\bfrom Latent_opm\.latent_scrap_pipeline import\b", "from pipelines.latent.scrap_pipeline import"),
    (r"\bfrom Latent_opm\.generate_run_report import\b", "from pipelines.latent.generate_run_report import"),
    (r"\bfrom Latent_opm\.latent_optimization_impedance import\b", "from pipelines.latent.optimize import"),
    (r"\bimport compare_generated_vs_real as compare\b", "from scrap.comparison import compare_generated_vs_real as compare"),
    (r"\bimport compare_generated_vs_real_occupancy as occ\b", "from scrap.comparison import compare_generated_vs_real_occupancy as occ"),
    (r"\bfrom batch_over_k import\b", "from scrap.comparison.batch_over_k import"),
]

SHIM_PKG_MAP = {
    "libs/data_creation/heatmap.py": "libs.data_creation.heatmap",
    "libs/data_creation/impedance.py": "libs.data_creation.impedance",
    "libs/data_creation/occupancy.py": "libs.data_creation.occupancy",
    "libs/data_creation/csv_to_occupancy.py": "libs.data_creation.csv_to_occupancy",
    "libs/peb/frequency.py": "libs.peb.frequency",
}


def _module_path(new_rel: str) -> str:
    return new_rel.replace("/", ".").removesuffix(".py")


def _shim_content(old_rel: str, new_rel: str) -> str:
    mod = _module_path(new_rel)
    old_name = Path(old_rel).name
    run_line = f'python {old_rel}'
    if new_rel in SHIM_PKG_MAP:
        pkg = SHIM_PKG_MAP[new_rel]
        return textwrap.dedent(f'''\
            """Compatibility shim — moved to ``{new_rel}``.

            Run: {run_line}
            """
            from {pkg} import *  # noqa: F403
        ''')
    return textwrap.dedent(f'''\
        """Compatibility shim — moved to ``{new_rel}``.

        Run: {run_line}
        """
        from __future__ import annotations

        import runpy
        import sys
        from pathlib import Path

        _ROOT = Path(__file__).resolve().parents[{"1" if old_rel.count("/") == 0 else "2" if old_rel.startswith("scrap/") and "/" in old_rel[6:] else "1"}]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        if __name__ == "__main__":
            import importlib.util
            _mod = importlib.import_module("{mod}")
            if hasattr(_mod, "main"):
                raise SystemExit(_mod.main())
            raise SystemExit(runpy.run_module("{mod}", run_name="__main__") or 0)
        else:
            from {mod} import *  # noqa: F403
    ''')


def _fix_shim_root_depth(old_rel: str, new_rel: str) -> str:
    content = _shim_content(old_rel, new_rel)
    # compute depth from old path to repo root
    depth = len(Path(old_rel).parts) - 1
    root_expr = ".join(['..'] * " + str(depth) + ")" if depth else "."
    if "parents[" in content:
        content = re.sub(
            r'_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[\d+\]',
            f"_ROOT = Path(__file__).resolve().parent{' / Path(' + root_expr + ')' if depth else ''}",
            content,
        )
    # simpler: always use parents[N] where N = depth
    return textwrap.dedent(f'''\
        """Compatibility shim — moved to ``{new_rel}``.

        Run: python {old_rel}
        """
        from __future__ import annotations

        import runpy
        import sys
        from pathlib import Path

        _ROOT = Path(__file__).resolve().parents[{depth}]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        _MOD = "{_module_path(new_rel)}"

        if __name__ == "__main__":
            import importlib
            mod = importlib.import_module(_MOD)
            if hasattr(mod, "main"):
                raise SystemExit(mod.main())
            raise SystemExit(runpy.run_module(_MOD, run_name="__main__") or 0)
        else:
            exec(f"from {{_MOD}} import *")  # noqa: S102
    ''')


def make_shim(old_rel: str, new_rel: str) -> str:
    depth = len(Path(old_rel).parts) - 1
    mod = _module_path(new_rel)
    if new_rel in SHIM_PKG_MAP:
        pkg = SHIM_PKG_MAP[new_rel]
        return textwrap.dedent(f'''\
            """Compatibility shim — moved to ``{new_rel}``.

            Run: python {old_rel}
            """
            from {pkg} import *  # noqa: F403
        ''')
    return textwrap.dedent(f'''\
        """Compatibility shim — moved to ``{new_rel}``.

        Run: python {old_rel}
        """
        from __future__ import annotations

        import importlib
        import runpy
        import sys
        from pathlib import Path

        _ROOT = Path(__file__).resolve().parents[{depth}]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        _MOD = "{mod}"

        if __name__ == "__main__":
            _m = importlib.import_module(_MOD)
            if hasattr(_m, "main"):
                raise SystemExit(_m.main())
            raise SystemExit(runpy.run_module(_MOD, run_name="__main__") or 0)
        else:
            _m = importlib.import_module(_MOD)
            for _n in getattr(_m, "__all__", dir(_m)):
                if not _n.startswith("_"):
                    globals()[_n] = getattr(_m, _n)
    ''')


def patch_imports(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    orig = text
    for pattern, repl in IMPORT_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    # SCRIPT_DIR.parent configs → repo_path where obvious
    if "libs/data_creation" in str(path) or "pipelines/data" in str(path):
        text = text.replace(
            'SCRIPT_DIR = Path(__file__).parent.absolute()',
            'from repo_paths import repo_path\nSCRIPT_DIR = Path(__file__).resolve().parent',
        )
    if text != orig:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    # ensure package __init__.py files
    pkgs = [
        "libs", "libs/data_creation", "libs/peb",
        "pipelines", "pipelines/data", "pipelines/dataset", "pipelines/normalize",
        "pipelines/analysis", "pipelines/visualize", "pipelines/latent",
        "pipelines/heatmaps", "pipelines/active_learning",
        "scrap/orchestration",
    ]
    for p in pkgs:
        d = ROOT / p
        d.mkdir(parents=True, exist_ok=True)
        init = d / "__init__.py"
        if not init.exists():
            init.write_text(f'"""Package: {p.replace("/", ".")}"""\n', encoding="utf-8")

    for old_rel, new_rel in MOVES:
        old = ROOT / old_rel
        new = ROOT / new_rel
        if not old.is_file():
            print(f"SKIP missing: {old_rel}")
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            print(f"SKIP exists: {new_rel}")
        else:
            shutil.move(str(old), str(new))
            print(f"MOVE {old_rel} → {new_rel}")
        patch_imports(new)
        old.write_text(make_shim(old_rel, new_rel), encoding="utf-8")
        print(f"SHIM {old_rel}")

    print("\nDone. Update docs and run: python3 -m py_compile pipelines/data/processing_single.py")


if __name__ == "__main__":
    main()
