# Folder Rename & Centralized Paths

## Summary of Changes

- Renamed project folder from `/home/ubuntu/gan` → `/home/ubuntu/genai_pdn`
- Added **`repo_paths.py`** as the single source of truth for repository root and canonical data paths
- Kept **`gan_paths.py`** as a backward-compatible re-export shim (imports from `repo_paths`)
- Updated **50+ Python scripts** (scrap, pipelines, scratch, active_learning) to import from `repo_paths` instead of hard-coding `parents[2]` or absolute paths
- Renamed active-learning config key **`repo_root`** (with `gan_root` kept as deprecated alias)
- Updated documentation references to the new folder name

## Implementation Details

### Central path module: `repo_paths.py`

All scripts should use:

```python
from repo_paths import REPO_ROOT, setup_path, repo_path
setup_path()  # adds repo root to sys.path
```

Key exports:

| Symbol | Purpose |
|--------|---------|
| `REPO_ROOT` | Auto-detected from `repo_paths.py` location |
| `PROJECT_ROOT` | Alias for `REPO_ROOT` |
| `setup_path()` | Inserts repo root on `sys.path` |
| `repo_path(*parts)` | Build paths under repo root |
| `DATASETS_DIR`, `EXPERIMENTS_DIR`, etc. | Canonical directory constants |

Optional override via environment variable `GENAI_PDN_ROOT` or `REPO_ROOT`.

### Migration tools (one-time helpers)

- `tools/migrate_repo_paths.py` — bulk-replaced `_PROJECT_ROOT = parents[2]` patterns
- `tools/fix_scratch_paths.py` — normalized scratch scripts
- `tools/fix_docstring_paths.py` — repaired docstrings after migration

### Active learning

- `active_learning_pi/al/paths.py` now delegates to `repo_paths.REPO_ROOT`
- Config files use `repo_root: null` (auto-detect); `gan_root` still works as alias

### Folder rename

```bash
mv /home/ubuntu/gan /home/ubuntu/genai_pdn
```

Because `REPO_ROOT` is derived from `__file__`, no code changes are needed when the folder is renamed again.

## Verification & Execution Results

```
REPO_ROOT: /home/ubuntu/genai_pdn
exists datasets: True
name: genai_pdn
```

- `python3 -m py_compile repo_paths.py gan_paths.py pipelines/_bootstrap.py ...` — **passed**
- `load_config()` auto-detects `repo_root: /home/ubuntu/genai_pdn` — **passed**

### Cursor workspace note

Re-open the project from `/home/ubuntu/genai_pdn` (or update the Cursor workspace path) so the IDE points at the renamed folder.
