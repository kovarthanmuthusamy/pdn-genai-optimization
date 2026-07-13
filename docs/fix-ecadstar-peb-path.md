# Fix: ECADStar PEB path mangling on WSL

## Summary of Changes

- Fixed `resolve_windows_path()` in `pipelines/dataset_sim/ecadstar.py` to map Windows paths to `/mnt/<drive>/...` whenever the drive is mounted, even if the target directory does not exist yet.
- Fixed `windows_path_str()` to avoid calling `Path.resolve()` on Windows-style paths (which produced `/home/ubuntu/genai_pdn/C:\Users\...` on Linux).
- Added recovery for already-mangled paths that contain a `C:\...` path component.
- Applied the same `resolve_windows_path()` fix to `scrap/peb_copy.py`.
- Moved the misplaced `combinations_dist_170MHz.peb` (~108 MB) from the accidental repo subdirectory to `C:\Users\muthusamy\Desktop\Raw\peb\`.

## Implementation Details

### Root cause

`PEB_DIR_WIN` was `C:\Users\muthusamy\Desktop\Raw\peb`, but `Raw\peb` did not exist on the Windows side when the pipeline started. The old helper only used `/mnt/c/...` when the **full** path already existed; otherwise it returned `Path(r"C:\Users\...")`, which Python on Linux treats as a **relative** path under the repo cwd.

That caused:

1. PEB files written to `/home/ubuntu/genai_pdn/C:\Users\muthusamy\Desktop\Raw\peb\` instead of the Desktop.
2. `windows_path_str()` passing `/home/ubuntu/genai_pdn/C:\Users\...` to PowerShell, so ECADStar reported **PEB file not found**.

### Fix

- `resolve_windows_path`: check only that `/mnt/<drive>` is mounted, then always build the WSL path.
- `windows_path_str`: detect Windows absolute paths before `resolve()`; convert `/mnt/c/...` to `C:\...`; recover mangled mixed paths.

## Verification & Execution Results

```text
resolve C:\Users\muthusamy\Desktop\Raw\peb -> /mnt/c/Users/muthusamy/Desktop/Raw/peb
windows_path_str: C:\Users\muthusamy\Desktop\Raw\peb\combinations_dist_170MHz.peb
powershell Test-Path: True
```

Re-run:

```bash
python pipelines/dataset_sim/run_combinations_sim_pipeline.py
```

The 170 MHz PEB is already on the Desktop; the pipeline will regenerate it if needed, then ECADStar should receive a valid Windows path.

### Optional cleanup

You may remove the accidental directory under the repo (only if empty):

`~/genai_pdn/C:\Users\muthusamy\Desktop\Raw\`
