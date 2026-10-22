"""Run a command under coverage restricted to thesis_code/ and list functions never executed.

Usage: python tests/thesis_code/tools/trace.py -- <command ...>
       e.g. python tests/thesis_code/tools/trace.py -- -m pytest tests/thesis_code/test_s10_search.py -q
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

PYLIB = "/tmp/claude-1000/-home-ubuntu-genai-pdn/cb846a21-3193-40dc-bb68-6e4ce92a9263/scratchpad/pylib"
REPO = Path(__file__).resolve().parents[3]
cmd = sys.argv[sys.argv.index("--") + 1:]
env = {**os.environ, "PYTHONPATH": PYLIB + os.pathsep + os.environ.get("PYTHONPATH", "")}
data = REPO / "thesis_code/runs/.coverage"
data.parent.mkdir(parents=True, exist_ok=True)
rc = data.parent / ".coveragerc"   # multiprocessing options are accepted only from a config file
rc.write_text(f"[run]\nsource = thesis_code\nconcurrency = multiprocessing\nparallel = True\ndata_file = {data}\n")
for old in data.parent.glob(".coverage*"):
    if old != rc:
        old.unlink()
subprocess.run([sys.executable, "-m", "coverage", "run", f"--rcfile={rc}", *cmd], cwd=REPO, env=env, check=False)
subprocess.run([sys.executable, "-m", "coverage", "combine", f"--rcfile={rc}"], cwd=REPO, env=env, check=False)
out = REPO / "thesis_code/runs/coverage.json"
subprocess.run([sys.executable, "-m", "coverage", "json", f"--rcfile={rc}", "-o", str(out)], cwd=REPO, env=env, check=True)
files = json.loads(out.read_text())["files"]
for name, info in sorted(files.items()):
    hit = set(info["executed_lines"])
    for n in ast.walk(ast.parse((REPO / name).read_text())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = {l for l in range(n.body[0].lineno, (n.end_lineno or n.lineno) + 1)}
            if not body & hit:
                print(f"never executed: {name}:{n.lineno} {n.name}")
