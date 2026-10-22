"""List module-level functions/classes in thesis_code/ that no file in thesis_code/ references.

Usage: python tests/thesis_code/tools/unreferenced.py     (prints file:line name; exit 1 if any)
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "thesis_code"
KEEP = {"main", "__init__"}
defs, used = [], set()
for f in sorted(ROOT.rglob("*.py")):
    tree = ast.parse(f.read_text())
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append((f, n.lineno, n.name))
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            used.add(n.attr)
        elif isinstance(n, ast.alias):
            used.add(n.name.split(".")[-1])
bad = [(f, l, n) for f, l, n in defs if n not in used and n not in KEEP]
for f, l, n in bad:
    print(f"{f.relative_to(ROOT.parent)}:{l} {n}")
sys.exit(1 if bad else 0)
