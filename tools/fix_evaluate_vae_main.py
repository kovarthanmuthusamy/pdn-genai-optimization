#!/usr/bin/env python3
from pathlib import Path
import re

template = Path("experiments/exp043/codes/evaluate_vae.py").read_text(encoding="utf-8")
m = re.search(r'def main\(\):.*?if __name__ == "__main__":', template, re.S)
main_block = m.group(0)

for p in Path("experiments").rglob("evaluate_vae.py"):
    if "exp043" in str(p):
        continue
    text = p.read_text(encoding="utf-8")
    if 'print(f"Device: {device}\\nCheckpoint:' in text:
        print("ok", p)
        continue
    text = re.sub(r'def main\(\):.*?if __name__ == "__main__":', lambda _m: main_block, text, flags=re.S)
    p.write_text(text, encoding="utf-8")
    print("fixed", p)
