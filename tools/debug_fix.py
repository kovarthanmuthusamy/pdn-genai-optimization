from pathlib import Path
import re

template = Path("experiments/exp043/codes/evaluate_vae.py").read_text(encoding="utf-8")
m = re.search(r'def main\(\):.*?if __name__ == "__main__":', template, re.S)
main_block = m.group(0)
Path("/tmp/main_block.py").write_text(main_block, encoding="utf-8")
print("len", len(main_block))
line = [l for l in main_block.splitlines() if "Device" in l][0]
print("main_block line repr:", repr(line))

p = Path("experiments/exp040/codes/evaluate_vae.py")
text = p.read_text(encoding="utf-8")
text2, n = re.subn(r'def main\(\):.*?if __name__ == "__main__":', main_block, text, flags=re.S)
print("replacements:", n)
Path("/tmp/exp040_fixed.py").write_text(text2, encoding="utf-8")
line = [l for l in text2.splitlines() if "Device" in l and "print" in l][0]
print("fixed line:", line)
