"""
Change the PI-Distribution frequency in a .peb file and save as a new file.
"""

# ── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_PEB  = "combined_all.peb"

 # e.g. "10e6""63e6",130e6, 200e6",270e6 ,400e6,"500e6"
set_freq = "590"
NEW_FREQ   = f"{set_freq}e6"

OUTPUT_PEB = f"peb_with_19k/combined_all_{set_freq}MHz.peb"         
# ───────────────────────────────────────────────────────────────────────────

import re, os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path  = os.path.join(script_dir, INPUT_PEB)
output_path = os.path.join(script_dir, OUTPUT_PEB)

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace every occurrence of the frequency value inside EditPIDistribution
pattern     = r'(<EditPIDistribution Frequency=")[^"]*(")'
replacement = rf'\g<1>{NEW_FREQ}\2'

new_content, count = re.subn(pattern, replacement, content)

print(f"Replaced {count} frequency occurrences  ->  {NEW_FREQ}")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Saved: {output_path}")
