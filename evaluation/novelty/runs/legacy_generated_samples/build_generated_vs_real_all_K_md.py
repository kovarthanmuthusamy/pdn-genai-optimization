"""Generate a single Markdown file embedding per-K comparison plots.

Expected folder layout (relative to this script's directory):
    K1/
      generated_vs_real_heatmap.png
      generated_vs_real_impedance_profile.png
      generated_occupancy.png
    K2/
      ...

It writes `generated_vs_real_all_K.md` next to this script.

Run:
    python scrap/generated_samples/build_generated_vs_real_all_K_md.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageSpec:
    title: str
    filename: str
    alt_prefix: str


IMAGES: list[ImageSpec] = [
    ImageSpec(title="Heatmap", filename="generated_vs_real_heatmap.png", alt_prefix="heatmap"),
    ImageSpec(title="Impedance profile", filename="generated_vs_real_impedance_profile.png", alt_prefix="impedance profile"),
    ImageSpec(title="Occupancy", filename="generated_occupancy.png", alt_prefix="occupancy"),
]


def _parse_k_dir_name(name: str) -> int | None:
    if not name.startswith("K"):
        return None
    suffix = name[1:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _find_k_dirs(base_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        k = _parse_k_dir_name(p.name)
        if k is None:
            continue
        out.append((k, p))
    out.sort(key=lambda x: x[0])
    return out


def _relpath(from_dir: Path, to_path: Path) -> str:
    return to_path.relative_to(from_dir).as_posix()


def build_markdown(base_dir: Path) -> str:
    k_dirs = _find_k_dirs(base_dir)
    if not k_dirs:
        raise SystemExit(f"No K* directories found under: {base_dir}")

    lines: list[str] = []
    lines.append("# Generated vs Real — All K")
    lines.append("")
    lines.append("This file aggregates the per-K comparison plots (heatmap, impedance, occupancy).")
    lines.append("")

    for k, k_dir in k_dirs:
        lines.append(f"## K={k}")
        lines.append("")

        for spec in IMAGES:
            lines.append(f"**{spec.title}**")
            lines.append("")
            img_path = k_dir / spec.filename
            if img_path.exists():
                rel = _relpath(base_dir, img_path)
                lines.append(f"![K={k} {spec.alt_prefix}]({rel})")
            else:
                rel = _relpath(base_dir, img_path)
                lines.append(f"*(missing: {rel})*")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "generated_vs_real_all_K.md"
    tmp_path = base_dir / (out_path.name + ".new")

    md = build_markdown(base_dir)
    tmp_path.write_text(md, encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
