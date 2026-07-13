"""Per-epoch train sampling and layout-path curriculum for exp051."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import Sampler


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * max(0.0, min(1.0, t)))


def schedule_boost_frac(
    epoch_1based: int,
    *,
    boost_frac: float,
    natural_frac: float,
    start_epoch: int,
    peak_end_epoch: int,
    decay_end_epoch: int,
) -> float:
    """Rise to ``boost_frac`` by ``peak_end_epoch``, decay to ``natural_frac`` by ``decay_end_epoch``."""
    if epoch_1based < start_epoch:
        return natural_frac
    if epoch_1based >= decay_end_epoch:
        return natural_frac
    if epoch_1based <= peak_end_epoch:
        t = (epoch_1based - start_epoch + 1) / max(peak_end_epoch - start_epoch + 1, 1)
        return _lerp(natural_frac, boost_frac, t)
    t = (epoch_1based - peak_end_epoch) / max(decay_end_epoch - peak_end_epoch, 1)
    return _lerp(boost_frac, natural_frac, t)


def tag_alpha_for_target_frac(target_frac: float, n_tag: int, n_other: int) -> float:
    """Relative weight multiplier for tagged rows to hit ``target_frac`` draw probability."""
    if n_tag <= 0 or n_other <= 0:
        return 1.0
    p = max(1e-6, min(1.0 - 1e-6, float(target_frac)))
    return (p * n_other) / ((1.0 - p) * n_tag)


def load_append_tags_for_dataset(data_dir: Path, heatmap_files: Sequence[Path]) -> list[str]:
    """``append_tag`` per dataset index (aligned with sorted heatmap stems)."""
    manifest = data_dir / "manifest.csv"
    if not manifest.is_file():
        return [""] * len(heatmap_files)
    stem_to_tag: dict[str, str] = {}
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tag_col = "append_tag" if reader.fieldnames and "append_tag" in reader.fieldnames else None
        for row in reader:
            stem = Path(row.get("sample_name", "")).stem
            if not stem:
                continue
            stem_to_tag[stem] = (row.get(tag_col) or "") if tag_col else ""
    return [stem_to_tag.get(hf.stem, "") for hf in heatmap_files]


def build_tag_boost_weights(
    train_indices: Sequence[int],
    row_tags: Sequence[str],
    tag_value: str,
    target_frac: float,
) -> list[float]:
    n_tag = sum(1 for i in train_indices if row_tags[i] == tag_value)
    n_other = len(train_indices) - n_tag
    alpha = tag_alpha_for_target_frac(target_frac, n_tag, n_other)
    weights = [
        float(alpha if row_tags[i] == tag_value else 1.0)
        for i in train_indices
    ]
    return weights


class CurriculumWeightedSampler(Sampler[int]):
    """Weighted random sampler; call ``set_epoch`` before each training epoch."""

    def __init__(
        self,
        weight_fn: Callable[[int], Sequence[float]],
        num_samples: int,
        *,
        seed: int = 42,
    ):
        self.weight_fn = weight_fn
        self.num_samples = int(num_samples)
        self.epoch = 1
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))

    def set_epoch(self, epoch_1based: int) -> None:
        self.epoch = int(epoch_1based)

    def __iter__(self):
        weights = torch.as_tensor(list(self.weight_fn(self.epoch)), dtype=torch.double)
        if weights.numel() == 0:
            raise RuntimeError("CurriculumWeightedSampler: empty weights")
        idx = torch.multinomial(weights, self.num_samples, replacement=True, generator=self._gen)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


def apply_layout_path_curriculum(c, epoch_1based: int) -> None:
    """Update ``layout_train_prob`` and ``cross_freq_layout_mix_prob`` on config."""
    start = int(getattr(c, "layout_path_boost_start_epoch", 1))
    peak_end = int(getattr(c, "layout_path_boost_peak_epoch", 100))
    decay_end = int(getattr(c, "layout_path_boost_decay_end_epoch", 150))

    base_layout = float(getattr(c, "layout_train_prob_base", c.layout_train_prob))
    boost_layout = float(getattr(c, "layout_train_prob_boost", base_layout))
    base_mix = float(getattr(c, "cross_freq_layout_mix_prob_base", c.cross_freq_layout_mix_prob))
    boost_mix = float(getattr(c, "cross_freq_layout_mix_prob_boost", base_mix))

    # layout prob uses same schedule shape as tag boost (boost high early)
    if epoch_1based >= decay_end:
        layout_p, mix_p = base_layout, base_mix
    elif epoch_1based <= peak_end:
        t = (epoch_1based - start + 1) / max(peak_end - start + 1, 1)
        layout_p = _lerp(base_layout, boost_layout, t)
        mix_p = _lerp(base_mix, boost_mix, t)
    else:
        t = (epoch_1based - peak_end) / max(decay_end - peak_end, 1)
        layout_p = _lerp(boost_layout, base_layout, t)
        mix_p = _lerp(boost_mix, base_mix, t)

    c.layout_train_prob = layout_p
    c.cross_freq_layout_mix_prob = mix_p
