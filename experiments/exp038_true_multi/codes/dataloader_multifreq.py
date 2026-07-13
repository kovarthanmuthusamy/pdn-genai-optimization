"""Multi-frequency PI heatmap dataloader for exp038_true_multi.

- Train/val split by **design** (same occ+imp = one layout; all MHz stay together).
- Optional **frequency-balanced** + **K-balanced** training sampler.
- Each row: heatmap @ PI_freq, shared occ/imp per design.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset, WeightedRandomSampler

from src_vae.others.dataloader import VAEDataset
from src_vae.others.multifreq_layout_store import load_manifest_index, validate_multifreq_dataset
from src_vae.others.multifreq_anchors import (
    anchor_bin_index,
    load_anchors_mhz,
    nearest_anchor_mhz,
)

# Anchor MHz from dataset_meta.json (raw pool fallback); YAML is legacy only.
def _anchors_for_dataset(data_dir: Path | str) -> tuple[float, ...]:
    return load_anchors_mhz(data_dir)


# Backward compat (raw dataset_meta.json); prefer _anchors_for_dataset(data_dir).
ANCHOR_MHZ: tuple[float, ...] = load_anchors_mhz()


def _nearest_anchor_mhz(mhz: float, anchors: tuple[float, ...]) -> float:
    return nearest_anchor_mhz(mhz, anchors)


def _load_pi_freq_mhz(pifreq_dir: Path, stem: str) -> float:
    p = pifreq_dir / f"{stem}.npy"
    if p.exists():
        return float(np.load(p)) / 1e6
    return float("nan")


def precompute_multifreq_metadata(dataset: VAEDataset) -> dict:
    """Cache design keys and frequency bins; written next to dataset root."""
    anchors = _anchors_for_dataset(dataset.data_dir)
    cache_path = dataset.data_dir / "multifreq_meta.json"
    n = len(dataset)
    if cache_path.exists():
        meta = json.loads(cache_path.read_text())
        anchors_match = list(meta.get("anchor_mhz", [])) == list(anchors)
        if meta.get("n_samples") == n and len(meta.get("design_keys", [])) == n and anchors_match:
            return meta
        if not anchors_match:
            print(f"  multifreq_meta.json stale (anchors changed) — rebuilding …")

    print(f"Building multifreq metadata for {n} samples …", flush=True)
    pifreq_dir = dataset.pifreq_dir
    manifest_index = dataset.manifest_index or load_manifest_index(dataset.data_dir)
    if not manifest_index:
        raise ValueError(f"{dataset.data_dir}: manifest.csv required for multifreq metadata")

    design_keys: list[str] = []
    freq_mhz: list[float] = []
    freq_bin: list[int] = []

    for hf in dataset.heatmap_files:
        stem = hf.stem
        dk = manifest_index.get(stem)
        if not dk:
            raise ValueError(f"manifest.csv missing design_id for {stem}")
        design_keys.append(dk)
        if pifreq_dir is not None:
            mhz = _load_pi_freq_mhz(pifreq_dir, stem)
        else:
            mhz = float("nan")
        freq_mhz.append(mhz)
        freq_bin.append(anchor_bin_index(mhz, anchors) if mhz == mhz else -1)

    meta = {
        "n_samples": n,
        "design_keys": design_keys,
        "freq_mhz": freq_mhz,
        "freq_bin": freq_bin,
        "anchor_mhz": list(anchors),
    }
    cache_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"  saved {cache_path.name}  designs={len(set(design_keys))}  "
          f"freq bins={Counter(freq_bin)}", flush=True)
    return meta


def _split_indices_by_design(
    design_keys: list[str],
    train_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    by_design: dict[str, list[int]] = defaultdict(list)
    for i, dk in enumerate(design_keys):
        by_design[dk].append(i)

    rng = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for dk in sorted(by_design.keys()):
        idxs = by_design[dk]
        if len(idxs) == 1:
            train_indices.extend(idxs)
            continue
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        idxs = [idxs[j] for j in perm]
        n_k = len(idxs)
        n_train_k = int(n_k * train_split)
        n_train_k = max(1, min(n_k - 1, n_train_k))
        train_indices.extend(idxs[:n_train_k])
        val_indices.extend(idxs[n_train_k:])

    return train_indices, val_indices


def _combined_train_weights(
    train_indices: list[int],
    design_keys: list[str],
    freq_bin: list[int],
    k_values: list[int],
    *,
    balance_k: bool,
    balance_freq: bool,
    k_power: float,
    freq_power: float,
    smooth: float,
) -> list[float]:
    d_train = [design_keys[i] for i in train_indices]
    f_train = [freq_bin[i] for i in train_indices]
    k_train = [k_values[i] for i in train_indices]

    d_cnt = Counter(d_train)
    f_cnt = Counter(f_train)
    k_cnt = Counter(k_train)

    weights: list[float] = []
    for d, f, k in zip(d_train, f_train, k_train):
        w = 1.0
        if balance_freq and f >= 0:
            w *= (f_cnt[f] + smooth) ** (-freq_power)
        if balance_k:
            w *= (k_cnt[k] + smooth) ** (-k_power)
        # mild down-weight of over-represented designs (same occ/imp repeated per MHz)
        w *= (d_cnt[d] + smooth) ** (-0.25)
        weights.append(w)
    return weights


def build_multifreq_pair_lookup(
    design_keys: list[str],
    freq_bin: list[int],
) -> dict[int, list[int]]:
    """Map sample index → other indices (same design, different PI_freq bin)."""
    by_design: dict[str, list[int]] = defaultdict(list)
    for i, dk in enumerate(design_keys):
        by_design[dk].append(i)

    pair_lut: dict[int, list[int]] = {}
    for dk, idxs in by_design.items():
        for i in idxs:
            fb = freq_bin[i]
            alts = [j for j in idxs if j != i and freq_bin[j] >= 0 and freq_bin[j] != fb]
            pair_lut[i] = alts if alts else [i]
    return pair_lut


def multifreq_collate_fn(batch: list) -> dict:
    out = {
        "heatmap_norm": torch.stack([b["heatmap_norm"] for b in batch]),
        "occupancy": torch.stack([b["occupancy"] for b in batch]),
        "impedance": torch.stack([b["impedance"] for b in batch]),
        "K": torch.stack([b["K"] for b in batch]),
        "filenames": [b["filename"] for b in batch],
    }
    if "PI_freq" in batch[0]:
        out["PI_freq"] = torch.stack([b["PI_freq"] for b in batch])
    if "sample_idx" in batch[0]:
        out["sample_idx"] = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
    return out


def multifreq_collate_with_alt_fn(batch: list) -> dict:
    """Stack batches; cross-freq alt fields are prefetched in ``_IndexedSubset``."""
    out = multifreq_collate_fn(batch)
    if batch and "heatmap_norm_alt" in batch[0]:
        out["heatmap_norm_alt"] = torch.stack([b["heatmap_norm_alt"] for b in batch])
        out["PI_freq_alt"] = torch.stack([b["PI_freq_alt"] for b in batch])
    return out


def make_multifreq_collate_with_pairs(
    pair_lut: dict[int, list[int]],
    full_dataset,
) -> Callable[[list], dict]:
    """Legacy collate: alt MHz lookup in collate (slow; used only without RAM prefetch)."""
    pair_arr: dict[int, np.ndarray] = {
        i: np.asarray(alts, dtype=np.int64) for i, alts in pair_lut.items()
    }
    rng = np.random.default_rng()
    ram = getattr(full_dataset, "_ram", None)

    def collate(batch: list) -> dict:
        if batch and "heatmap_norm_alt" in batch[0]:
            return multifreq_collate_with_alt_fn(batch)
        out = multifreq_collate_fn(batch)
        if not pair_arr or "sample_idx" not in batch[0]:
            return out
        alt_rows = []
        pi_alt = []
        for b in batch:
            i = int(b["sample_idx"])
            alts = pair_arr.get(i)
            j = int(alts[rng.integers(len(alts))]) if alts is not None and len(alts) else i
            alt = ram[j] if ram is not None else full_dataset[j]
            alt_rows.append(alt["heatmap_norm"])
            pi_alt.append(alt["PI_freq"])
        out["heatmap_norm_alt"] = torch.stack(alt_rows)
        out["PI_freq_alt"] = torch.stack(pi_alt)
        return out

    return collate


class _IndexedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and exposes global dataset indices for pair lookup."""

    def __init__(self, dataset, indices: list[int], *, pair_lut: dict[int, list[int]] | None = None):
        self.dataset = dataset
        self.indices = indices
        self.pair_lut = pair_lut
        self._ram = getattr(dataset, "_ram", None)
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        row = self.dataset[idx]
        out = {**row, "sample_idx": idx}
        if self.pair_lut:
            alts = self.pair_lut.get(idx)
            j = int(alts[self._rng.integers(len(alts))]) if alts else idx
            alt = self._ram[j] if self._ram is not None else self.dataset[j]
            out["heatmap_norm_alt"] = alt["heatmap_norm"]
            out["PI_freq_alt"] = alt["PI_freq"]
        return out


def create_multifreq_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.9,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = 2,
    *,
    split_by_design: bool = True,
    balance_k: bool = True,
    balance_freq: bool = True,
    k_balance_power: float = 0.5,
    freq_balance_power: float = 1.0,
    k_balance_smoothing: float = 1e-3,
    stratify_by_k: bool = True,
    cache_in_ram: bool = False,
    drop_last_train: bool = True,
    train_samples_per_epoch: int | None = None,
    cross_freq_pairs: bool = True,
    val_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    validate_multifreq_dataset(Path(data_dir))
    dataset = VAEDataset(
        data_dir=data_dir,
        precompute_k=(stratify_by_k or balance_k),
        cache_in_ram=cache_in_ram,
        require_layout_store=True,
    )

    meta = precompute_multifreq_metadata(dataset)
    anchor_mhz = [float(x) for x in meta["anchor_mhz"]]
    design_keys = meta["design_keys"]
    freq_bin = meta["freq_bin"]

    pair_lut = build_multifreq_pair_lookup(design_keys, freq_bin) if cross_freq_pairs else {}

    if split_by_design:
        train_indices, val_indices = _split_indices_by_design(design_keys, train_split, seed)
        train_dataset = (
            _IndexedSubset(dataset, train_indices, pair_lut=pair_lut or None)
            if cross_freq_pairs
            else torch.utils.data.Subset(dataset, train_indices)
        )
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        print(f"Multifreq split by design: {len(set(design_keys))} layouts")
    else:
        n = len(dataset)
        n_train = int(n * train_split)
        gen = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n - n_train], generator=gen,
        )
        train_indices = list(train_dataset.indices)
        val_indices = list(val_dataset.indices)

    n_train, n_val = len(train_dataset), len(val_dataset)
    print(f"Train samples: {n_train}, Validation samples: {n_val}")

    # freq distribution in train
    f_ctr = Counter(freq_bin[i] for i in train_indices if freq_bin[i] >= 0)
    if f_ctr:
        print("  Train PI_freq bins (anchor MHz):")
        for bi in sorted(f_ctr.keys()):
            print(f"    {anchor_mhz[bi]:.0f} MHz: {f_ctr[bi]}")

    dl_kwargs: dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

    n_draws = n_train
    if train_samples_per_epoch is not None and train_samples_per_epoch > 0:
        n_draws = min(int(train_samples_per_epoch), n_train)

    sampler = None
    if balance_k or balance_freq:
        if dataset.k_values is None:
            raise RuntimeError("k_values required for balanced sampling")
        weights = _combined_train_weights(
            train_indices, design_keys, freq_bin, dataset.k_values,
            balance_k=balance_k, balance_freq=balance_freq,
            k_power=k_balance_power, freq_power=freq_balance_power,
            smooth=k_balance_smoothing,
        )
        sampler = WeightedRandomSampler(weights, num_samples=n_draws, replacement=True)
        parts = []
        if balance_freq:
            parts.append("freq")
        if balance_k:
            parts.append("K")
        print(f"  Balanced sampling: {' + '.join(parts)}")
    elif n_draws < n_train:
        sampler = RandomSampler(train_dataset, num_samples=n_draws, replacement=True)

    if train_samples_per_epoch is not None and n_draws < n_train:
        print(f"  Train draws/epoch: {n_draws} (random subset of {n_train} train rows)")
    elif train_samples_per_epoch is not None:
        print(f"  Train draws/epoch: {n_train} (full train pool)")

    train_collate = (
        make_multifreq_collate_with_pairs(pair_lut, dataset)
        if cross_freq_pairs and pair_lut
        else multifreq_collate_fn
    )
    if cross_freq_pairs and pair_lut:
        print(f"  Cross-freq pairs: enabled ({len(pair_lut)} train indices with alt MHz)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=drop_last_train,
        collate_fn=train_collate,
        **dl_kwargs,
    )
    val_bs = int(val_batch_size) if val_batch_size and val_batch_size > 0 else batch_size
    if val_bs > batch_size:
        print(f"  Val batch_size={val_bs} (train={batch_size})")
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        collate_fn=multifreq_collate_fn,
        **dl_kwargs,
    )
    return train_loader, val_loader
