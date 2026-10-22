"""Train / held-out DataLoaders of the dataset.

What: one row = one map at one frequency; the placement and spectrum are shared by every row of the
    same placement (layouts/<design_id>/).
    * Hold-out by placement: all frequencies of a placement on one side, by ``split_seed`` over the
      sorted design ids (``_split_indices_by_design``).
    * Cross-frequency partner: every row carries a random other row of the same placement in a
      different frequency bin (``heatmap_norm_alt``, ``PI_freq_alt``) for the cross-frequency and
      anti-copy losses and the in-training cross-frequency check.
    * Fixed draws per epoch: ``train_samples_per_epoch`` rows with replacement, split over the GPUs.
      In the balanced dataset every placement has 11 maps and every K the same number of
      placements, so uniform draws are even over K and placements.
    The batch occupancy is a (B, 52, T) one-hot, built by ``multifreq_collate_fn``; indexing the
    dataset directly returns the type codes.
    The per-row metadata is cached in multifreq_meta.json (read from the dataset folder if there,
    else built in OUTPUT_DIR/s04_dataset/cache).
Usage (import only):
    create_multifreq_data_loaders(data_dir, batch_size, ...) -> (train_loader, val_loader)
    _split_indices_by_design(design_keys, train_split, seed) -> (train rows, val rows)
    multifreq_collate_fn(batch, n_decap_types)
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from thesis_code.common.frequency import anchor_bin_index, load_anchors_mhz
from thesis_code.common.placement import type_ids_to_onehot
from thesis_code.s04_dataset.dataset import VAEDataset, cache_file
from thesis_code.s04_dataset.store import validate_multifreq_dataset


def _occ_to_onehot_tensor(occ: torch.Tensor, n_decap_types: int) -> torch.Tensor:
    """One sample's occupancy (type codes 0..T, or legacy 0/1) -> (52, T) one-hot."""
    if occ.dim() == 2 and occ.shape == (52, n_decap_types):
        return occ.float()
    if occ.dim() == 1 or (occ.dim() == 2 and occ.shape[-1] == 52):
        return type_ids_to_onehot(occ, n_decap_types)
    return occ.float()


def multifreq_collate_fn(batch: list, *, n_decap_types: int = 3) -> dict:
    stack = lambda key: torch.stack([b[key] for b in batch])  # noqa: E731
    return {
        "heatmap_norm": stack("heatmap_norm"),
        "occupancy": torch.stack([_occ_to_onehot_tensor(b["occupancy"], n_decap_types) for b in batch]),
        "impedance": stack("impedance"),
        "K": stack("K"),
        "PI_freq": stack("PI_freq"),
        "heatmap_norm_alt": stack("heatmap_norm_alt"),
        "PI_freq_alt": stack("PI_freq_alt"),
        "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "filenames": [b["filename"] for b in batch],
    }


def precompute_multifreq_metadata(dataset: VAEDataset) -> dict:
    """Per-row design id, frequency (MHz) and frequency-bin index (multifreq_meta.json; rebuilt when
    the row count or the frequency list changes)."""
    anchors = load_anchors_mhz(dataset.data_dir)
    read, write = cache_file(dataset.data_dir, "multifreq_meta.json", dataset.cache_dir)
    n = len(dataset)
    if read.exists():
        meta = json.loads(read.read_text())
        if (meta.get("n_samples") == n and len(meta.get("design_keys", [])) == n
                and list(meta.get("anchor_mhz", [])) == list(anchors)):
            return meta

    print(f"Building multifreq metadata for {n} samples ...", flush=True)
    design_keys, freq_mhz, freq_bin = [], [], []
    for hf in dataset.heatmap_files:
        design_keys.append(dataset._design_id_for_stem(hf.stem))
        mhz = float(np.load(dataset.pifreq_dir / f"{hf.stem}.npy")) / 1e6
        freq_mhz.append(mhz)
        freq_bin.append(anchor_bin_index(mhz, anchors))
    meta = {"n_samples": n, "design_keys": design_keys, "freq_mhz": freq_mhz,
            "freq_bin": freq_bin, "anchor_mhz": list(anchors)}
    write.parent.mkdir(parents=True, exist_ok=True)
    write.write_text(json.dumps(meta), encoding="utf-8")
    print(f"  saved {write}  designs={len(set(design_keys))}  freq bins={len(set(freq_bin))}", flush=True)
    return meta


def _split_indices_by_design(design_keys: list[str], train_split: float, seed: int) -> tuple[list[int], list[int]]:
    """Row indices of the train and val layouts. Deterministic given ``seed`` and the sorted
    design ids; every row of a layout lands on the same side."""
    by_design: dict[str, list[int]] = defaultdict(list)
    for i, dk in enumerate(design_keys):
        by_design[dk].append(i)
    designs = sorted(by_design)
    n_d = len(designs)
    if n_d == 0:
        return [], []
    perm = torch.randperm(n_d, generator=torch.Generator().manual_seed(seed)).tolist()
    n_train_d = max(1, min(n_d - 1, int(n_d * train_split))) if n_d >= 2 else n_d
    train_designs = {designs[j] for j in perm[:n_train_d]}
    train_idx, val_idx = [], []
    for dk in designs:
        (train_idx if dk in train_designs else val_idx).extend(by_design[dk])
    return train_idx, val_idx


def build_multifreq_pair_lookup(design_keys: list[str], freq_bin: list[int]) -> dict[int, list[int]]:
    """Row -> rows of the same layout in a different frequency bin (itself if there are none)."""
    by_design: dict[str, list[int]] = defaultdict(list)
    for i, dk in enumerate(design_keys):
        by_design[dk].append(i)
    pair_lut: dict[int, list[int]] = {}
    for idxs in by_design.values():
        for i in idxs:
            alts = [j for j in idxs if j != i and freq_bin[j] >= 0 and freq_bin[j] != freq_bin[i]]
            pair_lut[i] = alts or [i]
    return pair_lut


class _IndexedSubset(torch.utils.data.Dataset):
    """Subset that adds ``sample_idx`` and a random cross-frequency partner of the same layout.

    The partner generator is unseeded, so pair draws differ run to run (and forked DataLoader
    workers start from the same generator state).
    """

    def __init__(self, dataset, indices: list[int], *, pair_lut: dict[int, list[int]]):
        self.dataset = dataset
        self.indices = indices
        self.pair_lut = pair_lut
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        out = {**self.dataset[idx], "sample_idx": idx}
        alts = self.pair_lut[idx]
        alt = self.dataset[int(alts[self._rng.integers(len(alts))])]
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
    drop_last_train: bool = True,
    train_samples_per_epoch: int | None = None,
    val_batch_size: int | None = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    n_decap_types: int = 3,
) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader) with the layout hold-out given by ``seed``. Every row carries a
    cross-frequency partner (val too: the cross-frequency eval scores against its real map)."""
    validate_multifreq_dataset(Path(data_dir))
    dataset = VAEDataset(data_dir=data_dir, precompute_k=True, cache_in_ram=False)
    meta = precompute_multifreq_metadata(dataset)
    design_keys, freq_bin = meta["design_keys"], meta["freq_bin"]
    pair_lut = build_multifreq_pair_lookup(design_keys, freq_bin)

    train_idx, val_idx = _split_indices_by_design(design_keys, train_split, seed)
    train_ds = _IndexedSubset(dataset, train_idx, pair_lut=pair_lut)
    val_ds = _IndexedSubset(dataset, val_idx, pair_lut=pair_lut)
    n_layouts = len(set(design_keys))
    print(f"Multifreq LAYOUT holdout (seed={seed}): {n_layouts} designs -> "
          f"train={len({design_keys[i] for i in train_idx})} val={len({design_keys[i] for i in val_idx})} "
          f"(all MHz of a design stay together; no layout overlap)", flush=True)

    n_train, n_val = len(train_ds), len(val_ds)
    dl_kw: dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_kw["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kw["prefetch_factor"] = prefetch_factor

    n_draws = (min(int(train_samples_per_epoch), n_train)
               if train_samples_per_epoch and train_samples_per_epoch > 0 else n_train)
    if ddp_world_size > 1:
        n_draws = max(1, n_draws // ddp_world_size)
    sampler = None
    if n_draws < n_train:
        gen = torch.Generator().manual_seed(seed + ddp_rank)
        sampler = RandomSampler(train_ds, num_samples=n_draws, replacement=True, generator=gen)

    draw_note = f" draws={n_draws}/{n_train}" if n_draws < n_train else f" draws={n_draws}"
    if ddp_world_size > 1:
        draw_note += f" rank={ddp_rank}/{ddp_world_size}"
    print(f"Data: train={n_train} val={n_val} layouts={n_layouts}{draw_note} cross_freq={len(pair_lut)}", flush=True)

    collate = partial(multifreq_collate_fn, n_decap_types=max(1, int(n_decap_types)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                              drop_last=drop_last_train, collate_fn=collate, **dl_kw)
    val_bs = int(val_batch_size) if val_batch_size and val_batch_size > 0 else batch_size
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, collate_fn=collate, **dl_kw)
    return train_loader, val_loader
