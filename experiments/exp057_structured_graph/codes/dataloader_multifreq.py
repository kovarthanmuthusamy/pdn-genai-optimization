"""exp055 multifreq dataloader — cross-freq pairs + optional K/freq balance."""

from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset

import experiments.exp057_structured_graph.codes.dataloader_base as dataloader_base
from experiments.exp057_structured_graph.codes.dataloader_base import (
    DataLoader,
    Optional,
    Path,
    RandomSampler,
    Subset,
    VAEDataset,
    WeightedRandomSampler,
    _combined_train_weights,
    _split_indices_by_design,
    build_multifreq_pair_lookup,
    make_multifreq_collate_with_pairs,
    multifreq_collate_fn,
    precompute_multifreq_metadata,
    validate_multifreq_dataset,
)


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
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    al_overlay_data_dir: str | None = None,
    al_overlay_sample_weight: float = 25.0,
) -> tuple[DataLoader, DataLoader]:
    validate_multifreq_dataset(Path(data_dir))
    dataset = VAEDataset(
        data_dir=data_dir,
        precompute_k=(stratify_by_k or balance_k),
        cache_in_ram=cache_in_ram,
        require_layout_store=True,
    )
    meta = precompute_multifreq_metadata(dataset)
    design_keys, freq_bin = meta["design_keys"], meta["freq_bin"]
    pair_lut = build_multifreq_pair_lookup(design_keys, freq_bin) if cross_freq_pairs else {}

    if split_by_design:
        train_idx, val_idx = _split_indices_by_design(design_keys, train_split, seed)
        train_ds = (
            dataloader_base._IndexedSubset(dataset, train_idx, pair_lut=pair_lut or None)
            if cross_freq_pairs
            else Subset(dataset, train_idx)
        )
        val_ds = Subset(dataset, val_idx)
        n_layouts = len(set(design_keys))
    else:
        n = len(dataset)
        n_train = int(n * train_split)
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n - n_train], generator=gen,
        )
        train_idx, val_idx = list(train_ds.indices), list(val_ds.indices)
        n_layouts = None

    overlay_note = ""
    overlay_ds = None
    if al_overlay_data_dir:
        overlay_path = Path(al_overlay_data_dir)
        if not overlay_path.is_dir():
            raise FileNotFoundError(f"AL overlay dataset not found: {overlay_path}")
        overlay_ds = VAEDataset(
            data_dir=str(overlay_path),
            precompute_k=False,
            cache_in_ram=False,
            require_layout_store=True,
        )
        train_ds = ConcatDataset([train_ds, overlay_ds])
        overlay_note = f" al_overlay={len(overlay_ds)}"

    n_train, n_val = len(train_ds), len(val_ds)
    dl_kw = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_kw["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kw["prefetch_factor"] = prefetch_factor

    n_draws = (
        min(int(train_samples_per_epoch), n_train)
        if train_samples_per_epoch and train_samples_per_epoch > 0
        else n_train
    )
    if ddp_world_size > 1:
        n_draws = max(1, n_draws // ddp_world_size)

    sampler = None
    balance_note = ""
    if balance_k or balance_freq:
        if dataset.k_values is None:
            raise RuntimeError("k_values required for balanced sampling")
        weights = _combined_train_weights(
            train_idx,
            design_keys,
            freq_bin,
            dataset.k_values,
            balance_k=balance_k,
            balance_freq=balance_freq,
            k_power=k_balance_power,
            freq_power=freq_balance_power,
            smooth=k_balance_smoothing,
        )
        if overlay_ds is not None:
            weights = list(weights) + [float(al_overlay_sample_weight)] * len(overlay_ds)
            balance_note = (
                " balanced="
                + "+".join(p for p, on in (("freq", balance_freq), ("K", balance_k)) if on)
                + f" +al_overlay(w={al_overlay_sample_weight})"
            )
        else:
            balance_note = " balanced=" + "+".join(
                p for p, on in (("freq", balance_freq), ("K", balance_k)) if on
            )
        if ddp_world_size > 1:
            torch.manual_seed(seed + ddp_rank)
        sampler = WeightedRandomSampler(weights, num_samples=n_draws, replacement=True)
    elif overlay_ds is not None:
        base_len = n_train - len(overlay_ds)
        weights = [1.0] * base_len + [float(al_overlay_sample_weight)] * len(overlay_ds)
        if ddp_world_size > 1:
            torch.manual_seed(seed + ddp_rank)
        sampler = WeightedRandomSampler(weights, num_samples=n_draws, replacement=True)
        balance_note = f" al_overlay(w={al_overlay_sample_weight})"
    elif n_draws < n_train:
        gen = torch.Generator().manual_seed(seed + ddp_rank)
        sampler = RandomSampler(train_ds, num_samples=n_draws, replacement=True, generator=gen)

    train_collate = (
        make_multifreq_collate_with_pairs(pair_lut, dataset)
        if cross_freq_pairs and pair_lut and overlay_ds is None
        else multifreq_collate_fn
    )
    cf_note = f" cross_freq={len(pair_lut)}" if cross_freq_pairs and pair_lut and overlay_ds is None else ""
    draw_note = f" draws={n_draws}/{n_train}" if n_draws < n_train else f" draws={n_draws}"
    if ddp_world_size > 1:
        draw_note += f" rank={ddp_rank}/{ddp_world_size}"
    layout_note = f" layouts={n_layouts}" if n_layouts is not None else ""
    print(
        f"Data: train={n_train} val={n_val}{layout_note}{draw_note}{cf_note}{balance_note}{overlay_note}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=drop_last_train,
        collate_fn=train_collate,
        **dl_kw,
    )
    val_bs = int(val_batch_size) if val_batch_size and val_batch_size > 0 else batch_size
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        collate_fn=multifreq_collate_fn,
        **dl_kw,
    )
    return train_loader, val_loader
