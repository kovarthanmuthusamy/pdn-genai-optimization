"""exp049 multifreq dataloader — bias cross-freq alt pairs toward high MHz."""

from __future__ import annotations

from experiments.exp038_true_multi.codes.dataloader_multifreq import (  # noqa: F401
    ANCHOR_MHZ,
    build_multifreq_pair_lookup,
    make_multifreq_collate_with_pairs,
    multifreq_collate_fn,
    precompute_multifreq_metadata,
)
from experiments.exp038_true_multi.codes import dataloader_multifreq as _base
from experiments.exp038_true_multi.codes.dataloader_multifreq import (
    Counter,
    DataLoader,
    Optional,
    Path,
    RandomSampler,
    Subset,
    VAEDataset,
    WeightedRandomSampler,
    _combined_train_weights,
    _split_indices_by_design,
    validate_multifreq_dataset,
)
import numpy as np
import torch


class _HighFreqBiasSubset(_base._IndexedSubset):
    """Same-design alt MHz; prefer 300+ MHz targets for cross-freq loss."""

    def __init__(
        self,
        dataset,
        indices: list[int],
        *,
        pair_lut: dict[int, list[int]] | None = None,
        freq_bin: list[int] | None = None,
        high_freq_pair_bias: float = 0.55,
        high_freq_mhz_threshold: float = 250.0,
    ):
        super().__init__(dataset, indices, pair_lut=pair_lut)
        self._freq_bin = list(freq_bin or [])
        self._high_bins = {
            i for i, mhz in enumerate(ANCHOR_MHZ) if float(mhz) >= float(high_freq_mhz_threshold)
        }
        self._pair_bias = float(high_freq_pair_bias)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        row = self.dataset[idx]
        out = {**row, "sample_idx": idx}
        if not self.pair_lut:
            return out
        alts = self.pair_lut.get(idx) or [idx]
        high = [
            j for j in alts
            if j < len(self._freq_bin) and self._freq_bin[j] in self._high_bins
        ]
        if high and self._rng.random() < self._pair_bias:
            pool = high
        else:
            pool = alts
        j = int(pool[self._rng.integers(len(pool))])
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
    high_freq_pair_bias: float = 0.55,
    high_freq_mhz_threshold: float = 250.0,
) -> tuple[DataLoader, DataLoader]:
    """Same as exp038 loader but with high-MHz cross-freq pair bias on train."""
    validate_multifreq_dataset(Path(data_dir))
    dataset = VAEDataset(
        data_dir=data_dir,
        precompute_k=(stratify_by_k or balance_k),
        cache_in_ram=cache_in_ram,
        require_layout_store=True,
    )

    meta = precompute_multifreq_metadata(dataset)
    design_keys = meta["design_keys"]
    freq_bin = meta["freq_bin"]

    pair_lut = build_multifreq_pair_lookup(design_keys, freq_bin) if cross_freq_pairs else {}

    if split_by_design:
        train_indices, val_indices = _split_indices_by_design(design_keys, train_split, seed)
        train_dataset = (
            _HighFreqBiasSubset(
                dataset,
                train_indices,
                pair_lut=pair_lut or None,
                freq_bin=freq_bin,
                high_freq_pair_bias=high_freq_pair_bias,
                high_freq_mhz_threshold=high_freq_mhz_threshold,
            )
            if cross_freq_pairs
            else Subset(dataset, train_indices)
        )
        val_dataset = Subset(dataset, val_indices)
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

    f_ctr = Counter(freq_bin[i] for i in train_indices if freq_bin[i] >= 0)
    if f_ctr:
        print("  Train PI_freq bins (anchor MHz):")
        for bi in sorted(f_ctr.keys()):
            print(f"    {ANCHOR_MHZ[bi]:.0f} MHz: {f_ctr[bi]}")

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
        print(
            f"  Cross-freq pairs: enabled ({len(pair_lut)} indices); "
            f"high-MHz alt bias={high_freq_pair_bias:.2f} (≥{high_freq_mhz_threshold:.0f} MHz)",
        )

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
