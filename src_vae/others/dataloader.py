"""Data loader for Multi-Input VAE training.

Layout-centric multifreq (``data_multifreq*``):
  - heatmap/sample_N.npy, PI_freq/sample_N.npy per row
  - layouts/{design_id}/imp.npy, occ.npy once per layout (via manifest.csv)

Legacy single-freq (``data_norm`` without layouts/):
  - heatmap/, Imp/, Occ_map/ per sample
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from src_vae.others.heatmap_z_clip import clip_heatmap_z, load_heatmap_z_clip_bounds
from src_vae.others.multifreq_layout_store import (
    has_layout_store,
    layout_imp_path,
    layout_occ_path,
    load_manifest_index,
    validate_multifreq_dataset,
)


class VAEDataset(Dataset):
    """PyTorch Dataset: heatmap + occupancy + impedance (+ optional PI_freq)."""

    def __init__(
        self,
        data_dir: str = "datasets/data_norm",
        normalize: bool = False,
        stats_path: Optional[str] = None,
        precompute_k: bool = False,
        cache_in_ram: bool = False,
        *,
        require_layout_store: bool | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.stats = None

        self.heatmap_dir = self.data_dir / "heatmap"
        self.pifreq_dir = self.data_dir / "PI_freq"
        self.manifest_index = load_manifest_index(self.data_dir)
        self.use_layout_store = has_layout_store(self.data_dir)

        # Legacy paths (single-freq only)
        self.impedance_dir = self.data_dir / "Imp"
        self.occupancy_dir = self.data_dir / "Occ_map"

        if not self.heatmap_dir.exists():
            raise ValueError(f"Directory not found: {self.heatmap_dir}")

        self.has_pifreq = self.pifreq_dir.is_dir()
        if self.has_pifreq:
            if require_layout_store is False:
                pass
            else:
                validate_multifreq_dataset(self.data_dir)
                self.use_layout_store = True
            print("PI_freq found — multifreq mode (layout store for imp/occ).")
        elif require_layout_store:
            raise ValueError(f"{data_dir}: expected multifreq dataset with PI_freq/ and layouts/")

        if self.use_layout_store:
            n_layouts = sum(1 for p in (self.data_dir / "layouts").iterdir() if p.is_dir())
            print(
                f"Layout store: {n_layouts} layouts, "
                f"{len(self.manifest_index)} manifest rows",
            )
        else:
            for dir_path in (self.impedance_dir, self.occupancy_dir):
                if not dir_path.exists():
                    raise ValueError(f"Directory not found: {dir_path}")

        self.heatmap_files = sorted(self.heatmap_dir.glob("*.npy"))
        if not self.heatmap_files:
            raise ValueError(f"No .npy files found in {self.heatmap_dir}")

        self.n_samples = len(self.heatmap_files)
        print(f"Loaded {self.n_samples} samples from {data_dir}")

        self.heatmap_z_clip: tuple[float, float] | None = load_heatmap_z_clip_bounds(self.data_dir)
        if self.heatmap_z_clip is not None:
            lo, hi = self.heatmap_z_clip
            print(f"  Heatmap z-clip at load: [{lo:.4f}, {hi:.4f}] (tail trim from stats)")

        self.k_values: Optional[list[int]] = None
        if precompute_k:
            self._precompute_k_values()

        if self.normalize:
            if stats_path is None:
                stats_path = str(self.data_dir / "normalization_stats.json")
            if Path(stats_path).exists():
                with open(stats_path, "r") as f:
                    self.stats = json.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: stats file not found at {stats_path}, skipping normalization")
                self.normalize = False

        self._layout_cache: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        self._ram: list[dict] | None = None
        if cache_in_ram:
            self._load_ram_cache()

    def _precompute_k_values(self) -> None:
        cache_path = self.data_dir / "k_values_cache.npy"
        if cache_path.exists():
            cached = np.load(str(cache_path))
            if len(cached) == self.n_samples:
                self.k_values = cached.tolist()
                print(f"K-values loaded from cache ({self.n_samples} samples)")
                return
            print(f"K-cache size mismatch ({len(cached)} vs {self.n_samples}), recomputing…")

        print(f"Computing K values for {self.n_samples} samples (one-time, will be cached)…")
        k_values: list[int] = []
        design_k: dict[str, int] = {}
        for hf in self.heatmap_files:
            stem = hf.stem
            if self.use_layout_store:
                did = self._design_id_for_stem(stem)
                if did not in design_k:
                    occ = np.load(layout_occ_path(self.data_dir, did), mmap_mode="r").reshape(-1)
                    design_k[did] = int(np.clip(occ, 0.0, 1.0).sum())
                k_values.append(design_k[did])
            else:
                occ_path = self.occupancy_dir / f"{stem}.npy"
                if not occ_path.exists():
                    raise ValueError(f"Missing occupancy file: {occ_path}")
                occ = np.load(occ_path, mmap_mode="r").reshape(-1)
                k_values.append(int(np.clip(occ, 0.0, 1.0).sum()))
        self.k_values = k_values
        np.save(str(cache_path), np.array(k_values, dtype=np.int16))
        print(f"K-values cached to {cache_path}")

    def _design_id_for_stem(self, stem: str) -> str:
        did = self.manifest_index.get(stem)
        if not did:
            raise ValueError(f"No design_id in manifest.csv for {stem}")
        return did

    def _load_ram_cache(self) -> None:
        t0 = time.perf_counter()
        print(f"Caching {self.n_samples} samples in RAM …", flush=True)
        self._ram = [self._load_sample(i) for i in range(self.n_samples)]
        print(f"  RAM cache ready in {time.perf_counter() - t0:.1f}s", flush=True)

    def _load_layout_modality(self, design_id: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        if design_id in self._layout_cache:
            return self._layout_cache[design_id]
        imp_path = layout_imp_path(self.data_dir, design_id)
        occ_path = layout_occ_path(self.data_dir, design_id)
        if not imp_path.is_file() or not occ_path.is_file():
            raise FileNotFoundError(
                f"Missing layout files for {design_id}: {imp_path} / {occ_path}",
            )
        impedance_raw = np.load(imp_path)
        occupancy_np = np.load(occ_path).reshape(-1)
        if impedance_raw.ndim == 2:
            impedance_t = torch.from_numpy(impedance_raw[:1]).float()
        else:
            impedance_t = torch.from_numpy(impedance_raw.flatten()).float().unsqueeze(0)
        occupancy_t = torch.from_numpy(occupancy_np).float().clamp(0.0, 1.0)
        k_int = int(occupancy_t.sum().item())
        triple = (impedance_t, occupancy_t, k_int)
        self._layout_cache[design_id] = triple
        return triple

    def _load_sample(self, idx: int) -> dict:
        filename = self._get_filename(idx)
        heatmap_norm = np.load(self.heatmap_dir / f"{filename}.npy")
        if self.heatmap_z_clip is not None:
            lo, hi = self.heatmap_z_clip
            heatmap_norm = np.clip(heatmap_norm, lo, hi)
        heatmap_norm_t = torch.from_numpy(heatmap_norm).float()

        if self.use_layout_store:
            design_id = self._design_id_for_stem(filename)
            impedance_t, occupancy_t, k_from_layout = self._load_layout_modality(design_id)
        else:
            design_id = ""
            impedance_raw = np.load(self.impedance_dir / f"{filename}.npy")
            occupancy_np = np.load(self.occupancy_dir / f"{filename}.npy").reshape(-1)
            if impedance_raw.ndim == 2:
                impedance_t = torch.from_numpy(impedance_raw[:1]).float()
            else:
                impedance_t = torch.from_numpy(impedance_raw.flatten()).float().unsqueeze(0)
            occupancy_t = torch.from_numpy(occupancy_np).float().clamp(0.0, 1.0)
            k_from_layout = int(occupancy_t.sum().item())

        if self.k_values is not None:
            k_t = torch.tensor(self.k_values[idx], dtype=torch.long)
        else:
            k_t = torch.tensor(k_from_layout, dtype=torch.long)

        out: dict = {
            "heatmap_norm": heatmap_norm_t,
            "occupancy": occupancy_t,
            "impedance": impedance_t,
            "K": k_t,
            "filename": filename,
        }
        if design_id:
            out["design_id"] = design_id
        if self.has_pifreq:
            pifreq_path = self.pifreq_dir / f"{filename}.npy"
            if pifreq_path.exists():
                from src_vae.others.pi_freq_utils import pi_freq_hz_to_norm
                out["PI_freq"] = torch.tensor(
                    pi_freq_hz_to_norm(float(np.load(pifreq_path))), dtype=torch.float32,
                )
        return out

    def __len__(self) -> int:
        return self.n_samples

    def _get_filename(self, idx: int) -> str:
        return self.heatmap_files[idx].stem

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        if self._ram is not None:
            return self._ram[idx]
        return self._load_sample(idx)


def create_data_loaders(
    data_dir: str = "datasets/data_norm",
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = False,
    stats_path: Optional[str] = None,
    train_split: float = 0.8,
    seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = 2,
    *,
    stratify_by_k: bool = False,
    balance_k: bool = False,
    k_balance_power: float = 1.0,
    k_balance_smoothing: float = 1e-3,
    k_balance_print: bool = True,
    cache_in_ram: bool = False,
    drop_last_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""

    dataset = VAEDataset(
        data_dir=data_dir,
        normalize=normalize,
        stats_path=stats_path,
        precompute_k=(stratify_by_k or balance_k),
        cache_in_ram=cache_in_ram,
        require_layout_store=False,
    )

    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train
    generator = torch.Generator().manual_seed(seed)

    if stratify_by_k:
        if dataset.k_values is None:
            raise RuntimeError("Internal error: dataset.k_values was not precomputed")

        rng = torch.Generator().manual_seed(seed)
        k_to_indices: dict[int, list[int]] = {}
        for i, k in enumerate(dataset.k_values):
            k_to_indices.setdefault(int(k), []).append(i)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for _, idxs in sorted(k_to_indices.items()):
            if len(idxs) == 1:
                train_indices.extend(idxs)
                continue
            perm = torch.randperm(len(idxs), generator=rng).tolist()
            idxs = [idxs[j] for j in perm]
            n_k = len(idxs)
            n_train_k = max(1, min(int(n_k * train_split), n_k - 1))
            train_indices.extend(idxs[:n_train_k])
            val_indices.extend(idxs[n_train_k:])

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        n_train, n_val = len(train_dataset), len(val_dataset)
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=generator,
        )

    print(f"Train samples: {n_train}, Validation samples: {n_val}")

    dl_kwargs: dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

    sampler = None
    if balance_k:
        subset_indices = getattr(train_dataset, "indices", None)
        if subset_indices is None:
            raise RuntimeError("K-balanced sampling requires train_dataset to be a Subset")
        train_indices = list(subset_indices)
        if dataset.k_values is None:
            raise RuntimeError("Internal error: dataset.k_values was not precomputed")

        k_train = [dataset.k_values[i] for i in train_indices]
        counts = Counter(k_train)
        weights = [float((counts[k] + k_balance_smoothing) ** (-k_balance_power)) for k in k_train]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        if k_balance_print:
            k_min, k_max = min(k_train), max(k_train)
            nonzero = [(k, counts[k]) for k in range(k_min, k_max + 1) if counts.get(k, 0) > 0]
            print("K-balanced sampling enabled")
            print(f"  K range in train split: {k_min}..{k_max}")
            print("  Train K counts (non-zero):")
            print("   " + ", ".join([f"{k}:{c}" for k, c in nonzero[:26]]) + (" ..." if len(nonzero) > 26 else ""))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=drop_last_train,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )
    return train_loader, val_loader


def collate_fn(batch: list) -> Dict[str, torch.Tensor | list[str]]:
    out: Dict = {
        "heatmap_norm": torch.stack([item["heatmap_norm"] for item in batch]),
        "occupancy": torch.stack([item["occupancy"] for item in batch]),
        "impedance": torch.stack([item["impedance"] for item in batch]),
        "K": torch.stack([item["K"] for item in batch]),
        "filenames": [item["filename"] for item in batch],
    }
    if batch and "PI_freq" in batch[0]:
        out["PI_freq"] = torch.stack([b["PI_freq"] for b in batch])
    return out


if __name__ == "__main__":
    print("Testing VAE DataLoader...")
    train_loader, val_loader = create_data_loaders(
        data_dir="datasets/data_multifreq_norm",
        batch_size=4,
        num_workers=0,
        normalize=False,
        train_split=0.8,
    )
    for batch in train_loader:
        print(f"Heatmap shape: {batch['heatmap_norm'].shape}")
        print(f"Impedance shape: {batch['impedance'].shape}")
        break
    print("DataLoader test completed successfully!")
