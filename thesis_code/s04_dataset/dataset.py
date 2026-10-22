"""PyTorch dataset of the layout store: one item per map, with its placement and spectrum.

What: item i is the i-th map file in name order (sample_1, sample_10, sample_100, ...), with
        heatmap_norm  (1, 64, 64) normalized map (not clipped)
        occupancy     (52,) part codes 0..3 as float (the collate function builds the one-hot)
        impedance     (1, 231) normalized spectrum of the placement
        K             number of decaps
        PI_freq       the map's frequency as model input in [0, 1]
        filename, design_id, heatmap_only (always 0)
    The per-row K values are cached in k_values_cache.npy. A cache already in the dataset folder
    is read; a missing or stale one is built in ``cache_dir`` (default OUTPUT_DIR/s04_dataset/cache),
    so the dataset folder is never written.
Usage (import only): VAEDataset(data_dir, precompute_k=True, cache_in_ram=False, cache_dir=None)
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from thesis_code.common.frequency import pi_freq_hz_to_norm
from thesis_code.common import settings
from thesis_code.s04_dataset.store import (layout_imp_path, layout_occ_path, layout_type_path, load_manifest_index,
                                           validate_multifreq_dataset)


def cache_file(data_dir: Path, name: str, cache_dir: Path | None = None) -> tuple[Path, Path]:
    """(path to read, path to write) of a cache file: the dataset's own copy is read when present;
    new caches go to a folder per dataset under ``cache_dir`` (OUTPUT_DIR/s04_dataset/cache)."""
    data_dir = Path(data_dir).resolve()
    tag = f"{data_dir.name}_{hashlib.sha1(str(data_dir).encode()).hexdigest()[:8]}"
    own = (Path(cache_dir) if cache_dir is not None else Path(settings.OUTPUT_DIR) / "s04_dataset" / "cache") / tag
    return (data_dir / name if (data_dir / name).exists() else own / name), own / name


class VAEDataset(Dataset):
    """Map + placement + spectrum (+ frequency) per row of a layout store."""

    def __init__(self, data_dir: str, precompute_k: bool = False, cache_in_ram: bool = False, *,
                 cache_dir: Path | None = None):
        self.data_dir = Path(data_dir)
        self.cache_dir = cache_dir
        self.heatmap_dir = self.data_dir / "heatmap"
        self.pifreq_dir = self.data_dir / "PI_freq"
        validate_multifreq_dataset(self.data_dir)
        self.manifest_index = load_manifest_index(self.data_dir)
        n_layouts = sum(1 for p in (self.data_dir / "layouts").iterdir() if p.is_dir())
        print(f"Layout store: {n_layouts} layouts, {len(self.manifest_index)} manifest rows")
        self.heatmap_files = sorted(self.heatmap_dir.glob("*.npy"))
        self.n_samples = len(self.heatmap_files)
        print(f"Loaded {self.n_samples} samples from {data_dir}")

        self.k_values: Optional[list[int]] = None
        if precompute_k:
            self._precompute_k_values()
        self._layout_cache: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        self._ram: list[dict] | None = None
        if cache_in_ram:
            t0 = time.perf_counter()
            self._ram = [self._load_sample(i) for i in range(self.n_samples)]
            print(f"  RAM cache ready in {time.perf_counter() - t0:.1f}s", flush=True)

    def _precompute_k_values(self) -> None:
        read, write = cache_file(self.data_dir, "k_values_cache.npy", self.cache_dir)
        if read.exists():
            cached = np.load(str(read))
            if len(cached) == self.n_samples:
                self.k_values = cached.tolist()
                print(f"K-values loaded from cache ({self.n_samples} samples)")
                return
        print(f"Computing K values for {self.n_samples} samples (one-time, cached in {write.parent})")
        design_k: dict[str, int] = {}
        k_values: list[int] = []
        for hf in self.heatmap_files:
            did = self._design_id_for_stem(hf.stem)
            if did not in design_k:
                design_k[did] = int((self._load_occ_codes(did) > 0).sum())
            k_values.append(design_k[did])
        self.k_values = k_values
        write.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(write), np.array(k_values, dtype=np.int16))

    def _load_occ_codes(self, design_id: str) -> np.ndarray:
        """Per-slot codes: type.npy (0..T), else occ.npy."""
        type_path = layout_type_path(self.data_dir, design_id)
        path = type_path if type_path.is_file() else layout_occ_path(self.data_dir, design_id)
        return np.asarray(np.load(path, mmap_mode="r")).reshape(-1)

    def _design_id_for_stem(self, stem: str) -> str:
        did = self.manifest_index.get(stem)
        if not did:
            raise ValueError(f"No design_id in manifest.csv for {stem}")
        return did

    def _load_layout_modality(self, design_id: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        if design_id in self._layout_cache:
            return self._layout_cache[design_id]
        impedance_raw = np.array(np.load(layout_imp_path(self.data_dir, design_id)), dtype=np.float32, copy=True)
        occupancy_np = np.array(self._load_occ_codes(design_id), dtype=np.float32, copy=True)
        impedance_t = torch.from_numpy(impedance_raw.reshape(1, -1).copy())   # stored (1, 231)
        triple = (impedance_t, torch.from_numpy(occupancy_np), int((occupancy_np > 0).sum()))
        self._layout_cache[design_id] = triple
        return triple

    def _load_sample(self, idx: int) -> dict:
        filename = self.heatmap_files[idx].stem
        heatmap_norm_t = torch.from_numpy(np.array(np.load(self.heatmap_dir / f"{filename}.npy"),
                                                   dtype=np.float32, copy=True))
        design_id = self._design_id_for_stem(filename)
        impedance_t, occupancy_t, k_from_layout = self._load_layout_modality(design_id)
        k = self.k_values[idx] if self.k_values is not None else k_from_layout
        return {"heatmap_norm": heatmap_norm_t, "occupancy": occupancy_t, "impedance": impedance_t,
                "K": torch.tensor(k, dtype=torch.long), "filename": filename, "design_id": design_id,
                "heatmap_only": torch.tensor(0.0, dtype=torch.float32),
                "PI_freq": torch.tensor(pi_freq_hz_to_norm(float(np.load(self.pifreq_dir / f"{filename}.npy"))),
                                        dtype=torch.float32)}

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        if self._ram is not None:
            return self._ram[idx]
        return self._load_sample(idx)
