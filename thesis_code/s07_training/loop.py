"""The training loop: set-up, one epoch, and the full run with validation and checkpoints.

What: per epoch, train on ``train_samples_per_epoch`` random draws (split over the GPUs). On epoch 1,
    every ``checkpoint_interval`` epochs and the last epoch, rank 0 validates on the held-out
    placements, writes the metric files and a checkpoint; every ``crossfreq_eval_interval`` epochs
    (on validated epochs) it runs the cross-frequency check and keeps the best weights as
    ``best_off_anchor_model.pt``. ``last_model.pt`` is always the final epoch.
Usage (import only; the entry point is train.py):
    train_vae(extra_config=None, max_batches=None)   the full run into config.experiment_dir
    load_config(extra_config=None)   common/inputs/config.yaml (+ an optional second file on top)
    build_model(c), build_optimizer(model, c), seed_everything(seed, rank)
    apply_normalization_stats(c)     background value + clip bounds from the dataset -> spectrum log-std
    prepare_batch(batch, c)          loader batch -> (map, placement one-hot, spectrum, K, frequency)
    run_epoch(model, loader, c, epoch, ...)   one train or validation pass -> mean of every loss term
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from thesis_code.s03_normalization.stats import load_norm_stats
from thesis_code.s08_monitoring.observability import RunObservability
from thesis_code.s08_monitoring.logger import VAETrainingLogger

from thesis_code.s07_training.config import CONFIG_FILE, TrainConfig, load_train_config, vae_model_kwargs
from thesis_code.common.board import zero_offboard
from thesis_code.s04_dataset.loader import create_multifreq_data_loaders
from thesis_code.common.placement import ensure_batch_onehot
from thesis_code.s06_losses.guards import clamp_logvar, prepare_recon_for_loss, sanitize_loss_dict
from thesis_code.s06_losses.objective import LOSS_KEYS, anticopy_loss, cross_freq_loss, cycle_consistency_loss, vae_loss
from thesis_code.s05_model import MultiInputVAE
from thesis_code.s07_training.checkpoints import checkpoint_dict, clean_previous_run, load_model_weights, resolve_resume
from thesis_code.s08_monitoring.crossfreq_eval import run_crossfreq_eval
from thesis_code.s07_training.distributed import (barrier, broadcast_float, cleanup_distributed, get_world_size, init_distributed,
                          is_main_process, unwrap_model, wrap_ddp)
from thesis_code.s08_monitoring.metrics import (EPOCH_TIMING_HEADER, RESUMABLE_CSVS, append_epoch_csv, log_epoch_metrics,
                      truncate_csv_after_epoch)
from thesis_code.s07_training.schedule import (curriculum_summary, milestones, modality_dropout_at, peak_loss_scale, penalty_scale,
                       phase_weights)

LATENT_STATS_EVERY = 50     # per-K latent statistics on checkpoint epochs divisible by this (+ last)


# ── Set-up ───────────────────────────────────────────────────────────────────

def load_config(extra_config: Path | None = None) -> TrainConfig:
    paths = [CONFIG_FILE]
    if extra_config is not None and Path(extra_config).is_file():
        paths.append(Path(extra_config))
    return load_train_config(paths)


def build_model(c: TrainConfig) -> MultiInputVAE:
    model = MultiInputVAE(**vae_model_kwargs(c))
    if is_main_process():
        print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params  latent={c.latent_dim}  "
              f"K 0..{c.k_max}  mod_drop={c.modality_dropout}", flush=True)
    return model


def build_optimizer(model, c: TrainConfig) -> optim.AdamW:
    """AdamW, weight decay 1e-4, one learning rate for every parameter (fused kernel on CUDA)."""
    params = list(model.parameters())
    if c.is_cuda():
        for fast in ({"fused": True}, {"foreach": True}):
            try:
                return optim.AdamW(params, lr=float(c.learning_rate), weight_decay=1e-4, **fast)
            except (TypeError, RuntimeError):
                pass
    return optim.AdamW(params, lr=float(c.learning_rate), weight_decay=1e-4)


def seed_everything(seed: int, rank: int) -> None:
    """Seed python / numpy / torch with ``seed + rank``. Not bit-reproducible: the cross-frequency
    partner draws are unseeded."""
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if is_main_process():
        print(f"Seeded: {seed} (+rank {rank} -> {s})", flush=True)


def apply_normalization_stats(c: TrainConfig) -> float:
    """Set background_value and the map clip bounds from the dataset's normalization_stats.json;
    attach the statistics (``c._norm_stats``). Returns the spectrum log-std."""
    raw = json.loads((Path(c.data_dir) / "normalization_stats.json").read_text())
    c.background_value = float(raw["background_value"])
    c.heatmap_z_clip_min = float(raw["Heatmap"]["clip_min"])
    c.heatmap_z_clip_max = float(raw["Heatmap"]["clip_max"])
    c._norm_stats = load_norm_stats(c.data_dir)
    return float(raw["Impedance"]["log_std"]) or 1.0


def prepare_batch(batch: dict, c: TrainConfig):
    """Batch -> (heatmap (B,1,64,64), occupancy (B,52,T), impedance (B,1,231), K (B,), PI_freq (B,)).

    The heatmap is clipped to the dataset's clip bounds and set to exactly 0 off the board, so the
    encoder input and the loss target agree without any value threshold.
    """
    nb, dev = c.is_cuda(), c.device
    hm = batch["heatmap_norm"].to(dev, non_blocking=nb)
    occ = ensure_batch_onehot(batch["occupancy"].to(dev, non_blocking=nb), c.n_decap_types)
    imp = batch["impedance"].to(dev, non_blocking=nb)
    K = batch["K"].to(dev, non_blocking=nb)
    pi = batch["PI_freq"].to(dev, non_blocking=nb)
    if hm.dim() == 3:
        hm = hm.unsqueeze(1)
    if imp.dim() == 1:
        imp = imp.unsqueeze(0)
    if imp.dim() == 2 and imp.shape[-1] == 231:
        imp = imp.unsqueeze(1)
    elif imp.dim() == 3:
        imp = imp[:, :1]
    hm = hm.clamp(c.heatmap_z_clip_min, c.heatmap_z_clip_max)
    return zero_offboard(hm), occ, imp, K, pi


# ── One epoch ────────────────────────────────────────────────────────────────

class LatentStats:
    """Validation-only accumulator: per-dim posterior mean / aggregate spread, per-expert summary
    and (optionally) the same per K -- what checkpoints store for the latent search."""

    def __init__(self, per_k: bool):
        self.per_k = per_k
        self.n = 0
        self.mu_sum = self.mu_sq = self.sig_sum = None
        self.experts: dict[str, list[float]] = {}
        self.k_buckets: dict[int, dict] = {}
        self.batches = 0

    def add(self, mu: torch.Tensor, logvar: torch.Tensor, experts: dict, K: torch.Tensor) -> None:
        dim = mu.shape[1]
        if self.mu_sum is None:
            self.mu_sum, self.mu_sq, self.sig_sum = torch.zeros(dim), torch.zeros(dim), torch.zeros(dim)
        mc, sc = mu.cpu(), (0.5 * logvar).exp().cpu()
        self.mu_sum += mc.sum(0)
        self.mu_sq += mc.pow(2).sum(0)
        self.sig_sum += sc.sum(0)
        self.n += mc.shape[0]
        self.batches += 1
        for name, (em, elv) in experts.items():
            es = (elv / 2).exp()
            a = self.experts.setdefault(name, [0.0, 0.0, 0.0, 0.0])
            a[0] += em.mean().item()
            a[1] += em.std().item()
            a[2] += es.mean().item()
            a[3] += es.std().item()
        if self.per_k:
            kc = K.cpu()
            for kv in kc.unique().tolist():
                m = kc == int(kv)
                b = self.k_buckets.setdefault(int(kv), {"mu": torch.zeros(dim), "sq": torch.zeros(dim),
                                                        "sig": torch.zeros(dim), "n": 0})
                b["mu"] += mc[m].sum(0)
                b["sq"] += mc[m].pow(2).sum(0)
                b["sig"] += sc[m].sum(0)
                b["n"] += m.sum().item()

    @staticmethod
    def _spread(s_mu, s_sq, s_sig, n):
        mu = s_mu / n
        return mu, ((s_sq / n - mu.pow(2)).clamp(min=0) + (s_sig / n).pow(2)).sqrt()

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"modality_stats": {}, "per_K_latent_stats": {}}
        mu, std = self._spread(self.mu_sum, self.mu_sq, self.sig_sum, self.n)
        per_k = {}
        for kv, b in self.k_buckets.items():
            if b["n"] >= 2:
                mk, sk = self._spread(b["mu"], b["sq"], b["sig"], b["n"])
                per_k[str(kv)] = {"latent": {"mu_mean_per_dim": mk.tolist(), "agg_std_per_dim": sk.tolist()}}
        return {
            "mu_mean": float(mu.mean()), "mu_std": float(mu.std()), "mu_min": float(mu.min()),
            "mu_max": float(mu.max()), "std_mean": float(std.mean()),
            "per_dim_stats": {"latent": {"mu_mean_per_dim": mu.tolist(), "agg_std_per_dim": std.tolist()}},
            "modality_stats": {n: {"mu_mean": a[0] / self.batches, "mu_std": a[1] / self.batches,
                                   "std_mean": a[2] / self.batches, "std_std": a[3] / self.batches}
                               for n, a in self.experts.items()},
            "per_K_latent_stats": per_k,
        }


def _skip_batch(optimizer, why: str, epoch: int, i: int, skips: int) -> int:
    optimizer.zero_grad(set_to_none=True)
    skips += 1
    if skips <= 2 or skips % 50 == 0:
        print(f"  [nan-guard ep {epoch}] skip batch {i} ({why}) skips={skips}", flush=True)
    return skips


def run_epoch(model, loader, c: TrainConfig, epoch: int, beta: float, md: float, imp_log_std: float,
              *, train: bool, optimizer=None, collect_per_k: bool = False,
              max_batches: int | None = None) -> dict[str, Any]:
    """One pass over ``loader`` (at most ``max_batches`` batches). Returns the mean of every
    LOSS_KEYS term; validation adds the latent statistics (``LatentStats``)."""
    model.train(train)
    base = unwrap_model(model)
    # Set on the unwrapped module: the model never reads attributes of the DDP wrapper.
    base.modality_dropout = md if train else 0.0
    base.modality_dropout_protect_heatmap = train and epoch >= c.modality_dropout_protect_heatmap_epoch
    gpu_acc = train and c.is_cuda()   # accumulate on the GPU: no host sync per batch
    acc = {k: torch.zeros((), device=c.device) if gpu_acc else 0.0 for k in LOSS_KEYS}
    n_batches = nan_skips = 0
    latent = None if train else LatentStats(per_k=collect_per_k)
    weights = phase_weights(epoch, c)
    scales = {"penalty_scale": penalty_scale(epoch, c), "peak_scale": peak_loss_scale(epoch, c)}
    params = list(model.parameters())

    with torch.enable_grad() if train else torch.inference_mode():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            hm, occ, imp, K, pi = prepare_batch(batch, c)
            if train:
                optimizer.zero_grad(set_to_none=True)
                z, mu, lv, ex = base.encode(hm, occ, imp, K, pi)
                rh, ro, ri = base.decode(z, K, pi, occupancy=occ)
                cycle = cycle_consistency_loss(base, z, K, pi)
            else:
                rh, ro, ri, mu, lv, ex = model(hm, occ, imp, K, pi)
            rh, lv = prepare_recon_for_loss(rh, c), clamp_logvar(lv, c)
            losses = vae_loss(rh, ro, ri, hm, occ, imp, mu, lv, ex, K, c=c, beta=beta,
                              imp_log_std=imp_log_std, weights=weights, **scales)

            if train:
                losses["cycle_consistency_loss"] = cycle
                losses["total_loss"] = losses["total_loss"] + c.full_encode_cycle_consistency_weight * cycle
                if epoch >= c.cross_freq_start_epoch:
                    pi_alt = batch["PI_freq_alt"].to(c.device, non_blocking=c.is_cuda())
                    hm_alt = batch["heatmap_norm_alt"].to(c.device, non_blocking=c.is_cuda())
                    cf, rh_cf, hm_alt = cross_freq_loss(base, z, K, occ, pi_alt, hm_alt, c=c)
                    losses["cross_freq_heatmap_loss"] = cf
                    losses["total_loss"] = losses["total_loss"] + weights["cross_freq_weight"] * cf
                    if epoch >= c.anticopy_start_epoch:
                        ac = anticopy_loss(rh_cf, hm_alt, hm)
                        losses["anticopy_loss"] = ac
                        losses["total_loss"] = losses["total_loss"] + c.anticopy_weight * ac
            losses = sanitize_loss_dict(losses, c)

            if train:
                # Under DDP every rank must agree to skip BEFORE backward, or one rank all-reduces
                # gradients the other never sends and NCCL hangs.
                finite = torch.tensor([1.0 if torch.isfinite(losses["total_loss"]).all() else 0.0],
                                      device=losses["total_loss"].device)
                if get_world_size() > 1:
                    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if c.training_skip_nonfinite_batches and finite.item() < 0.5:
                    nan_skips = _skip_batch(optimizer, "nonfinite loss", epoch, i, nan_skips)
                    continue
                losses["total_loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float(c.training_grad_clip_norm))
                if not (torch.isfinite(grad_norm) and all(p.grad is None or torch.isfinite(p.grad).all() for p in params)):
                    nan_skips = _skip_batch(optimizer, "nonfinite grad", epoch, i, nan_skips)
                    continue
                optimizer.step()
            else:
                latent.add(mu, lv, ex, K)

            for k in LOSS_KEYS:
                if k in losses:
                    acc[k] = acc[k] + losses[k].detach() if gpu_acc else acc[k] + losses[k].item()
            n_batches += 1

    out: dict[str, Any] = {k: ((v / n_batches).item() if gpu_acc else v / n_batches) for k, v in acc.items()}
    out["beta"], out["nan_skips"] = beta, nan_skips
    out.update(latent.summary() if latent is not None else {"modality_stats": {}, "per_K_latent_stats": {}})
    return out


# ── Full run ─────────────────────────────────────────────────────────────────

def _save(path: Path, ep, model, opt, sched, tr, val, c) -> None:
    torch.save(checkpoint_dict(ep, model, opt, sched, tr["total_loss"], val, c,
                               VAETrainingLogger.build_latent_stats(val), val.get("per_K_latent_stats", {})), path)


def _resuming(c: TrainConfig) -> bool:
    return isinstance(c.resume_checkpoint, str) and Path(c.resume_checkpoint).exists()


def _optimizer_and_scheduler(c: TrainConfig, model, start: int):
    """AdamW + ReduceLROnPlateau; on resume the optimizer moments are restored, and the learning
    rate is reset (``reset_lr_on_resume``) or the scheduler state restored."""
    opt = build_optimizer(model, c)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=c.lr_factor,
                                                 patience=c.lr_patience, min_lr=c.lr_min)
    if not _resuming(c):
        return opt, sched
    saved = torch.load(c.resume_checkpoint, map_location="cpu", weights_only=False)
    if "optimizer_state_dict" in saved:
        try:
            opt.load_state_dict(saved["optimizer_state_dict"])
        except ValueError:
            print("  optimizer state skipped (param groups changed)")
    if c.reset_lr_on_resume:
        for pg in opt.param_groups:
            pg["lr"] = c.learning_rate
        print(f"  Reset LR on resume (epoch start={start}, optimizer moments kept, scheduler fresh)")
    elif "scheduler_state_dict" in saved:
        sched.load_state_dict(saved["scheduler_state_dict"])
    return opt, sched


def train_vae(extra_config: Path | None = None, max_batches: int | None = None) -> None:
    # ── set-up ──
    local_rank, rank, world_size = init_distributed()
    c = load_config(extra_config)
    main = is_main_process()
    if main:
        print("  " + curriculum_summary(c), flush=True)
    seed_everything(c.seed, rank)
    if c.is_cuda() and world_size > 1:
        c.device = f"cuda:{local_rank}"
    if c.resume_checkpoint is not None:
        resolve_resume(c)

    exp = Path(c.experiment_dir)
    ckpt_dir, log_dir, metrics_dir = exp / "checkpoints", exp / "logs", exp / "metrics"
    if main and not c.resume_checkpoint:
        clean_previous_run(exp)
    barrier()
    for p in (ckpt_dir, log_dir, metrics_dir / "plots", exp / "evals"):
        p.mkdir(parents=True, exist_ok=True)
    logger = VAETrainingLogger(str(metrics_dir), str(ckpt_dir), str(metrics_dir / "loss.csv"),
                               checkpoint_interval=c.checkpoint_interval)
    if main:
        logger.log_start(exp.name, {"device": c.device, "batch": c.batch_size, "world_size": world_size,
                                    "epochs": c.num_epochs, "checkpoint_interval": c.checkpoint_interval,
                                    "train_samples_per_epoch": c.train_samples_per_epoch})
    imp_log_std = apply_normalization_stats(c)

    workers = c.num_workers
    train_ld, val_ld = create_multifreq_data_loaders(
        data_dir=c.data_dir, batch_size=c.batch_size, num_workers=workers, train_split=c.train_split,
        seed=c.split_seed, pin_memory=c.is_cuda(), persistent_workers=c.persistent_workers and workers > 0,
        prefetch_factor=c.prefetch_factor if workers > 0 else None,
        train_samples_per_epoch=c.train_samples_per_epoch, val_batch_size=c.val_batch_size,
        ddp_rank=rank, ddp_world_size=world_size, n_decap_types=c.n_decap_types)
    if c.is_cuda():
        # full fp32: mixed precision gave NaN gradients in this model
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    model = build_model(c).to(c.device)
    start = load_model_weights(c.resume_checkpoint, model, c.device)[0] if _resuming(c) else 0
    model = wrap_ddp(model, local_rank=local_rank)
    barrier()
    opt, sched = _optimizer_and_scheduler(c, model, start)

    timing_path = metrics_dir / "timing.json"
    prev_t = json.loads(timing_path.read_text()).get("training_time_seconds", 0) if timing_path.exists() else 0
    t0 = time.time()
    epoch_timing_path = metrics_dir / "epoch_timing.csv"
    if main:
        print(f"epochs {start + 1}-{c.num_epochs}  batch={c.batch_size}  checkpoint_interval={c.checkpoint_interval}\n")
        if start > 0:
            for name in RESUMABLE_CSVS:
                if truncate_csv_after_epoch(metrics_dir / name, start):
                    print(f"  Truncated post-resume rows in metrics/{name}")
        elif not epoch_timing_path.is_file():
            with epoch_timing_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(EPOCH_TIMING_HEADER)
    obs = None
    if main:   # metrics/run_meta.json + logs/train_events.jsonl
        obs = RunObservability(exp_name=exp.name, metrics_dir=metrics_dir, log_dir=log_dir,
                               cfg=c.to_checkpoint_dict(), start_epoch=int(start), world_size=int(world_size),
                               device=str(c.device))
        obs.start()
    barrier()

    # ── epochs ──
    phase_msgs = milestones(c)
    train_times, val_times = [], []
    val: dict[str, Any] = {}
    tr: dict[str, Any] = {}
    last_val_loss, best_cf = float("inf"), float("-inf")
    cf_csv = metrics_dir / "crossfreq_eval.csv"
    if main and start > 0 and cf_csv.is_file():   # resume: a later, lower score must not replace the best checkpoint
        with cf_csv.open(newline="", encoding="utf-8") as f:
            done = [float(r["pearson_at_target"]) for r in csv.DictReader(f) if int(r["epoch"]) <= start]
        if done and (ckpt_dir / "best_off_anchor_model.pt").is_file():
            best_cf = max(done)
            print(f"  Resume: best crossfreq score so far {best_cf:.4f}", flush=True)
    beta = float(c.beta_final)
    ep = start
    for epoch in range(start, c.num_epochs):
        if main and epoch in phase_msgs and epoch > 0:
            print(f"  >> epoch {epoch + 1}: {phase_msgs[epoch]}", flush=True)
        t_ep0 = time.perf_counter()
        md = modality_dropout_at(epoch, c)
        tr = run_epoch(model, train_ld, c, epoch, beta, md, imp_log_std, train=True, optimizer=opt,
                       max_batches=max_batches)
        t_train = time.perf_counter() - t_ep0
        train_times.append(t_train)
        barrier()   # rank-0-only validation must not race the other ranks into the broadcast

        ep = epoch + 1
        save_ckpt = logger.at_checkpoint_epoch(ep)
        run_val = ep == 1 or ep >= c.num_epochs or ep % c.checkpoint_interval == 0
        t_val = 0.0
        if run_val:
            if main:
                t_v0 = time.perf_counter()
                val = run_epoch(unwrap_model(model), val_ld, c, epoch, beta, 0.0, imp_log_std, train=False,
                                collect_per_k=save_ckpt and (ep % LATENT_STATS_EVERY == 0 or ep >= c.num_epochs),
                                max_batches=max_batches)
                t_val = time.perf_counter() - t_v0
                last_val_loss = float(val["total_loss"])
                val_times.append(t_val)
            last_val_loss = broadcast_float(last_val_loss, device=c.device)
            barrier()
            sched.step(last_val_loss)
        t_total = time.perf_counter() - t_ep0

        if main:
            append_epoch_csv(epoch_timing_path, ep, EPOCH_TIMING_HEADER,
                             [ep, f"{t_train:.2f}", f"{t_val:.2f}", f"{t_total:.2f}", int(run_val),
                              f"{tr['total_loss']:.6f}", f"{last_val_loss:.4f}" if run_val else ""])
            if save_ckpt or ep == 1:
                log_epoch_metrics(logger, obs, metrics_dir, ep, beta, tr, val, run_val)
            if ep == 1 or ep >= c.num_epochs or ep % max(1, c.epoch_print_interval) == 0:
                w = phase_weights(epoch, c)
                val_str = f"  val={val['total_loss']:.4f}" if run_val else ""
                print(f"Ep {ep:>3}/{c.num_epochs}  {t_total:.1f}s  loss={tr['total_loss']:.4f}{val_str}  "
                      f"hm={tr['heatmap_loss']:.4f}x{w['heatmap_weight']:.1f}  cf={tr['cross_freq_heatmap_loss']:.4f}  "
                      f"acp={tr['anticopy_loss']:.4f}  cyc={tr['cycle_consistency_loss']:.4f}  "
                      f"occ={tr['occupancy_loss']:.4f}  imp={tr['impedance_loss']:.4f}  md={md:.2f}", flush=True)
            if save_ckpt:
                _save(ckpt_dir / f"checkpoint_epoch_{ep}.pt", ep, model, opt, sched, tr, val, c)
                timing_path.write_text(json.dumps({"training_time_seconds": prev_t + int(time.time() - t0)}))
            if run_val and c.crossfreq_eval_interval > 0 and ep % c.crossfreq_eval_interval == 0:
                row = run_crossfreq_eval(unwrap_model(model), val_ld, max_batches=c.crossfreq_eval_max_batches,
                                         device=c.device, epoch=ep, out_csv=metrics_dir / "crossfreq_eval.csv",
                                         n_decap_types=c.n_decap_types)
                score = row.get("pearson_at_target")
                obs.log_off_anchor(ep, [], aggregate_score=score)
                if score is not None and score > best_cf:
                    best_cf = score
                    _save(ckpt_dir / "best_off_anchor_model.pt", ep, model, opt, sched, tr, val, c)
                    print(f"  crossfreq score {score:.4f} (best) -> best_off_anchor_model.pt", flush=True)
                elif score is not None:
                    print(f"  crossfreq score {score:.4f}  (best {best_cf:.4f})", flush=True)
        barrier()

    # ── finish (rank 0 writes every file: two ranks writing timing.json at once corrupted it) ──
    total_s = prev_t + int(time.time() - t0)
    avg_train = sum(train_times) / max(len(train_times), 1)
    avg_val = sum(val_times) / len(val_times) if val_times else 0.0
    if main:
        timing_path.write_text(json.dumps({
            "training_time_seconds": total_s, "segment_epochs": len(train_times), "resume_start_epoch": int(start),
            "last_completed_epoch": int(start + len(train_times)), "num_epochs_target": int(c.num_epochs),
            "avg_train_sec": round(avg_train, 2), "avg_val_sec": round(avg_val, 2), "val_runs": len(val_times),
        }, indent=2))
        if val and ep % c.checkpoint_interval != 0:          # last epoch is not a checkpoint epoch
            log_epoch_metrics(logger, obs, metrics_dir, ep, beta, tr, val, True)
        if val.get("modality_stats"):                        # last_model.pt = the last epoch, always
            _save(ckpt_dir / "last_model.pt", ep, model, opt, sched, tr, val, c)
            best = f"; best crossfreq score={best_cf:.4f} in best_off_anchor_model.pt" if best_cf > float("-inf") else ""
            print(f"  last_model.pt <- epoch {ep}{best}", flush=True)
        else:
            print(f"  Skipping last_model.pt -- no validation metrics (start={start}, last_epoch={ep})", flush=True)
        print(f"\nTiming: {total_s // 60}m total | avg train {avg_train:.1f}s/ep | avg val {avg_val:.1f}s", flush=True)
        logger.log_complete(val.get("total_loss", float("nan")), ckpt_dir, f"{total_s // 60}m")
        obs.end(status="finished", best_oa_score=best_cf if best_cf > float("-inf") else None)
    barrier()
    cleanup_distributed()
    if main:
        plots = metrics_dir / "plots"
        logger.plot(save_path=str(plots / "convergence_final.png"))
        logger.plot_loss_components(save_path=str(plots / "loss_components_final.png"))
        logger.plot_overfitting(save_path=str(plots / "overfitting_final.png"))
        logger.print_statistics()
        # runtime snapshot (with the values read from the dataset); config.yaml is never rewritten
        (exp / "config.runtime.yaml").write_text(json.dumps(c.to_checkpoint_dict(), indent=2), encoding="utf-8")
