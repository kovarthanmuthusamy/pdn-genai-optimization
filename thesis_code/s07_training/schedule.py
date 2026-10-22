"""Curriculum: which weights and ramps apply at a given (0-based) epoch.

    epoch  0 ............ penalty_warmup_epochs   impedance top-k / peak terms ramp 0 -> 1
           cross_freq_start_epoch                 cross-frequency loss on
           anticopy_start_epoch                   anti-copy loss on
           modality_dropout_protect_heatmap_epoch heatmap expert never dropped
           heatmap_focus_start_epoch              heatmap_focus_* weights replace the base ones
           impedance_peak_start_epoch             peak-index term ramps 0 -> 1 over impedance_peak_ramp_epochs

Usage (import only):
phase_weights(epoch, c)       {"heatmap_weight", "impedance_weight", "cross_freq_weight"}
modality_dropout_at(epoch, c) expert dropout probability
penalty_scale(epoch, c)       ramp of the impedance top-k and peak terms
peak_loss_scale(epoch, c)     ramp of the impedance peak-index term
milestones(c)                 {epoch: message} printed when a phase starts
curriculum_summary(c)         one-line description for the log
"""
from __future__ import annotations


def phase_weights(epoch: int, c) -> dict[str, float]:
    if epoch >= c.heatmap_focus_start_epoch:
        return {"heatmap_weight": c.heatmap_focus_heatmap_weight,
                "impedance_weight": c.heatmap_focus_impedance_weight,
                "cross_freq_weight": c.heatmap_focus_cross_freq_weight}
    return {"heatmap_weight": c.heatmap_weight, "impedance_weight": c.impedance_weight,
            "cross_freq_weight": c.cross_freq_weight}


def modality_dropout_at(epoch: int, c) -> float:
    return c.heatmap_focus_modality_dropout if epoch >= c.heatmap_focus_start_epoch else c.modality_dropout


def penalty_scale(epoch: int, c) -> float:
    return min(1.0, epoch / max(c.penalty_warmup_epochs, 1))


def peak_loss_scale(epoch: int, c) -> float:
    if epoch < c.impedance_peak_start_epoch:
        return 0.0
    return min(1.0, (epoch - c.impedance_peak_start_epoch + 1) / max(c.impedance_peak_ramp_epochs, 1))


def milestones(c) -> dict[int, str]:
    return {c.cross_freq_start_epoch: "cross-frequency + anti-copy on",
            c.heatmap_focus_start_epoch: "heatmap-focus weights, heatmap expert pinned on",
            c.impedance_peak_start_epoch: "impedance peak-index term starts"}


def curriculum_summary(c) -> str:
    return (f"Curriculum (of {c.num_epochs} epochs): impedance penalty warm-up 0->{c.penalty_warmup_epochs}, "
            f"cross_freq@{c.cross_freq_start_epoch}, anticopy@{c.anticopy_start_epoch}, "
            f"protect_hm@{c.modality_dropout_protect_heatmap_epoch}, heatmap_focus@{c.heatmap_focus_start_epoch}, "
            f"imp_peak@{c.impedance_peak_start_epoch} (ramp {c.impedance_peak_ramp_epochs})")
