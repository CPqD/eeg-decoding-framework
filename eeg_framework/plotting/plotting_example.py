"""
plotting
====================
Plotting utilities for EEG exploration and debugging.
All functions accept an mne.Epochs (or EpochsArray) and return
the matplotlib Figure so the caller can save, tweak or close it.
 
Functions
---------
plot_erp_compare    — One subplot per class, side-by-side comparison
"""

import mne 
import warnings
import matplotlib.pyplot as plt
from typing import Union, Optional

# ---- Default colors -----------------------------------------------------------------------------------------

_CLASS_COLORS = ["#082bc5", "#f30e0e", "#09b917", "#814e0a", "#ba2bc7",
                 "#8c564b", "#d3239e", "#3b3a3a"]

# ---- Helpers ------------------------------------------------------------------------------------------------

def _class_evokeds(
    epochs:   mne.BaseEpochs, 
    event_id: Optional[dict[str, int]] = None,
) -> dict[str, mne.Evoked]:
    """Return a dict {class_name: Evoked} averaged across trials."""
    eid = event_id or epochs.event_id
    evokeds: dict[str, mne.Evoked] = {}
    for name, code in eid.items():
        mask = epochs.events[:, 2] == code
        if mask.sum() == 0:
            warnings.warn(f"No trials found for class '{name}' (code {code}).")
            continue
        evokeds[name] = epochs[name].average()
    return evokeds

def _fig_title(
    epochs: mne.BaseEpochs, 
    extra:  str = ""
) -> str:
    meta = (getattr(epochs, "_bci2020_meta", None) or
            getattr(epochs, "_inner_speech_meta", None) or {})
    parts = []
    if meta.get("dataset"):
        parts.append(meta["dataset"])
    if meta.get("subject"):
        parts.append(f"S{meta['subject']:02d}")
    if meta.get("split"):
        parts.append(meta["split"])
    if extra:
        parts.append(extra)
    return " · ".join(parts) if parts else extra

# ---- 1. ERP per-class comparison ---------------------------------------------------------------------------------------------------------------------------------------

def plot_erp_compare(
    epochs:   mne.BaseEpochs, 
    picks:    Union[str, list[str]]       = "eeg", 
    event_id: Optional[dict[str, int]]    = None,
    fmin:     Optional[float]             = None, 
    fmax:     Optional[float]             = None, 
    tmin:     Optional[float]             = None, 
    tmax:     Optional[float]             = None, 
    baseline: Optional[tuple]             = (None, 0), 
    show:     bool                        = True, 
    verbose:  Union[bool, str, int, None] = False
) -> plt.Figure:
    """
    One subplot per class showing butterfly ERP.
 
    Useful to quickly inspect whether each class has a distinguishable
    temporal pattern.
    """
    if fmin is not None or fmax is not None:
        epochs = epochs.copy().filter(l_freq=fmin, h_freq=fmax, verbose=verbose)

    evokeds = _class_evokeds(epochs, event_id)
    if not evokeds:
        raise ValueError("No evokeds could be computed.")


    n = len(evokeds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (idx, (name, evk)) in zip(axes, enumerate(evokeds.items())):
        evk = evk.copy()
        if tmin is not None or tmax is not None:
            evk.crop(tmin=tmin, tmax=tmax)
        if baseline is not None:
            evk.apply_baseline(baseline, verbose=verbose)

        ch_idx = mne.pick_types(evk.info, eeg=True) if picks == "eeg" \
            else [evk.ch_names.index(p) for p in picks if p in evk.ch_names]
 
        data_uv = evk.data[ch_idx]
        color   = _CLASS_COLORS[idx % len(_CLASS_COLORS)]

        for ch_data in data_uv:
            ax.plot(evk.times, ch_data, color=color, alpha=0.10, linewidth=0.5)
        ax.plot(evk.times, data_uv.mean(axis=0), color=color, linewidth=2)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="k", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Time (s)")
        if idx == 0:
            ax.set_ylabel("Amplitude (µV)")
    
    fig.suptitle(_fig_title(epochs, "ERP per class"), fontsize=12, y=1.02)
    fig.tight_layout()

    if show:
        plt.show()

# ---- 2 Topomap -----
# ....

# ---- 3. Time-frequency (ERDS / Morlet) ----
# ...

# ---- 4. Other plots -----
# ...