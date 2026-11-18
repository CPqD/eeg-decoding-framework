import mne 
import warnings
import matplotlib.pyplot as plt
from typing import Union, Optional
import numpy as np

# ---- Default colors -----------------------------------------------------------------------------------------

_CMAP_COLOR = "jet"

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
    """# ...
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

def plot_topomap(
    epochs:   mne.BaseEpochs, 
    times:    Union[float, list[float]]   = 0.1, 
    event_id: Optional[dict[str, int]]    = None,
    fmin:     Optional[float]             = None, 
    fmax:     Optional[float]             = None, 
    baseline: Optional[tuple]             = (None, 0), 
    vmin:     Optional[float]             = None, 
    vmax:     Optional[float]             = None, 
    savepath: Optional[str]               = None, 
    show:     bool                        = True, 
    verbose:  Union[bool, str, int, None] = False
) -> plt.Figure:
    """
    Scalp topography at one or more time points, one row per class.
 
    Parameters
    ----------
    times : float | list[float]
        Time point(s) in seconds to plot.
    """
    if isinstance(times, (int, float)):
        times = [times]
    
    if fmin is not None or fmax is not None:
        epochs = epochs.copy().filter(l_freq=fmin, h_freq=fmax, verbose=verbose)

    evokeds = _class_evokeds(epochs, event_id)
    if not evokeds:
        raise ValueError("No evokeds could be computed.")

    n_classes = len(evokeds)
    n_times = len(times)

    fig, axes = plt.subplots(n_classes, n_times, figsize=(3*n_times, 3*n_classes), squeeze=False)
    
    for row, (name, evk) in enumerate(evokeds.items()):
        evk = evk.copy()
        if baseline is not None:
            evk.apply_baseline(baseline, verbose=verbose)
        
        for col, t in enumerate(times):
            ax = axes[row][col]
            evk.plot_topomap(times=t, axes=ax, show=False, colorbar=False, vlim=[vmin, vmax], time_unit='s', cmap=_CMAP_COLOR)
            if col == 0:
                ax.set_ylabel(name, fontsize=10, labelpad=40)
            if row == 0:
                ax.set_title(f"{t*1000:.0f} ms", fontsize=10)
            else:
                ax.set_title('')
    
    fig.suptitle(_fig_title(epochs, 'Topomaps'), fontsize=12, y=1.01)
    fig.tight_layout()

    if show:
        plt.show()

# ---- 3. Power Spectral Density (PSD)

def plot_psd(
    epochs:   mne.BaseEpochs, 
    picks:    Union[str, list[str]]       = "eeg", 
    event_id: Optional[dict[str, int]]    = None,
    fmin:     float                       = 1.0, 
    fmax:     float                       = 60.0, 
    tmin:     Optional[float]             = 0.0, 
    tmax:     Optional[float]             = None,
    method:   str                         = "welch", 
    n_fft:    int                         = 512, 
    savepath: Optional[str]               = None, 
    show:     bool                        = True,
    verbose:  Union[bool, str, int, None] = False
) -> plt.Figure:
    """
    Power spectral density averaged across channels and trials, per class.
 
    Parameters
    ----------
    fmin, fmax : float
        Frequency range in Hz.
    tmin, tmax : float | None
        Time window within the epoch to use for PSD estimation.
    method : str
        ``'welch'`` (default) or ``'multitaper'``.
    n_fft : int
        FFT length for Welch method.
    """
    evokeds = _class_evokeds(epochs, event_id)
    if not evokeds:
        raise ValueError("No evokeds could be computed.")

    fig, ax = plt.subplots(figsize=(9, 4))

    for idx, (name, _) in enumerate(evokeds.items()):
        
        sub = epochs[name]
        if tmin is not None or tmax is not None:
            sub = sub.copy().crop(tmin=tmin, tmax=tmax)
        
        spectrum = sub.compute_psd(method=method, fmin=fmin, fmax=fmax, n_fft=n_fft, picks=picks, verbose=verbose)

        # Mean across channels and epochs -> (n_freqs, )
        psds, freqs = spectrum.get_data(return_freqs=True)
        psd_db = 10 * np.log10(psds.mean(axis=(0,1)) + 1e-30)
        
        ax.plot(freqs, psd_db, color=_CLASS_COLORS[idx % len(_CLASS_COLORS)],linewidth=1.8,label=name)
    
    ax.legend(loc='upper right', fontsize=9)
    fig.canvas.draw()
    
    # Mark canonical bands
    for band, (lo, hi, color) in {
        "δ": (1,  4,    "#aec6cf"),
        "θ": (4,  8,    "#b5ead7"),
        "α": (8,  13,   "#ffdac1"),
        "β": (13, 30,   "#ffdac1"),
        "γ": (30, fmax, "#ff9aa2"),
    }.items():
        ax.axvspan(lo, min(hi, fmax), alpha=0.18, color=color, label=None)
        mid = (lo + min(hi, fmax)) / 2
        if mid < fmax:
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] > -200 else -80,
                    band, ha="center", fontsize=8, color="grey")
    
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(_fig_title(epochs, "PSD"))
    fig.tight_layout()

    if show:
        plt.show()
    
# ---- 4. Time-frequency (ERDS / Morlet) ----

def plot_tfr(
    epochs:        mne.BaseEpochs,
    picks:         Union[str, list[str]]       = "eeg",
    event_id:      Optional[dict[str, int]]    = None,
    freqs:         Optional[np.ndarray]        = None,
    n_cycles:      Union[float, np.ndarray]    = 7.0,
    tmin:          Optional[float]             = None,
    tmax:          Optional[float]             = None,
    baseline:      Optional[tuple]             = (-0.5, 0.0),
    baseline_mode: str                         = "percent",
    vmin:          float                       = -1.0,
    vmax:          float                       = 1.0,
    log_scale:     bool                        = True,
    savepath:      Optional[str]               = None,
    show:          bool                        = True,
    verbose:       Union[bool, str, int, None] = False,
) -> plt.Figure:
    """
    Time-frequency representation (Morlet wavelets) averaged per class.
 
    Displays the mean TFR across selected channels. Uses ERDS-style
    baseline normalisation by default (``baseline_mode='percent'``).
 
    Parameters
    ----------
    freqs : np.ndarray | None
        Frequencies to analyse. Default: 4–40 Hz, 40 points log-spaced
        (``np.geomspace``). Pass a custom array to override.
    n_cycles : float | np.ndarray
        Number of cycles per frequency (or array of per-frequency values).
    baseline_mode : str
        ``'percent'``, ``'ratio'``, ``'logratio'``, ``'mean'``, ``'zscore'``.
    vmin, vmax : float
        Colour scale limits.
    log_scale : bool
        If True (default), use logarithmic Y axis with pcolormesh.
        If False, use linear imshow (legacy behaviour).
    """
    if freqs is None:
        # Log-spaced: better resolution at low freqs, matches perceptual scale
        freqs = np.geomspace(4, 100, num=40, dtype=float)
 
    if isinstance(n_cycles, (int, float)):
        n_cycles = np.clip(freqs / 2.0, 1.0, n_cycles)
 
    evokeds_id = event_id or epochs.event_id
    classes    = {k: v for k, v in evokeds_id.items()
                  if (epochs.events[:, 2] == v).sum() > 0}
 
    if not classes:
        raise ValueError("No valid classes found in epochs.")
 
    n   = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
 
    # Reserve space on the right for the shared colorbar
    fig.subplots_adjust(right=0.88)
 
    im = None
    for ax, (idx, (name, code)) in zip(axes[0], enumerate(classes.items())):
        sub = epochs[name]
        if picks != "eeg":
            pick_list = picks if isinstance(picks, list) else [picks]
        else:
            pick_list = "eeg"
 
        tfr = sub.compute_tfr(
            method     = 'morlet',
            freqs      = freqs,
            n_cycles   = n_cycles,
            picks      = pick_list,
            average    = True,
            return_itc = False,
            verbose    = verbose,
        )
 
        if baseline is not None:
            tfr.apply_baseline(baseline, mode=baseline_mode, verbose=verbose)
 
        if tmin is not None or tmax is not None:
            tfr.crop(tmin=tmin, tmax=tmax)
 
        # Mean across channels → (n_freqs, n_times)
        data       = tfr.data.mean(axis=0)
        times_plot = tfr.times
 
        if log_scale:
            # ---- Time edges --------------------------------------------------
            dt      = np.diff(times_plot)
            t_edges = np.concatenate([
                [times_plot[0]  - dt[0]  / 2],
                (times_plot[:-1] + times_plot[1:]) / 2,
                [times_plot[-1] + dt[-1] / 2],
            ])
 
            # ---- Frequency edges in log space --------------------------------
            log_f   = np.log10(freqs)
            df_log  = np.diff(log_f)
            f_edges = np.concatenate([
                [10 ** (log_f[0]  - df_log[0]  / 2)],
                10 ** ((log_f[:-1] + log_f[1:]) / 2),
                [10 ** (log_f[-1] + df_log[-1] / 2)],
            ])
 
            im = ax.pcolormesh(
                t_edges, f_edges, data,
                vmin    = vmin,
                vmax    = vmax,
                cmap    = _CMAP_COLOR,
                shading = 'flat',
            )
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
 
            # Ticks at canonical EEG band landmarks within freq range
            tick_freqs = [f for f in [4, 8, 13, 20, 30, 40, 60, 80]
                          if freqs[0] <= f <= freqs[-1]]
            if tick_freqs:
                ax.set_yticks(tick_freqs)
                ax.set_yticklabels([str(f) for f in tick_freqs])
 
        else:
            # Legacy linear imshow
            im = ax.imshow(
                data,
                aspect = 'auto',
                origin = 'lower',
                extent = [times_plot[0], times_plot[-1], freqs[0], freqs[-1]],
                vmin   = vmin,
                vmax   = vmax,
                cmap   = _CMAP_COLOR,
            )
 
        ax.axvline(0, color="k", linewidth=0.8, linestyle=":")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Time (s)')
        if idx == 0:
            ax.set_ylabel('Frequency (Hz)')
 
    # Shared colorbar in dedicated axes — does not resize any subplot
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(im, cax=cbar_ax, label=baseline_mode)
 
    fig.suptitle(_fig_title(epochs, "TFR (Morlet)"), fontsize=12, y=1.02)
 
    if show:
        plt.show()