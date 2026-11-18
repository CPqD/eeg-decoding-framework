from .preprocessing_base import BaseStep
from typing import Optional, Union
import mne
import warnings
import numpy as np

# -------------------------------------------------------------------------------------
# 1. CAR - Common Average Reference
# -------------------------------------------------------------------------------------
class CAR(BaseStep):
    """
    Common Average Reference (re-reference to the mean of all EEG channels).

    Parameters
    ----------
    projection : bool
        If True, add CAR as a projector (lazy, can be removed later).
        If False (default), apply directly to the data.
    verbose : bool | str | int | None
    """
    
    def __init__(
            self,
            projection: bool                        = False,
            verbose:    Union[bool, str, int, None] = False
    ) -> None:
        self.projection = projection
        self.verbose    = verbose

    def transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        out = epochs.copy()
        out.set_eeg_reference(
            ref_channels = 'average',
            projection   = self.projection,
            verbose      = self.verbose,
        )
        if self.projection:
            out.apply_proj(verbose=self.verbose)
        return out
    
# ---------------------------------------------------------------------------------------------
# 2. Resample
# ---------------------------------------------------------------------------------------------
class Resample(BaseStep):
    """
    Resample epochs to a new sampling frequency.
 
    Useful to reduce data dimensionality and computational cost before
    feature extraction. Uses MNE's anti-aliasing resampling.
 
    Note: Resample before ICA and AutoReject for faster processing.
    Do not upsample — it does not add information.
 
    Parameters
    ----------
    sfreq : float
        Target sampling frequency in Hz. Default 256.0.
    verbose : bool | str | int | None
 
    Examples
    --------
    >>> Resample(256)   # downsample to 256 Hz
    >>> Resample(128)   # downsample to 128 Hz
    """

    def __init__(
            self,
            sfreq:   float                        = 256.0,
            verbose: Union[bool, str, int, None]  = False 
    ) -> None:
        self.sfreq   = sfreq
        self.verbose = verbose
    
    def transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        return epochs.copy().resample(sfreq=self.sfreq, verbose=self.verbose)

# Preprocessing -----------------------------------------------------------------------------
class Crop(BaseStep):
    """
    Crop epochs to a time window of interest.
 
    Useful to focus on the relevant interval of a trial and reduce
    data dimensionality. For imagined speech, the action interval
    is typically 0–3.5 s relative to cue onset.
 
    Parameters
    ----------
    tmin : float | None
        Start of the crop window in seconds. None = keep original start.
    tmax : float | None
        End of the crop window in seconds. None = keep original end.
    verbose : bool | str | int | None
 
    Examples
    --------
    >>> Crop(tmin=0, tmax=3.5)   # keep action interval only
    >>> Crop(tmin=0)              # remove pre-stimulus baseline
    """

    def __init__(
            self, 
            tmin:    Optional[float] = None,
            tmax:    Optional[float] = None,
            verbose: Union[bool, str, int, None] = False
    ) -> None:
        self.tmin    = tmin
        self.tmax    = tmax
        self.verbose = verbose

    def transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        return epochs.copy().crop(tmin=self.tmin,tmax=self.tmax)

# ---------------------------------------------------------------------------------------------
# 4. ICA
# ---------------------------------------------------------------------------------------------
class ICA(BaseStep):
    """
    ICA-based artifact removal.
 
    Strategy (automatic, in priority order):
      1. EOG channels present  → find_bads_eog per channel (correlation-based)
      2. EMG channels present  → find_bads_eog per channel (correlation-based)
      3. Neither available     → ICLabel (mne-icalabel) neural classifier
 
    For best results with ICLabel, use method='infomax' with extended=True.
    For best ICA performance, data should be high-pass filtered at 1 Hz
    before fitting — though this may not always be possible with short epochs.
 
    Parameters
    ----------
    n_components : int | float | None
        Number of ICA components. Float = fraction of variance explained.
        Default None (uses rank of data).
    method : str
        ICA algorithm: 'fastica' (default), 'infomax', 'picard'.
    max_iter : int | str
        Max iterations. 'auto' lets MNE decide.
    eog_ch : str | list[str] | None
        Explicit EOG channel name(s). If None, auto-detected by channel type.
    emg_ch : str | list[str] | None
        Explicit EMG channel name(s). If None, auto-detected by channel type.
    eog_threshold : float
        Z-score threshold for EOG component detection (default 3.0).
        Lower = more aggressive removal.
    emg_threshold : float
        Z-score threshold for EMG component detection (default 3.0).
    icalabel_threshold : float
        ICLabel confidence threshold (default 0.7). Components classified as
        artifact above this threshold are removed.
    random_state : int
        Random seed for reproducibility.
    verbose : bool | str | int | None
 
    Attributes
    ----------
    ica_ : mne.preprocessing.ICA
        Fitted ICA object (available after fit).
    exclude_ : list[int]
        Indices of components marked for removal.
    strategy_ : str
        Which strategy was used: 'eog', 'emg', 'eog+emg', or 'icalabel'.
 
    Examples
    --------
    >>> ICA()                              # auto-detect strategy
    >>> ICA(eog_ch='EOG1')                 # explicit EOG channel
    >>> ICA(eog_threshold=2.0)             # more aggressive
    >>> ICA(method='infomax',              # best for ICLabel
    ...     eog_ch=None, emg_ch=None)
    """

    def __init__(
            self, 
            n_components:       Optional[Union[int, float]] = None,
            method:             str                         = 'fastica',
            fit_params:         Optional[dict]              = None,
            max_iter:           Union[int, str]             = 'auto',
            eog_ch:             Optional[Union[str, list]]  = None,
            emg_ch:             Optional[Union[str, list]]  = None,
            eog_threshold:      Union[float, str]           = 3.0,
            emg_threshold:      Union[float, str]           = 3.0,
            icalabel_threshold: float                       = 0.7,
            random_state:       int                         = 42,
            verbose:            Union[bool, str, int, None] = False
    ) -> None:
        self.n_components       = n_components
        self.method             = method
        self.fit_params         = fit_params
        self.max_iter           = max_iter
        self.eog_ch             = eog_ch
        self.emg_ch             = emg_ch
        self.eog_threshold      = eog_threshold
        self.emg_threshold      = emg_threshold
        self.icalabel_threshold = icalabel_threshold
        self.random_state       = random_state
        self.verbose            = verbose
    
        self.ica_      = None
        self.exclude_  = []
        self.strategy_ = None

    def fit(self, epochs: mne.BaseEpochs) -> "ICA":
        ica = mne.preprocessing.ICA(
            n_components = self.n_components,
            method       = self.method, 
            fit_params   = self.fit_params,
            max_iter     = self.max_iter,
            random_state = self.random_state,
            verbose      = self.verbose
        )
        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        epochs_eeg = epochs.copy().pick(eeg_picks)

        ica.fit(epochs_eeg, verbose=self.verbose)
        self.ica_ = ica

        # --- Detect which channels are available -------------------------------
        eog_chs = self._resolve_channels(epochs, self.eog_ch, 'eog')
        emg_chs = self._resolve_channels(epochs, self.emg_ch, 'emg')

        exclude    = []
        strategies = []
    
        # EMG and EOG
        if eog_chs:
            found = []
            for eog_ch in eog_chs: 
                idx, _ = ica.find_bads_eog(
                    epochs,
                    ch_name=eog_ch,
                    threshold=self.eog_threshold,
                    verbose=self.verbose,
                )
                found.extend(idx)
            if found:
                exclude.extend(found)
                strategies.append('eog')
        
        if emg_chs:
            found = []
            for emg_ch in emg_chs:
                idx, _ = ica.find_bads_eog(
                    epochs,
                    ch_name=emg_ch,
                    threshold=self.emg_threshold,
                    verbose=self.verbose,
                )
                found.extend(idx)
            if found:
                exclude.extend(found)
                strategies.append('emg')

        # Falls into icalabel only if there is no external signals
        if not eog_chs and not emg_chs:
            exclude = self._icalabel(epochs_eeg, ica)
            strategies.append("icalabel")
        
        self.exclude_  = list(set(exclude))
        self.strategy_ = "+".join(strategies) if strategies else "none_found"

        print(
            f"[ICA] strategy={self.strategy_} | "
            f"components removed: {self.exclude_}"
        )
        return self
    
    def transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        if self.ica_ is None:
            raise RuntimeError("Call fit() before transform().")
        out = epochs.copy()
        self.ica_.exclude = self.exclude_
        self.ica_.apply(out, verbose=self.verbose)
        return out

    # ---- helpers --------------------------------------------------
    def _resolve_channels(
        self,
        epochs: mne.BaseEpochs,
        explicit: Optional[Union[str, list]],
        ch_type: str,
    ) -> list[str]:
        """Return explicit channel list, or auto-detect by channel type."""
        if explicit is not None:
            chs = [explicit] if isinstance(explicit, str) else list(explicit)
            missing = [c for c in chs if c not in epochs.ch_names]
            if missing:
                warnings.warn(f"Channels not found and will be ignored: {missing}")
            return [c for c in chs if c in epochs.ch_names]
 
        # auto-detect by type
        picks = mne.pick_types(epochs.info, **{ch_type: True})
        return [epochs.ch_names[p] for p in picks]

    def _icalabel(
            self, 
            epochs: mne.BaseEpochs,
            ica:    mne.preprocessing.ICA,
    ) -> list[int]:
        """USE ICLabel neural classifier to flag artifact components."""
        try:
            from mne_icalabel import label_components
        except ImportError:
            warnings.warn(
                "mne-icalabel not installed. Skipping automatic ICA labeling.\n"
                "Install with: pip install mne-icalabel",
                RuntimeWarning,
            )
            return []
        
        labels = label_components(epochs, ica, method='iclabel')
        artifact_labels = {"eye blink", "muscle artifact", "heart beat",
                           "channel noise", "line noise"}
        exclude = []
        for i, (label, proba) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
            confidence = float(np.max(proba))
            if label in artifact_labels and confidence >= self.icalabel_threshold:
                exclude.append(i)
                print(f"  [ICLabel] component {i:02d}: {label} ({confidence:.2f})")
        return exclude

# -------------------------------------------------------------------------------------
# 1. Filter
# -------------------------------------------------------------------------------------
class Filter(BaseStep):
    """
    Frequency filter (bandpass / highpass / lowpass / notch).
 
    All four filter types are available in a single step.
    Any combination can be used simultaneously.
 
    Note on filter length: FIR filters require a minimum signal length
    that depends on the cutoff frequency. For short epochs (< 5s),
    avoid l_freq < 1.0 Hz — it will cause distortion. Use l_freq=None
    and rely on CAR to remove slow drifts instead.
 
    Parameters
    ----------
    l_freq : float | None
        High-pass cutoff in Hz. None = no high-pass.
    h_freq : float | None
        Low-pass cutoff in Hz. None = no low-pass.
    notch_freqs : float | list[float] | None
        Notch filter frequency/frequencies in Hz (e.g. 50 or 60 for
        line noise, [50, 100] for harmonics). Applied after band filter.
    notch_width : float
        Width of the notch in Hz (default 0.5).
    picks : str | list[str]
        Channels to filter. Default 'eeg'.
    method : str
        Filter method passed to MNE: 'fir' (default) or 'iir'.
    verbose : bool | str | int | None
 
    Examples
    --------
    >>> Filter(l_freq=1, h_freq=40)                  # bandpass
    >>> Filter(l_freq=1)                              # highpass only
    >>> Filter(h_freq=40)                             # lowpass only
    >>> Filter(l_freq=1, h_freq=40, notch_freqs=50)  # bandpass + notch
    >>> Filter(notch_freqs=[50, 100])                 # notch only
    >>> Filter(h_freq=40, notch_freqs=50)             # safe for short epochs
    """

    def __init__(
            self, 
            l_freq:       Optional[float]             = None, 
            h_freq:       Optional[float]             = None, 
            notch_freqs:  Optional[float]             = None, 
            notch_width:  float                       = 0.5,
            picks:        Union[str, list]            = 'eeg', 
            method:       str                         = 'fir', 
            verbose:      Union[bool, str, int, None] = False
            )-> None:
        self.l_freq      = l_freq
        self.h_freq      = h_freq
        self.notch_freqs = notch_freqs
        self.notch_width = notch_width
        self.picks       = picks
        self.method      = method
        self.verbose     = verbose
    
    def transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        out = epochs.copy()

        if self.l_freq is not None or self.h_freq is not None:
            out = out.filter(
                l_freq  = self.l_freq, 
                h_freq  = self.h_freq, 
                picks   = self.picks, 
                method  = self.method, 
                verbose = self.verbose
            )
        
        if self.notch_freqs is not None:
            freqs = (
                [self.notch_freqs]
                if isinstance(self.notch_freqs, (int, float))
                else list(self.notch_freqs)
            )
            data = out.get_data()
            data = mne.filter.notch_filter(
                x            = data,
                Fs           = out.info['sfreq'],
                freqs        = freqs,
                notch_widths = self.notch_width,
                verbose      = self.verbose
            )
            out._data = data

        return out