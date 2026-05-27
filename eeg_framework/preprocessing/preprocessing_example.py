from .preprocessing import BaseStep
from typing import Optional, Union
import mne

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
# class Example(BaseStep)
# ...

# Preprocessing -----------------------------------------------------------------------------
# class Example(BaseStep)
# ...