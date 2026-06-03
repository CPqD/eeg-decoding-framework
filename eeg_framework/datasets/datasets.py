from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
import mne

class EpochBuilder:
    """
    Generic builder for creating mne.Epochs objects from NumPy arrays.
    
    This class optimizes the preprocessing pipeline by configuring structural 
    parameters (Info, montage, filters, and windowing) only once during 
    initialization. The `build` method can then be called repeatedly for 
    different subjects or sessions.
    """
    def __init__(
        self,
        sfreq:        float,
        ch_names:     List[str],
        event_id:     Dict[str, int],
        tmin_ms:      float,
        tmax_ms:      float, 
        montage_name: str = "standard_1005",
        l_freq:       Optional[float] = None, 
        h_freq:       Optional[float] = None,
        notch_freq:   Optional[float] = None,
        verbose:      Union[bool, str, int, None]  = False ,
    ):
        """
        Parameters
        ----------
        sfreq : float
            Sampling frequency (Hz) of the dataset.
        ch_names : List[str]
            List of EEG channel names.
        event_id : Dict[str, int]
            Dictionary mapping class names to integers (e.g., {'right': 1, 'left': 2}).
        tmin_ms : float
            Epoch start time in milliseconds, relative to the stimulus (e.g., -500.0).
        tmax_ms : float
            Epoch end time in milliseconds, relative to the stimulus (e.g., 4000.0).
        montage_name : str, optional
            Name of the standard MNE montage for spatial mapping (default: "standard_1005").
        l_freq : float, optional
            Cut-off frequency for the high-pass filter in Hz.
        h_freq : float, optional
            Cut-off frequency for the low-pass filter in Hz.
        notch_freq : float, optional
            Central frequency for the notch filter in Hz (e.g., 60.0).
        verbose : bool | str | int | None, optional
            Controls the verbosity of internal MNE functions.
        """
        # Convert milliseconds to seconds (MNE standard)
        self.tmin_s = tmin_ms * 1e-3
        self.tmax_s = tmax_ms * 1e-3
        self.event_id = event_id

        # Filtering parameters
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.verbose = verbose

        # Build Info object (optimized: executed only once)
        self.info = mne.create_info(
            ch_names=ch_names, 
            sfreq=sfreq, 
            ch_types=["eeg"] * len(ch_names)
        )
        self.info.set_montage(montage_name, on_missing="warn", verbose=self.verbose) 
    
    def _apply_filter(
        self,
        raw:        mne.io.RawArray,
    ) -> None:
        """Apply bandpass and/or notch filter to EEG channels in-place."""
        if self.l_freq is not None or self.h_freq is not None:
            raw.filter(
                l_freq  = self.l_freq,
                h_freq  = self.h_freq,
                picks   = "eeg",
                method  = "fir",
                verbose = self.verbose,
            )
        if self.notch_freq is not None:
            raw.notch_filter(
                freqs   = self.notch_freq,
                picks   = "eeg",
                method  = "fir",
                verbose = self.verbose,
            )
        
    def build(
        self,
        x:             np.ndarray,
        onsets:        np.ndarray,
        labels:        np.ndarray,
        metadata_dict: Dict,
    ) -> mne.Epochs:
        """
        Builds the mne.Epochs object from raw signal and event matrices.
        
        Parameters
        ----------
        x : np.ndarray
            Continuous signal matrix with shape (n_samples, n_channels).
        onsets : np.ndarray
            1D array with the indices (in samples) of the start of each trial.
        labels : np.ndarray
            1D array containing the class (integer) corresponding to each trial.
        metadata_dict : Dict
            Dictionary containing the metadata of the current block (e.g., {'subject': 1, 'session': 2}).
            
        Returns
        -------
        mne.Epochs
            Loaded, filtered, and epoched MNE object with attached metadata.
        """
        n_trials = len(onsets)

        # 1. Create continuous data
        # The input matrix (n_samples, n_channels) is transposed because 
        # mne.io.RawArray expects the shape (n_channels, n_samples).
        raw = mne.io.RawArray(x.T, self.info, verbose=self.verbose)

        # 2. Apply temporal filters (In-place)
        self._apply_filter(raw)

        # 3. Build MNE-style events array (n_trials, 3)
        # Column 1: Sample index | Column 2: 0 (unused) | Column 3: Event code
        events_arr = np.column_stack([
            onsets,
            np.zeros(n_trials, dtype=int),
            labels,
        ])

        # 4. Epoching
        epochs = mne.Epochs(
            raw,
            events=events_arr,
            event_id=self.event_id,
            tmin=self.tmin_s,
            tmax=self.tmax_s,
            picks="eeg",
            preload=True,
            baseline=None,
            verbose=self.verbose
        )

        # 5. Attach metadata to facilitate tracking and cross-validation
        epochs.metadata = pd.DataFrame(metadata_dict, index=range(n_trials))

        return epochs
