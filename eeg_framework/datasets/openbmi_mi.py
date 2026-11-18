import warnings
import numpy as np
import mne
import pandas as pd
 
from pathlib import Path
from typing import Literal, Union, Optional
from scipy.io import loadmat

from .datasets import EpochBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_TYPE = Literal["train", "test"]

EVENT_ID = {
    "right": 1,
    "left":  2,
}

# 62 EEG channel names in order
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",   "F4",   "F8",   "FC5",  "FC1",  "FC2",
    "FC6", "T7",  "C3",  "Cz",  "C4",   "T8",   "TP9",  "CP5",  "CP1",  "CP2",
    "CP6", "TP10","P7",  "P3",  "Pz",   "P4",   "P8",   "PO9",  "O1",   "Oz",
    "O2",  "PO10","FC3", "FC4", "C5",   "C1",   "C2",   "C6",   "CP3",  "CPz",
    "CP4", "P1",  "P2",  "POz", "FT9",  "FTT9h","TTP7h","TP7",  "TPP9h","FT10",
    "FTT10h","TPP8h","TP8","TPP10h","F9","F10",  "AF7",  "AF3",  "AF4",  "AF8",
    "PO3", "PO4",
]

# Trial Window
DEFAULT_TMIN_MS = -500
DEFAULT_TMAX_MS = 4000

# EEG Montage
MONTAGE = "standard_1005"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mat_path(root_dir: Path, subject: int, session: int) -> Path:
    return root_dir / f"subj{subject:02d}" / f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"

def _extract_channel_names(chan_field) -> list[str]:
    """Parse channel names from the nested MATLAB char cell array."""
    try:
        names = [str(ch[0]) for ch in chan_field[0]]
        if len(names) > 0:
            return names
    except Exception:
        pass
    warnings.warn(
        "Could not parse channel names from mat file. "
        "Falling back to built-in CHANNEL_NAMES list.",
        RuntimeWarning,
    )
    return CHANNEL_NAMES

# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
 
class OpenBMIMIReader:
    """
    Reader for the OpenBMI Motor Imagery dataset (Lee et al., 2019).
 
    Loads a single .mat file (one subject, one session) and returns
    an mne.EpochsArray with optional bandpass filtering.
 
    Parameters
    ----------
    root_dir : str | Path
        Directory containing the .mat files
        (e.g., sess01_subj01_EEG_MI.mat).
    split : {'train', 'test'}
        Which partition to load. Default 'train'.
    tmin_ms : float
        Epoch start in milliseconds relative to cue onset. Default -500.
    tmax_ms : float
        Epoch end in milliseconds relative to cue onset. Default 4000.
    l_freq : float | None
        High-pass cut-off applied to continuous data before epoching.
        Paper uses 8 Hz for MI classification. None = no high-pass.
    h_freq : float | None
        Low-pass cut-off. Paper uses 30 Hz. None = no low-pass.
    notch_freq : float | None
        Notch filter frequency (e.g., 60.0 for Korea power line).
        None = no notch.
    verbose : bool | str | int | None
    """

    def __init__(
            self,
            root_dir:   Union[str, Path],
            split:      SPLIT_TYPE                  = "train",
            tmin_ms:    float                       = DEFAULT_TMIN_MS,
            tmax_ms:    float                       = DEFAULT_TMAX_MS,
            l_freq:     Optional[float]             = None,
            h_freq:     Optional[float]             = None,
            notch_freq: Optional[float]             = None,
            verbose:    Union[bool, str, int, None] = False,
    ) -> None:
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.tmin_ms    = tmin_ms
        self.tmax_ms    = tmax_ms
        self.l_freq     = l_freq
        self.h_freq     = h_freq
        self.notch_freq = notch_freq
        self.verbose    = verbose

        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{self.split}'")        
    

    def get_epochs(self, subject: int, session: int) -> mne.EpochsArray:
        """
        Load and return mne.EpochsArray for one subject and session.
 
        Pipeline applied internally:
            1. Load .mat file
            2. Extract fields
            3. Apply framework's epoch builder
 
        Parameters
        ----------
        subject : int, 1–54
        session : int, 1–2
 
        Returns
        -------
        mne.EpochsArray
            event_id = {'right': 1, 'left': 2}
            sfreq    = 1000 Hz (or as in file)
            channels = 62 EEG
        """
        if not (1 <= subject <= 54):
            raise ValueError(f"subject must be 1-54, got {subject}")
        if not (1 <= session <= 2):
            raise ValueError(f"session must be 1-2, got {session}")
        
        mat_path = _mat_path(self.root_dir, subject, session)
        if not mat_path.exists():
            raise FileNotFoundError(
                f"File not found: {mat_path}\n"
                "Download from https://gigadb.org/dataset/100542"
            )

        # ---- 1. Load .mat ------------------------------------------------
        mat      = loadmat(str(mat_path), squeeze_me=False, struct_as_record=True)
        key      = f"EEG_MI_{self.split}"
        eeg_struct = mat[key][0, 0]

        # ---- 2. Extract fields -------------------------------------------
        x        = np.array(eeg_struct["x"]) # (n_samples, 62) µV
        t        = np.squeeze(eeg_struct["t"]).astype(int) # (n_trials, ) sample onsets
        fs       = float(np.squeeze(eeg_struct["fs"]))
        labels   = np.squeeze(eeg_struct["y_dec"]).astype(int) # 1 = right, 2 = left
        ch_names = _extract_channel_names(eeg_struct["chan"])

        # ---- 3. build epochs -------------------------------------------
        metadata = pd.DataFrame({
            "subject": subject,
            "session": session,
            "split":   self.split,
        }, index=range(len(t)))

        epochs = EpochBuilder(
            sfreq=fs,
            ch_names=ch_names,
            event_id=EVENT_ID,
            montage_name=MONTAGE,
            tmin_ms=self.tmin_ms,
            tmax_ms=self.tmax_ms,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            notch_freq=self.notch_freq,
            verbose=self.verbose,
        ).build(x=x, onsets=t, labels=labels,metadata_dict=metadata)

        # ---- Print ---------------------------------------------------
                
        print(
            f"[OpenBMIMI] subject={subject} session={session} "
            f"split='{self.split}' | "
            f"{len(epochs)} epochs | {len(epochs.ch_names)} ch | "
            f"{epochs.info['sfreq']:.0f} Hz"
        )
        return epochs
        

# ---------------------------------------------------------------------------
# Multi-subject loader
# ---------------------------------------------------------------------------

def load_openbmi_mi(
    root_dir:    Union[str, Path],
    subjects:    Optional[list[int]]         = None,
    sessions:    Optional[list[int]]         = None,
    split:       SPLIT_TYPE                  = "train",
    tmin_ms:     float                       = DEFAULT_TMIN_MS,
    tmax_ms:     float                       = DEFAULT_TMAX_MS,
    l_freq:      Optional[float]             = None,
    h_freq:      Optional[float]             = None,
    notch_freq:  Optional[float]             = None,
    concatenate: bool                        = True,
    verbose:     Union[bool, str, int, None] = False,
) -> Union[mne.Epochs, dict]:
    """
    Load OpenBMI MI data for one or more subjects and sessions.
 
    Parameters
    ----------
    root_dir : str | Path
        Directory containing the .mat files.
    subjects : list[int] | None
        Subject IDs (1–54). None = all 54 subjects.
    sessions : list[int] | None
        Session numbers (1–2). None = both sessions.
    split : {'train', 'test'}
        Which partition to load. Default 'train'.
    tmin_ms : float
        Epoch start in ms relative to cue onset. Default -500 ms.
    tmax_ms : float
        Epoch end in ms. Default 4000 ms.
        Paper analysis window: 1000–3500 ms.
    l_freq : float | None
        High-pass cut-off. Paper uses 8.0 Hz. None = no filtering.
    h_freq : float | None
        Low-pass cut-off. Paper uses 30.0 Hz.
    notch_freq : float | None
        Notch filter frequency. None = no notch.
    concatenate : bool
        If True, return a single concatenated mne.Epochs.
        If False, return dict {(subject, session): mne.Epochs}.
    verbose : bool | str | int | None
 
    Returns
    -------
    mne.Epochs or dict {(subject, session): mne.Epochs}
 
    Examples
    --------
    >>> # Single subject, session 1, paper analysis window
    >>> epochs = load_openbmi_mi(
    ...     root_dir = "/data/OpenBMI",
    ...     subjects = [1],
    ...     sessions = [1],
    ...     split    = "train",
    ...     l_freq   = 8.0,
    ...     h_freq   = 30.0,
    ...     tmin_ms  = 1000,
    ...     tmax_ms  = 3500,
    ... )
 
    >>> # Multi-subject, no concatenation
    >>> epochs_dict = load_openbmi_mi(
    ...     root_dir    = "/data/OpenBMI",
    ...     subjects    = [1, 2, 3],
    ...     sessions    = [1, 2],
    ...     split       = "train",
    ...     l_freq      = 8.0,
    ...     h_freq      = 30.0,
    ...     concatenate = False,
    ... )
    >>> # Access by (subject, session)
    >>> epochs_s1_sess1 = epochs_dict[(1, 1)]
    >>> epochs_s1_sess1.metadata  # contains subject, session, split columns
    """
    if subjects is None:
        subjects = list(range(1, 55))
    if sessions is None:
        sessions = [1, 2]
    
    reader = OpenBMIMIReader(
        root_dir   = root_dir,
        split      = split, 
        tmin_ms    = tmin_ms,
        tmax_ms    = tmax_ms,
        l_freq     = l_freq,
        h_freq     = h_freq,
        notch_freq = notch_freq,
        verbose    = verbose,
    )

    result = {}
    for subj in subjects:
        for sess in sessions:
            try:
                result[(subj, sess)] = reader.get_epochs(
                    subject=subj, session=sess
                )
            except FileNotFoundError as exc:
                warnings.warn(str(exc), RuntimeWarning)
    
    if not result:
        raise RuntimeError(
            "No epochs could be loaded — check root_dir and subject/session IDs."
        )
 
    if concatenate:
        return mne.concatenate_epochs(list(result.values()), verbose=verbose)
 
    return result