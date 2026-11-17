from pathlib import Path 
from typing import Union, Literal, Optional
from scipy.io import loadmat

import mne
import warnings

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------

SPLIT_TYPE = Literal["train", "validation", "test"]

# ------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------

def _mat_str(field) -> str:
    """safely convert a MATLAB char field to a Python str."""
    if isinstance(field, np.ndarray):
        field = np.squeeze(field)
        if field.dtype.kind in ("U", "S", "O"):
            return str(field.flat[0]).strip()
        return str(field).strip()
    
def _mat_strlist(field) -> list[str]:
    """Convert a MATLAB cell-of-chars to a Python list[str]."""
    field = np.squeeze(field)
    if field.ndim == 0:
        return [_mat_str(field)]
    return [str(np.squeeze(item) if isinstance(item, np.ndarray) else item).strip()
            for item in field.flat]
 

class BCI2020Track3Reader:
    """
    Reader for BCI Competition 2020 – Track #3 (Multi-class Imagined Speech).
 
    Converts the original ``.mat`` files into an :class:`mne.EpochsArray`
    with a standard 10-20 montage and properly labelled events.
 
    Parameters
    ----------
    root_dir : str | Path
        Root directory of the dataset, containing sub-folders
        ``Training Set/`` and ``Validation Set/``.
    subject : int
        Subject index, 1 to 15.
    split : {'train', 'validation', 'test'}
        Dataset partition to load.
    preload : bool
        Passed to :class:`mne.EpochsArray` — keeps data in RAM.
    verbose : bool | str | int | None
        MNE verbosity level.
 
    Examples
    --------
    >>> reader = BCI2020Track3Reader("/data/BCI2020_Track3", subject=1)
    >>> epochs = reader.get_epochs()
    >>> epochs.plot_psd()
    """

    # Constants -----------------------------------------------------
    _SPLIT_FOLDER: dict[str, str] = {
        "train": "Training set",
        "validation": "Validation set",
        "test": "Test set"
    }

    #: Ordered channel names following the 10-20 system (64 channels)
    _CHANNEL_NAMES: list[str] = [
        "Fp1", "Fp2", "F7",  "F3",  "Fz",   "F4",   "F8",   "FC5",  "FC1",  "FC2",
        "FC6", "T7",  "C3",  "Cz",  "C4",   "T8",   "TP9",  "CP5",  "CP1",  "CP2",
        "CP6", "TP10","P7",  "P3",  "Pz",   "P4",   "P8",   "PO9",  "O1",   "Oz",
        "O2",  "PO10","AF7", "AF3", "AF4",  "AF8",  "F5",   "F1",   "F2",   "F6",
        "FT9", "FT7", "FC3", "FC4", "FT8",  "FT10", "C5",   "C1",   "C2",   "C6",
        "TP7", "CP3", "CPz", "CP4", "TP8",  "P5",   "P1",   "P2",   "P6",   "PO7",
        "PO3", "POz", "PO4", "PO8",]

    #: Event codes used by MNE (``event_id`` dict)
    _EVENT_ID: dict[str, int] = {
        "hello":     1,
        "help_me":   2,
        "stop":      3,
        "thank_you": 4,
        "yes":       5,
    }
 

    def __init__(
        self, 
        root_dir: Union[str, Path], 
        subject:  int                         = 1, 
        split:    SPLIT_TYPE                  = 'train', 
        preload:  bool                        = True, 
        verbose:  Union[bool, str, int, None] = False
    ) -> None:
        self.root_dir = Path(root_dir)
        self.subject  = subject
        self.split    = split
        self.preload  = preload
        self.verbose  = verbose

        self._validate_inputs()
        self._mat_path: Path = self._resolve_mat_path()

        self._raw_mat: Optional[dict]           = None
        self._epochs: Optional[mne.EpochsArray] = None

    # ---- Public API ----------------------------------------------------------
    
    def get_epochs(self) -> mne.EpochsArray:
        """
        Build and return the :class:`mne.EpochsArray`.
 
        The returned object has:
 
        * shape ``(n_trials, 64, n_times)``
        * ``tmin = -0.5`` s (−500 ms)
        * standard 10-20 montage
        * ``event_id = {'hello':1, 'help_me':2, 'stop':3, 'thank_you':4, 'yes':5}``
        """
        if self._epochs is None:
            self._epochs = self._build_epochs()
        return self._epochs
    
    @property
    def info(self) -> dict:
        """ Return a summary dict of the loaded recording"""
        mat = self._load_mat()
        epo = mat[f'epo_{self.split}']
        x = np.squeeze(epo["x"][0, 0]) # (time, ch, trials)
        t = np.squeeze(epo["t"][0, 0]).ravel()
        return {
            "dataset":    "BCI Competition 2020 – Track #3",
            "subject":    self.subject,
            "split":      self.split,
            "n_trials":   int(x.shape[2]),
            "n_channels": int(x.shape[1]),
            "n_times":    int(x.shape[0]),
            "sfreq_hz":   float(np.squeeze(epo["fs"][0, 0])),
            "tmin_ms":    float(t[0]),
            "tmax_ms":    float(t[-1]),
            "classes":    _mat_strlist(epo["className"][0, 0]),
            "mat_file":   str(self._mat_path),
        }


    # ---- Private helpers -----------------------------------------------------
    def _validate_inputs(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")
        if self.split not in self._SPLIT_FOLDER:
            raise ValueError(f"Split must be one of {list(self._SPLIT_FOLDER)}, "
                             f"got '{self.split}'"
                             )
        if not (1 <= self.subject <= 15):
            raise ValueError(f"Subject must be between 1 and 15, got {self.subject}")
    
    def _resolve_mat_path(self) -> Path:
        folder   = self.root_dir / self._SPLIT_FOLDER[self.split] 
        filename = f"Data_Sample{self.subject:02d}.mat"
        path     = folder / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}\n"
                                    "Download it from https://osf.io/pq7vb/files/osfstorage"
                                    )
        return path 
    
    def _load_mat(self) -> dict:
        """Load .mat file (cached after first call). """
        if self._raw_mat is None:
            self._raw_mat = loadmat(
                str(self._mat_path),
                squeeze_me = False,
                struct_as_record = True
            )
        return self._raw_mat
    
    def _parse_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Return a 1-D int array of event codes (1–5) from ``epo.y``.
    
        Handles both:
        * 1-D integer vector  (already event codes)
        * 2-D one-hot matrix  (5 × n_trials)  OR  (n_trials × 5)
        """
        y = np.squeeze(y)
        if y.ndim == 1:
            return y.astype(int)
        if y.shape[0] == 5: # (5, n_trials)
            return np.argmax(y, axis=0).astype(int) + 1
        if y.shape[1] == 5: # (n_trials, 5)
            return np.argmax(y, axis=1).astype(int) + 1
        raise ValueError(f"Cannot parse labels from epo.y with shape {y.shape}")
    
    def _build_epochs(self) -> mne.EpochsArray:
        mat = self._load_mat()
        epo = mat[f'epo_{self.split}']
    

        # ---- 1. Data (time, channels, trials) -> (trials, channels, time) -----
        x    = np.squeeze(epo['x'][0, 0])               # (t, C, T)
        data = x.transpose(2, 1, 0).astype(np.float64)  # (T, C, t)

        # ---- 2. Labels --------------------------------------------------------
        labels = self._parse_labels(epo['y'][0,0]) # 1-D int, values 1-5

        # ---- 3. Time / sfreq --------------------------------------------------
        sfreq = float(np.squeeze(epo['fs'][0, 0]))
        t_ms = np.squeeze(epo['t'][0, 0]).ravel()
        tmin = float(t_ms[0]) / 1000.0 # ms -> s

        # ---- 4. Channel names -------------------------------------------------
        try:
            ch_names = _mat_strlist(epo["clab"][0, 0])
            if len(ch_names) != data.shape[1]:
                raise ValueError('length mismatch')
        except:
            warnings.warn(
                "Could not parse channel names from epo.clab; "
                "falling back to built-in CHANNEL_NAMES list.",
                RuntimeWarning,
            )
            ch_names = self._CHANNEL_NAMES[: data.shape[1]]
    
        # ---- 5. MNE Info -----------------------------------------------------
        mne_info = mne.create_info(
            ch_names = ch_names,
            sfreq=sfreq,
            ch_types=['eeg'] * len(ch_names)
        )
        mne_info.set_montage(
            "standard_1020", on_missing="warn", verbose=self.verbose
        )

        # ---- 6. Events Array (n trials x 3) ---------------------------------
        n = data.shape[0]
        events = np.column_stack([
            np.arange(n, dtype=int),    # sample onset (one per epoch)
            np.zeros(n, dtype=int),     # previous event (unused)
            labels,                     # event code
        ])

        # ---- 7. Metadata -----------------------------------------------------
        metadata = pd.DataFrame({
            "subject": self.subject,
        }, index=range(n))

        # ---- 7. Epochs Array ------------------------------------------------
        epochs = mne.EpochsArray(
            data=data,
            info=mne_info,
            events=events,
            tmin=tmin,
            event_id=self._EVENT_ID,
            metadata=metadata,
            verbose=self.verbose,
        )

        # Store metadata on the object
        epochs._bci2020_meta = {
            "dataset":  "BCI Competition 2020 Track#3",
            "subject":  self.subject,
            "split":    self.split,
            "mat_file": str(self._mat_path),
        }

        return epochs

# ------------------------------------------------------------------------
# Multi-subject loader
# ------------------------------------------------------------------------
def load_bci2020_track3(
    root_dir:    Union[str, Path], 
    subjects:    Optional[list[int]]         = None, 
    split:       SPLIT_TYPE                  = 'train', 
    concatenate: bool                        = True, 
    preload:     bool                        = True, 
    verbose:     Union[bool, str, int, None] = False
) -> Union[dict[int, mne.EpochsArray], mne.EpochsArray]:
    """
    Load one or more subjects from BCI Competition 2020 Track#3.
 
    Parameters
    ----------
    root_dir : str | Path
        Root directory of the dataset.
    subjects : list[int] | None
        Subject IDs (1–15). ``None`` loads all 15 subjects.
    split : {'train', 'validation', 'test'}
        Dataset partition.
    concatenate : bool
        If ``True``, merge all subjects into a single :class:`mne.EpochsArray`.
    preload : bool
        Whether to preload data into memory.
    verbose : bool | str | int | None
        MNE verbosity.
 
    Returns
    -------
    dict[int, EpochsArray] or EpochsArray
        A dict keyed by subject ID, or a single concatenated EpochsArray.
 
    Examples
    --------
    Load a single subject:
 
    >>> epochs_s1 = load_bci2020_track3("/data/BCI2020", subjects=[1])[1]
 
    Load subjects 1–5 and concatenate:
 
    >>> all_epochs = load_bci2020_track3(
    ...     "/data/BCI2020", subjects=range(1, 6), concatenate=True
    ... )
    """
    if subjects is None:
        subjects = list(range(1, 16))
    
    result: dict[int, mne.EpochsArray] = {}
    for subj in subjects:
        try:
            reader = BCI2020Track3Reader(root_dir=root_dir, subject=subj, split=split, preload=preload, verbose=verbose)
            result[subj] = reader.get_epochs()
        except FileNotFoundError as exc:
            warnings.warn(str(exc), RuntimeWarning)
    
    if not result:
        raise RuntimeError("No subjects could be loaded. Check root_dir and split.")

    if concatenate:
        return mne.concatenate_epochs(list(result.values()), verbose=verbose)
    
    return result