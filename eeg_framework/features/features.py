"""
eeg_toolbox.features
====================
Feature extraction pipeline for MNE Epochs, sklearn-style.
 
Different from preprocessing, feature extractors:
* Receive  mne.Epochs
* Return   np.ndarray (X) + np.ndarray (y)
* Are      composable via FeaturePipeline
 
Classes
-------
BaseFeature       — abstract base
FeaturePipeline   — Concatenates features from multiple extractors
 
"""

import mne
import warnings
import numpy as np
from typing import Optional, Union

# ------------------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------------------

class BaseFeature:
    """
    Abstract base for all feature extractors.
 
    fit(epochs, y)        — learn parameters from labelled training data
    transform(epochs)     — extract features → (X, y)
    fit_transform(e, y)   — fit then transform
 
    All extractors follow the same contract:
        X shape: (n_epochs, n_features)
        y shape: (n_epochs,)  integer labels
    """
    def fit(self, epochs: mne.BaseEpochs, y: Optional[np.ndarray] = None) -> "BaseFeature":
        sorted_codes = sorted(epochs.event_id.values())
        self._label_mapping = {code: idx for idx, code in enumerate(sorted_codes)}
        return self

    def transform(self, epochs: mne.BaseEpochs) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit_transform(self, epochs:mne.BaseEpochs, y: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(epochs, y).transform(epochs)
    
    # ---- sklearn compatibility -----------------------------------------
    def get_params(self) -> dict:
        import inspect
        sig    = inspect.signature(self.__class__.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            params[name] = getattr(self, name, param.default)
        return params
 
    def set_params(self, **params) -> "BaseFeature":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"{self.__class__.__name__} has no parameter '{k}'")
            setattr(self, k, v)
        return self
 
    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"
 
    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _get_y(epochs: mne.BaseEpochs, mapping=None) -> np.ndarray:
        """Extract integer labels from epochs.events."""
        if mapping is None:
            sorted_codes = sorted(epochs.event_id.values())
            mapping = {code: idx for idx, code in enumerate(sorted_codes)}
        labels = epochs.events[:, 2].astype(int)
        return np.array([mapping[l] for l in labels])
    
    @staticmethod
    def _get_windows(
        n_times:     int,
        sfreq:       float,
        window_size: Optional[float],
        stride:      Optional[float],
    ) -> list[tuple[int, int]]:
        """
        Compute (start, end) sample indices for sliding windows.
 
        Parameters
        ----------
        n_times : int
            Total number of time samples.
        sfreq : float
            Sampling frequency in Hz.
        window_size : float | None
            Window duration in seconds. None = full trial.
        stride : float | None
            Step between windows in seconds.
            None = same as window_size (no overlap).
 
        Returns
        -------
        list of (start, end) tuples in samples.
        """
        if window_size is None:
            return [(0, n_times)]
 
        win_samples    = int(window_size * sfreq)
        stride_samples = int((stride if stride is not None else window_size) * sfreq)
 
        windows = []
        start   = 0
        while start + win_samples <= n_times:
            windows.append((start, start + win_samples))
            start += stride_samples
 
        if not windows:
            warnings.warn(
                f"window_size ({window_size}s) larger than signal duration "
                f"({n_times/sfreq:.2f}s). Using full trial.",
                RuntimeWarning,
            )
            return [(0, n_times)]
 
        return windows

# --------------------------------------------------------------------------------------------------
# Feature Pipeline
# --------------------------------------------------------------------------------------------------
class FeaturePipeline:
    """
    Concatenates features from multiple extractors into a single (X, y).
 
    Each step extracts features independently from the same epochs,
    and results are horizontally concatenated.
 
    Parameters
    ----------
    steps : list[tuple[str, BaseFeature]]
 
    Examples
    --------
    >>> pipe = FeaturePipeline([
    ...     ('raw',     RawFeatures()),
    ... ])
    >>> X_train, y_train = pipe.fit_transform(epochs_train)
    >>> X_test,  y_test  = pipe.transform(epochs_test)
    """
    def __init__(self, steps: list[tuple[str, BaseFeature]]) -> None:
        self._validate_steps(steps)
        self.steps = steps

    # ---- dict-like acess --------------------------------------------------------
    def __getitem__(self, name: str) -> BaseFeature:
        for n, step in self.steps:
            if n == name:
                return step
        raise KeyError(f"Step '{name}' not found. Available: {list(self.named_steps)}")
 
    @property
    def named_steps(self) -> dict[str, BaseFeature]:
        return {n: s for n, s in self.steps}
    
    # ---- core API ---------------------------------------------------------------
    def fit(
        self, epochs: mne.BaseEpochs, y: Optional[np.ndarray] = None
    ) -> "FeaturePipeline":
        if y is None:
            y = BaseFeature._get_y(epochs)
        for name, step in self.steps:
            print(f"[FeaturePipeline] fitting → {name}")
            step.fit(epochs, y)
        return self
 
    def transform(self, epochs: mne.BaseEpochs) -> tuple[np.ndarray, np.ndarray]:
        X_parts = []
        y_ref   = None
        for name, step in self.steps:
            print(f"[FeaturePipeline] extracting → {name}")
            X_i, y_i = step.transform(epochs)
            if y_ref is None:
                y_ref = y_i
            X_parts.append(X_i)
        return np.concatenate(X_parts, axis=1), y_ref
 
    def fit_transform(
        self, epochs: mne.BaseEpochs, y: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(epochs, y)
        return self.transform(epochs)

    # ---- inspection -------------------------------------------------------------
    def __repr__(self) -> str:
        lines = ["FeaturePipeline("]
        for name, step in self.steps:
            lines.append(f"  ('{name}', {step!r})")
        lines.append(")")
        return "\n".join(lines)

    # ---- validation -------------------------------------------------------------
    def _validate_steps(self, steps: list) -> None:
        if not steps:
            raise ValueError("FeaturePipeline must have at least one step.")
        names = [n for n, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique.")
        for name, step in steps:
            if not isinstance(step, BaseFeature):
                raise TypeError(
                    f"Step '{name}' must be a BaseFeature subclass, "
                    f"got {type(step).__name__}."
                )