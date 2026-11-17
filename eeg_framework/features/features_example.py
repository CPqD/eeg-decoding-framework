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
RawFeatures       - Return the raw data
FeaturePipeline   — Concatenates features from multiple extractors
 
"""

import mne
import numpy as np
from typing import Optional, Union
from .features import BaseFeature

# --------------------------------------------------------------------------------------------------
# 1. Raw Features
# --------------------------------------------------------------------------------------------------
class RawFeatures(BaseFeature):
    """
    Returns raw EEG data as features — no extraction.
    Useful for deep learning models that learn features automatically (e.g. EEGNet).

    Output shape: (n_epochs, n_channels × n_times) or (n_epochs, n_channels, n_times)
    """
    def __init__(
        self,
        picks:   Union[str, list] = "eeg",
        flatten: bool             = True,
    ) -> None:
        self.picks   = picks
        self.flatten = flatten

    def transform(self, epochs: mne.BaseEpochs) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.picks, str):
            ch_idx = mne.pick_types(epochs.info, **{self.picks: True})
        else:
            ch_idx = [epochs.ch_names.index(c) for c in self.picks]
            
        data = epochs.get_data()[:, ch_idx, :]  # (n_epochs, n_ch, n_times)

        if self.flatten:
            X = data.reshape(data.shape[0], -1) # (n_epochs, n_ch x n_times)
        else:
            X = data                            # (n_epochs, n_ch, n_times)

        y = self._get_y(epochs, mapping=getattr(self, '_label_mapping', None))
        print(f"[RawFeatures] shape: {X.shape}")
        return X, y

# -----------------------------------------------------------------------------------------
# 2. CSP - Common Spatial Patterns
# -----------------------------------------------------------------------------------------
class CSP(BaseFeature):
    """
    Common Spatial Patterns with One-vs-Rest multiclass extension.
 
    For each class, fits a binary CSP (class vs all others), extracts
    log-variance features, and concatenates across all classes.
 
    Note: CSP operates on the full trial — windowing is not supported
    as it would change the covariance structure.
 
    Parameters
    ----------
    n_components : int
        Number of CSP components per binary problem (default 6).
        Total features = n_components × n_classes.
    reg : float | str | None
        Regularization for covariance estimation.
    log : bool
        If True (default), use log-variance. If False, use variance.
    verbose : bool | str | int | None
 
    Attributes
    ----------
    csp_list_ : list[mne.decoding.CSP]
        Fitted CSP objects, one per class (OVR).
    classes_ : np.ndarray
        Unique class labels found during fit.
    n_features_ : int
        Total number of features after fit.
    """
    def __init__(
            self, 
            n_components: int                         = 6,
            reg:          Optional[Union[float, str]] = None,
            log:          bool                        = True,
            verbose:      Union[bool, str, int, None] = False,
    ) -> None:
        self.n_components = n_components
        self.reg          = reg
        self.log          = log
        self.verbose      = verbose

        self.csp_list_    = []
        self.classes_     = None
        self.n_features_  = 0

    def fit(self, epochs: mne.BaseEpochs, y: Optional[np.ndarray] = None) -> "CSP":
        super().fit(epochs, y)
        if y is None:
            y = self._get_y(epochs, mapping=getattr(self, '_label_mapping', None))
        
        self.classes_  = np.unique(y)
        self.csp_list_ = []

        for cls in self.classes_:
            y_bin = (y==cls).astype(int)
            csp = mne.decoding.CSP(
                n_components=self.n_components,
                reg         =self.reg,
                log         =self.log,
                norm_trace  =False,
            )
            csp.fit(epochs.get_data(), y_bin)
            self.csp_list_.append(csp)
        
        self.n_features_ = self.n_components * len(self.classes_)
        print(
            f"[CSP] fitted OVR for {len(self.classes_)} classes | "
            f"total features: {self.n_features_}"
        )
        return self
    
    def transform(self, epochs: mne.BaseEpochs) -> tuple[np.ndarray, np.ndarray]:
        if not self.csp_list_:
            raise RuntimeError("Call fit() before transform().")
 
        X_parts = [csp.transform(epochs.get_data()) for csp in self.csp_list_]
        X = np.concatenate(X_parts, axis=1)
        y = self._get_y(epochs, mapping=getattr(self, '_label_mapping', None))
        return X, y