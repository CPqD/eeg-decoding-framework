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