# EEG Framework

A modular Python framework for EEG-based Brain-Computer Interface (BCI) research, with a focus on imagined speech classification. Built on top of [MNE-Python](https://mne.tools/) and scikit-learn conventions.

---

## Overview

The framework provides end-to-end support for EEG classification pipelines:

- **Dataset readers**
- **Preprocessing**
- **Feature extraction**
- **Deep learning models**
- **Evaluation**
- **Plotting**
- **Utilities**

---

## Framework Structure

```
eeg_framework/
├── datasets/
├── preprocessing/
├── features/
├── models/
├── evaluate/
├── plotting/
└── utils/
```

---

## Datasets

### OpenBMI MI

54 subjects, 62 EEG channels, ~1000 Hz.  
2 classes: left, right.

```python
from eeg_framework.datasets import load_openbmi_mi

epochs_train = load_openbmi_mi(root_dir="data/OpenBMI MI", split="train", subjects=[1])
epochs_test  = load_openbmi_mi(root_dir="data/OpenBMI MI", split="test",  subjects=[1])
```

Download: [OpenBMI](https://gigadb.org/dataset/100542)

---

## Preprocessing

All steps follow the `fit` / `transform` / `fit_transform` convention.  
Each step accepts `mne.BaseEpochs` and returns `mne.BaseEpochs` (copy — never in-place).

```python
from eeg_framework.preprocessing import *

pipe = Pipeline([
    ('car',        CAR()),
    ('resample',   Resample(256.)),
    # ...
])

epochs_train_clean = pipe.fit_transform(epochs_train)
epochs_valid_clean = pipe.transform(epochs_valid)
```

---

## Feature Extraction

All extractors return `(X, y)` where `X` is `(n_epochs, n_features)` and `y` is `(n_epochs,)`.  
The label mapping is captured during `fit` and reused during `transform` — consistent even when validation epochs are missing a class.

```python
from eeg_framework.features import *

pipe_feat = FeaturePipeline([
    ('RAW', RawFeatures(flatten=False)),
    # ...
])
X_train, y_train = pipe_feat.fit_transform(epochs_train)
X_test,  y_test  = pipe_feat.transform(epochs_test)
```

---

## Models

### EEGNet

```python
from eeg_framework.models import *

params = eegnet_params_from_epochs(epochs_train)
model  = build_eegnet(
    n_classes          = len(epochs_train.event_id),
    compile_model      = True,
    include_classifier = True,
    **params
)

model.fit(
    X_train, y_train,
    validation_split = 0.2,
    epochs           = 150,
)
```

---

## Evaluation

```python
from eeg_framework.evaluate import *

# Evaluation Metrics
metrics = compute_metrics(y_valid, y_pred, class_names=list(epochs_train.event_id))
# ...

# Evaluation Plots
# ...
```

---

## Plotting

```python
from eeg_framework.plotting import *

plot_erp_compare(epochs) 
# ...
```

---

## Utilities

```python
from eeg_framework.utils *

# Utils...
```

---

## Dependencies

```
python 3.12.3
mne >= 1.0
numpy
scipy
scikit-learn
matplotlib
pandas
tensorflow >= 2.10   # for EEGNet
```
