"""
eeg_toolbox.models.eegnet
=========================
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
 
Reference:
    Lawhern et al. (2018) — EEGNet: a compact convolutional neural network
    for EEG-based brain–computer interfaces.
    Journal of Neural Engineering, 15(5), 056013.
    https://doi.org/10.1088/1741-2552/aace8c
 
Architecture:
    Input      → (n_channels, n_times, 1)
    Block 1    → Temporal conv + Depthwise spatial conv + BN + ELU + AvgPool + Dropout
    Block 2    → Separable conv + BN + ELU + AvgPool + Dropout
    Classifier → Flatten + Dense(softmax)
"""
import mne
from typing import Optional

def build_eegnet(
    n_classes:          Optional[int]     = None,
    n_channels:         int               = 64,
    n_times:            int               = 795,
    temporal_filters:   int               = 8,
    depth_multiplier:   int               = 2,
    separable_filters:  int               = 16,
    temporal_kernel:    int               = 64,
    separable_kernel:   int               = 16,
    pool1:              int               = 4,
    pool2:              int               = 8,
    dropout:            float             = 0.5,
    max_norm_depthwise: float             = 1.0,
    max_norm_dense:     float             = 0.25,
    bn_momentum:        float             = 0.99,
    compile_model:      bool              = False,
    include_classifier: bool              = True,
    optimizer:          str               = "adam",
    loss:               str               = "sparse_categorical_crossentropy",
) -> "tf.keras.Model":
    """
    Build an EEGNet model.
 
    Parameters
    ----------
    n_classes : int
        Number of output classes.
    n_channels : int
        Number of EEG channels (default 64).
    n_times : int
        Number of time samples per epoch (default 795).
    temporal_filters : int
        Number of temporal filters in Block 1 (default 8).
        Paper recommends F1=8 for most BCIs.
    depth_multiplier : int
        Depth multiplier for depthwise conv — controls spatial filters.
        Total spatial filters = temporal_filters × depth_multiplier (default 2 → 16).
    separable_filters : int
        Number of separable filters in Block 2 (default 16).
        Paper recommends F2 = temporal_filters × depth_multiplier.
    temporal_kernel : int
        Temporal kernel size in samples (default 64).
        Paper recommends sfreq / 2 (e.g. 64 for 128 Hz, 128 for 256 Hz).
    separable_kernel : int
        Separable conv kernel size (default 16).
        Paper recommends sfreq / 8.
    pool1 : int
        First average pooling size (default 4).
    pool2 : int
        Second average pooling size (default 8).
    dropout : float
        Dropout rate (default 0.5).
    max_norm_depthwise : float
        MaxNorm constraint on depthwise conv (default 1.0).
    max_norm_dense : float
        MaxNorm constraint on Dense layer (default 0.25).
    bn_momentum : float
        BatchNormalization momentum. Default 0.99.
        Use 0.1 for small datasets — default 0.99 prevents convergence
        with few batches.
    compile_model : bool
        If True (default), compile with optimizer and loss.
        If False, return uncompiled model.
    optimizer : str
        Optimizer name (default 'adam').
    loss : str
        Loss function (default 'sparse_categorical_crossentropy').
 
    Returns
    -------
    tf.keras.Model
        EEGNet model, compiled if compile_model=True.
 
    Examples
    --------
    >>> model = build_eegnet(n_classes=5, n_channels=64, n_times=512)
    >>> model.summary()
 
    Extracting params from MNE epochs:
 
    >>> n_channels = len(mne.pick_types(epochs.info, eeg=True))
    >>> n_times    = epochs.get_data().shape[2]
    >>> n_classes  = len(epochs.event_id)
    >>> model = build_eegnet(n_classes, n_channels, n_times)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, constraints
    except ImportError:
        raise ImportError(
            "TensorFlow not installed.\n"
            "Install with: pip install tensorflow"
        )
    
    if include_classifier and n_classes is None:
        raise ValueError("n_classes must be provided when include_classifier=True")

    inputs = tf.keras.Input(shape=(n_channels, n_times))
    x = layers.Reshape((n_channels, n_times, 1))(inputs)

    # ---- Block 1: Temporal + Deptwise Spatial --------------------------
    x = layers.Conv2D(
        filters     = temporal_filters,
        kernel_size = (1, temporal_kernel),
        padding     = "same",
        use_bias    = False,
        name        = "temporal_conv"
    )(x)
    x = layers.BatchNormalization(name='bn_temporal', momentum=bn_momentum)(x)

    x = layers.DepthwiseConv2D(
        kernel_size          = (n_channels, 1),
        use_bias             = False,
        depth_multiplier     = depth_multiplier,
        depthwise_constraint = constraints.MaxNorm(max_norm_depthwise),
        name                 = "depthwise_conv",
    )(x)
    x = layers.BatchNormalization(name="bn_depthwise", momentum=bn_momentum)(x)
    x = layers.Activation("elu", name="elu_1")(x)
    x = layers.AveragePooling2D(pool_size=(1, pool1), name="pool_1")(x)
    x = layers.Dropout(dropout, name="drop_1")(x)

    # ---- Block 2: Separable Conv --------------------------------------
    x = layers.SeparableConv2D(
        filters     = separable_filters,
        kernel_size = (1, separable_kernel),
        use_bias    = False,
        padding     = "same",
        name        = "separable_conv",
    )(x)
    x = layers.BatchNormalization(name="bn_separable", momentum=bn_momentum)(x)
    x = layers.Activation("elu", name="elu_2")(x)
    x = layers.AveragePooling2D(pool_size=(1, pool2), name="pool_2")(x)
    x = layers.Dropout(dropout, name="drop_2")(x)

    # ---- Classifier --------------------------------------------------
    if include_classifier:
        x = layers.Flatten(name="flatten")(x)
        outputs = layers.Dense(
            units             = n_classes,
            activation        = "softmax",
            kernel_constraint = constraints.MaxNorm(max_norm_dense),
            name              = "classifier"
        )(x)
    else:
        outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EEGNet")

    if compile_model:
        model.compile(
            optimizer = optimizer,
            loss      = loss,
            metrics   = ["accuracy"]
        )
    
    return model

def eegnet_params_from_epochs(epochs: mne.BaseEpochs) -> dict:
    """
    Extract EEGNet-compatible parameters directly from an MNE Epochs object.
 
    Returns a dict ready to unpack into build_eegnet():
        build_eegnet(**eegnet_params_from_epochs(epochs))
 
    Parameters
    ----------
    epochs : mne.BaseEpochs
 
    Returns
    -------
    dict with keys: n_channels, n_times, temporal_kernel, separable_kernel
    """

    n_channels      = len(mne.pick_types(epochs.info, eeg=True))
    n_times         = epochs.get_data().shape[2]
    sfreq           = epochs.info['sfreq']
    temporal_kernel = int(sfreq / 2) # original paper: sfreq / 2
    separable_kernel  = int(sfreq / 8)  # original paper: sfreq / 8

    return {
        "n_channels" :      n_channels,
        "n_times":          n_times,
        "temporal_kernel" : temporal_kernel,
        "separable_kernel": separable_kernel
    }