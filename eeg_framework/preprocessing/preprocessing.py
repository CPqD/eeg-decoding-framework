"""
eeg_toolbox.preprocessing
=========================
Modular preprocessing pipeline for MNE Epochs, sklearn-style.

Usage
-----
from preprocessing import PreprocessingPipeline, CAR, Resample, ...

pipe = PreprocessingPipeline([
    ('car',         CAR()),
    ('resample',    Resample()),
    ...
])

epochs_clean = pipe.fit_transform(epochs)

# Inspect individual steps
pipe['...'].get_params()

All steps
---------
* Accept  mne.BaseEpochs, return mne.BaseEpochs  (copy, never in-place)
* Expose  fit(epochs) / transform(epochs) / fit_transform(epochs)
* Are     sklearn-compatible (get_params / set_params)
"""

import mne

# ------------------------------------------------------------------------
# Base Class
# ------------------------------------------------------------------------

class BaseStep:
    """
    Abstract base for all preprocessing steps.

    Subclasses must implement ``fit`` and ``transform``.
    ``fit_transform`` is provided for free.
    sklearn-style ``get_params`` / ``set_params`` are also provided.
    """

    def fit(self, epochs: mne.BaseEpochs) -> "BaseStep":
        return self

    def transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        raise NotImplementedError

    def fit_transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        return self.fit(epochs).transform(epochs)
    
    # ---- Sklearn compability --------------------------------------------
    def get_params(self) -> dict:
        import inspect 
        sig    = inspect.signature(self.__class__.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            params[name] = getattr(self, name, param.default)
        return params
    
    def set_params(self, **params) -> "BaseStep":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"{self.__class__.__name__} has no parameter '{k}'")
            setattr(self, k, v)
        return self
 
    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

# ---------------------------------------------------------------------------------------------
# PreprocessingPipeline
# ---------------------------------------------------------------------------------------------
class PreprocessingPipeline:
    """
    Sequential preprocessing pipeline for MNE Epochs.
 
    Chains multiple BaseStep objects, passing the output of each step
    as input to the next. Follows the sklearn fit/transform/fit_transform
    convention, adapted for MNE Epochs.
 
    Parameters
    ----------
    steps : list[tuple[str, BaseStep]]
        List of (name, step) pairs. Names must be unique.
 
    Examples
    --------
    >>> pipe = PreprocessingPipeline([
    ...     ('filter', Resample(sfreq=256)),
    ...     ('car',    CAR()),
    ... ])
    >>> epochs_clean = pipe.fit_transform(epochs_train)
    >>> epochs_valid_clean = pipe.transform(epochs_valid)
 
    Access individual steps:
 
    >>> pipe.get_params()         # all parameters
    >>> pipe.set_params(filter__h_freq=30)  # update a parameter
    """

    def __init__(self, steps: list[tuple[str, BaseStep]]) -> None:
        self._validate_steps(steps)
        self.steps = steps

    # ---- dict-like access by name -----------------------------------------------------------
    def __getitem__(self, name: str) -> BaseStep:
        for n, step in self.steps:
            if n == name:
                return step
        raise KeyError(f"Step '{name}' not found. Available: {self.named_steps}")

    @property
    def named_steps(self) -> dict[str, BaseStep]:
        return {n: s for n, s in self.steps}
    
    # ---- core API ---------------------------------------------------------------------------
    def fit(self, epochs: mne.BaseEpochs) -> "PreprocessingPipeline":
        """Fit all steps sequentially, passing transformed output forward."""
        current = epochs
        for name, step in self.steps:
            print(f"[Pipeline] fitting  → {name}")
            step.fit(current)
            current = step.transform(current)
        return self

    def transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        """Apply all already-fitted steps."""
        current = epochs
        for name, step in self.steps:
            print(f"[Pipeline] applying → {name}")
            current = step.transform(current)
        return current
    
    def fit_transform(self, epochs: mne.BaseEpochs) -> mne.BaseEpochs:
        """Fit each step on the output of the previous, return final result."""
        current = epochs
        for name, step in self.steps:
            print(f"[Pipeline] fit_transform → {name}")
            current = step.fit_transform(current)
        return current
    
    # ---- sklearn-style param access ---------------------------------------------------------
    def get_params(self) -> dict:
        """Return all params as {step_name__param: value}."""
        out = {}
        for name, step in self.steps:
            for k, v in step.get_params().items():
                out[f"{name}__{k}"] = v
        return out
    
    def set_params(self, **params) -> "PreprocessingPipeline":
        """Set params using 'step_name__param_name' notation."""
        for key, value in params.items():
            if "__" not in key:
                raise ValueError(
                    f"Param key must be 'step__param', got '{key}'"
                )
            step_name, param = key.split("__", 1)
            self[step_name].set_params(**{param: value})
        return self
    
    # ---- inspection --------------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = ["Pipeline("]
        for name, step in self.steps:
            lines.append(f"  ('{name}', {step!r})")
        lines.append(")")
        return "\n".join(lines)
    
    # ---- validation ---------------------------------------------------------------------------
    def _validate_steps(self, steps: list) -> None:
        if not steps:
            raise ValueError("Pipeline must have at least one step.")
        names = [n for n, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique.")
        for name, step in steps:
            if not isinstance(step, BaseStep):
                raise TypeError(
                    f"Step '{name}' must be a BaseStep subclass, "
                    f"got {type(step).__name__}."
                )