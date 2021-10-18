from enum import Enum, auto
from typing import Union, Protocol
import numpy as np


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class ExperimentTracker(Protocol):

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging an epoch-level metric."""

    def add_epoch_confusion_matrix(self, y_true: np.array, y_pred: np.array, step: int):
        """Implements logging a confusion matrix at epoch-level."""

    def add_hparams(self, hparams: dict[str, Union[str, float]], metrics: dict[str, float]):
        """Implements logging hyperparameters."""

    def set_stage(self, stage: Stage):
        """Sets the stage of the experiment."""

    def flush(self):
        """Flushes the experiment."""

