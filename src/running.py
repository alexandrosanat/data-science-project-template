import numpy as np
import torch.nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from typing import Any, Optional
import torch
from tqdm import tqdm

from src.metrics import Metric
from src.tracking import Stage, ExperimentTracker


class Runner:

    def __init__(self,
                 loader: DataLoader[Any],
                 model: torch.nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None):

        self.run_count = 0
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.accuracy_metric = Metric()
        self.compute_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []

        # Assume stage is based on the presence of the optimizer
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN

    @property
    def avg_accuracy(self) -> float:
        return self.accuracy_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):
        # Training Loop
        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            batch_accuracy = self._run_single(x, y)
            experiment.add_batch_metric('accuracy', batch_accuracy, self.run_count)

    def _run_single(self, x, y):
        self.run_count += 1
        batch_size = x.shape[0]
        prediction = self.model(x)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Metrics
        y_np = y.detach().numpy()
        y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)
        batch_accuracy: float = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(batch_accuracy, batch_size)

        if self.optimizer is not None:
            # Reverse-mode AutoDiff (backpropagation)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return batch_accuracy

    def reset(self):
        self.accuracy_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []
