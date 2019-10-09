from typing import List
from collections import defaultdict

import numpy as np
import torch

from steel.custom.precisionrecallfscore import PrecisionRecallFScoreMeter
from catalyst.dl.core import Callback, RunnerState, CallbackOrder

class PrecisionRecallFScoreCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        class_names: List[str] = None,
        num_classes: int = 1,
        threshold: float = 0.5
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key

        self.list_args = ["precision", "recall", "fscore"]
        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        assert self.num_classes is not None

        self.meters = [PrecisionRecallFScoreMeter(threshold) for _ in range(self.num_classes)]

    def _reset_stats(self):
        for meter in self.meters:
            meter.reset()

    def on_loader_start(self, state):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        logits: torch.Tensor = state.output[self.output_key].detach().float()
        targets: torch.Tensor = state.input[self.input_key].detach().float()
        probabilities: torch.Tensor = torch.sigmoid(logits)

        if self.num_classes == 1 and len(probabilities.shape) == 1:
            self.meters[0].add(probabilities, targets)
        else:
            for i in range(self.num_classes):
                self.meters[i].add(probabilities[:, i], targets[:, i])

    def on_loader_end(self, state: RunnerState):
        prec_recall_fscore = defaultdict(list)
        for i, meter in enumerate(self.meters):
            metrics = meter.value()
            postfix = self.class_names[i] \
                if self.class_names is not None \
                else str(i)
            for prefix, metric_ in zip(self.list_args, metrics):
                prec_recall_fscore[prefix] = metric_
                metric_name = f"{prefix}/class_{postfix}"
                state.metrics.epoch_values[state.loader_name][metric_name] = metric_

        for prefix in self.list_args:
            mean_value = float(np.mean(prec_recall_fscore[prefix]))
            metric_name = f"{prefix}/_mean"
            state.metrics.epoch_values[state.loader_name][metric_name] = mean_value

        self._reset_stats()


__all__ = ["PrecisionRecallFScoreCallback"]
