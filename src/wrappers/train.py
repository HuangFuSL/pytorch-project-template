import collections
import functools
import time
from typing import Callable, Optional, Type

import torch.nn
import torch.optim
import torch.utils.data
from PySide6.QtCore import QThread, Signal


class QTrainingWorker(QThread):

    ended = Signal(name='ended')
    scalars = Signal(str, float, bool, name='scalar')
    epochStart = Signal(int, float, name='epochStart')
    epochEnd = Signal(int, float, name='epochEnd')

    def __init__(self):
        super().__init__()
        self._epoch: Optional[int] = None
        self._model: Optional[torch.nn.Module] = None
        self._optim: Optional[torch.optim.Optimizer] = None
        self._closure: Optional[Callable[..., float]] = None
        self._scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self._dataloader: Optional[torch.utils.data.DataLoader] = None
        self._thread: Optional[QThread] = None
        self._currentEpoch = 0

    def __len__(self):
        return self.epoch

    def sendScalar(self, name: str, value: float, silent: bool = False):
        self.scalars.emit(name, value, silent)

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError('Model not set')
        return self._model

    @property
    def epoch(self):
        if self._epoch is None:
            raise RuntimeError('Epoch not set')
        return self._epoch

    @property
    def optimizer(self):
        if self._optim is None:
            raise RuntimeError('Optimizer not set')
        return self._optim

    @property
    def closure(self):
        if self._closure is None:
            raise RuntimeError('Optimizing closure not set')
        return self._closure

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def dataloader(self):
        if self._dataloader is None:
            raise RuntimeError('Dataloader not set')
        return self._dataloader

    def setModel(self, model: torch.nn.Module):
        if hash(model) != hash(self._model):
            self._model = model
            self._optim = None
            self._scheduler = None

    def setOptimizer(self, optim: Type[torch.optim.Optimizer], param_src, **kwargs):
        if hash(optim) != hash(self._optim):
            self._optim = optim(param_src, **kwargs)
            self._scheduler = None

    def setClosure(self, closure):
        self._closure = closure

    def setEpoch(self, n: int):
        self._epoch = n

    def resetEpoch(self):
        self._currentEpoch = 0

    def setScheduler(self, scheduler: Type[torch.optim.lr_scheduler._LRScheduler], **kwargs):
        if hash(scheduler) != hash(self._scheduler):
            self._scheduler = scheduler(self.optimizer, **kwargs)

    def setDataloader(self, dataloader: torch.utils.data.DataLoader):
        self._dataloader = dataloader

    def run(self):
        try:
            start = self._currentEpoch + 1
            for _ in range(start, start + self.epoch):
                self._currentEpoch = _
                epochStartTime = time.time()
                self.epochStart.emit(_ - start, epochStartTime)
                scalars = collections.defaultdict(lambda: 0.0)
                for data in self.dataloader:
                    kwargs = {
                        'model': self.model,
                        'optim': self.optimizer,
                        'data': data,
                        'scalars': scalars
                    }
                    self.optimizer.step(functools.partial(
                        self.closure, **kwargs
                    ))
                for i, (key, value) in enumerate(scalars.items(), 1):
                    self.sendScalar(key, value, True)
                epochEndTime = time.time()
                self.sendScalar('_epoch', _, True)
                self.sendScalar('_time', epochEndTime - epochStartTime)
                self.epochEnd.emit(_ - start, epochEndTime)
                if self.scheduler is not None:
                    self.scheduler.step()
        finally:
            self.ended.emit()
