import collections
import functools
import os
import time
from typing import Callable, Dict, Optional, Type

import torch.nn
import torch.optim
import torch.utils.data
import tqdm
from PySide6.QtCore import QThread, Signal
from typing_extensions import Literal

from ..common import PytorchProfileRecord


def fixJson(in_: str, out: str):
    o = open(out, 'w')
    with open(in_, 'r') as f:
        for line in f:
            o.write(line.replace('\\', '\\\\'))
    o.close()
    os.remove(in_)


class QTrainingWorker(QThread):

    ended = Signal(name='ended')
    scalars = Signal(str, float, bool, name='scalar')
    epochStart = Signal(int, float, name='epochStart')
    epochEnd = Signal(int, float, name='epochEnd')
    pythonProf = Signal(dict, float, name='profile')
    pytorchProf = Signal(list, dict, name='pytorchProfile')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch: Optional[int] = None
        self._model: Optional[torch.nn.Module] = None
        self._optim: Optional[torch.optim.Optimizer] = None
        self._closure: Optional[Callable[..., float]] = None
        self._scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self._dataloader: Optional[torch.utils.data.DataLoader] = None
        self._thread: Optional[QThread] = None
        self._currentEpoch = 0
        self._pythonProfile: Optional[Literal['c', 'py']] = None
        self._pytorchProfileArgs: Optional[Dict[str, bool]] = None
        self._device = torch.device('cpu')
        self._pytorchChromePath: Optional[str] = None

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

    @property
    def device(self):
        return self._device

    @property
    def pythonProfile(self):
        return self._pythonProfile

    @property
    def pytorchProfileArgs(self):
        if self._pytorchProfileArgs is None:
            return {'enabled': False}
        return self._pytorchProfileArgs

    @property
    def pytorchChromePath(self):
        return self._pytorchChromePath

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

    def setDevice(self, dev: torch.device):
        self.model.to(dev)
        self._device = dev

    def setPythonProfile(self, profile: Optional[Literal['c', 'py']] = None):
        self._pythonProfile = profile

    def setPytorchProfile(
        self, enabled: bool, cpu: bool, gpu: bool, mem: bool
    ):
        self._pytorchProfileArgs = {
            'enabled': enabled, 'use_cuda': gpu, 'use_cpu': cpu,
            'profile_memory': mem, 'use_kineto': not cpu,
            'with_stack': True, 'with_modules': True,
        }

    def setPytorchChromePath(self, path: Optional[str] = None):
        self._pytorchChromePath = path

    def performEpoch(self, start: int, index: int):
        self._currentEpoch = index
        epochStartTime = time.time()
        self.epochStart.emit(index - start, epochStartTime)
        scalars = collections.defaultdict(lambda: 0.0)
        for data in self.dataloader:
            kwargs = {
                'model': self.model,
                'optim': self.optimizer,
                'device': self.device,
                'data': data,
                'scalars': scalars
            }
            self.optimizer.step(functools.partial(
                self.closure, **kwargs
            ))
        for key, value in scalars.items():
            self.sendScalar(key, value, True)
        epochEndTime = time.time()
        self.sendScalar('_epoch', index, True)
        self.sendScalar('_time', epochEndTime - epochStartTime)
        self.epochEnd.emit(index - start, epochEndTime)
        if self.scheduler is not None:
            self.scheduler.step()

    def run(self):
        if self.pythonProfile is not None:
            pstats = __import__('pstats')
        else:
            pstats = None
        if self.pythonProfile == 'c':
            profile = __import__('cProfile')
        elif self.pythonProfile == 'py':
            profile = __import__('profile')
        else:
            profile = None

        def pytorchProfilerWrapper(func, *func_args):
            with torch.autograd.profiler.profile(**self.pytorchProfileArgs) as prof:
                func(*func_args)
            return prof

        try:
            start = self._currentEpoch + 1
            self.model.train()
            for _ in range(start, start + self.epoch):
                if profile is not None and pstats is not None:
                    pr = profile.Profile()
                    result = pr.runcall(
                        pytorchProfilerWrapper,
                        self.performEpoch, start, _
                    )
                    p = pstats.Stats(pr).get_stats_profile()
                    self.pythonProf.emit(p.func_profiles, p.total_tt)
                else:
                    result = pytorchProfilerWrapper(
                        self.performEpoch, start, _
                    )
                if result is not None:
                    if self.pytorchChromePath is not None:
                        temp = os.path.join(self.pytorchChromePath, 'temp')
                        newPath = os.path.join(
                            self.pytorchChromePath, f'epoch_{_}.json'
                        )
                        result.export_chrome_trace(temp)
                        fixJson(temp, newPath)
                    keyAvgs = [
                        PytorchProfileRecord.fromProfile(result, _).toDict()
                        for _ in result.key_averages()
                    ]
                    totalAvg = PytorchProfileRecord.fromProfile(
                        result, result.total_average()
                    ).toDict()
                    self.pytorchProf.emit(keyAvgs, totalAvg)
        finally:
            self.ended.emit()