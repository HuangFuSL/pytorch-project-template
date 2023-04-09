import functools
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mkldnn
import torch.backends.mps
import torch.backends.openmp

try:
    import torch.backends.opt_einsum
    opteinsum = True
except ImportError:
    opteinsum = False
import torch.cuda


def saveModelDict(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)


def loadModelDict(model: torch.nn.Module, path: str):
    model.load_state_dict(torch.load(path))


def setSeed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resetSeed():
    torch.manual_seed(int(time.time()))
    torch.cuda.manual_seed(int(time.time()))
    torch.cuda.manual_seed_all(int(time.time()))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setCudnnEnabled(enabled: bool):
    torch.backends.cudnn.enabled = enabled


def setOpteinsumEnabled(enabled: bool):
    if opteinsum:
        torch.backends.opt_einsum.enabled = enabled


def setCudnnBenchmark(enabled: bool):
    torch.backends.cudnn.benchmark = enabled


def setCudnnDeterministic(enabled: bool):
    torch.backends.cudnn.deterministic = enabled


def setCudnnAllowTF32(enabled: bool):
    torch.backends.cudnn.allow_tf32 = enabled


def setCudaMatmulAllowTF32(enabled: bool):
    torch.backends.cuda.matmul.allow_tf32 = enabled


def setCudaMatmulAllowFP16(enabled: bool):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = enabled


def setCudaMatmulAllowBF16(enabled: bool):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = enabled


class BackendAttribute():
    def __init__(
        self, enabled: bool, value: Optional[Callable[[], bool]],
        setter: Optional[Callable[[bool], None]] = None
    ):
        self._enabled = enabled
        if self._enabled:
            self._value = value
            self.setter = setter
        else:
            self._value, self.setter = None, None

    @property
    def value(self):
        if self._enabled and self._value is not None:
            return self._enabled and self._value()
        return False

    @value.setter
    def value(self, value: bool):
        if self._enabled and self.setter is not None:
            self.setter(value)


def detectBackends():
    attributes: Dict[str, BackendAttribute] = {}
    disabled = [False, None, None]
    attributes['cuda'] = BackendAttribute(True, torch.cuda.is_available)
    attributes['mps'] = BackendAttribute(True, torch.backends.mps.is_available)
    attributes['mkl'] = BackendAttribute(True, torch.backends.mkl.is_available)
    attributes['mkldnn'] = BackendAttribute(
        True, torch.backends.mkldnn.is_available
    )
    attributes['openmp'] = BackendAttribute(
        True, torch.backends.openmp.is_available
    )
    attributes['cudnn'] = BackendAttribute(
        torch.cuda.is_available(),
        torch.backends.cudnn.is_available, setCudnnEnabled
    )
    if opteinsum:
        attributes['opteinsum'] = BackendAttribute(
            True, torch.backends.opt_einsum.is_available, setOpteinsumEnabled
        )
    else:
        attributes['opteinsum'] = BackendAttribute(*disabled)
    if torch.cuda.is_available():
        attributes['cuda_matmul_allow_tf32'] = BackendAttribute(
            True, torch.backends.cuda.matmul.allow_tf32, setCudaMatmulAllowTF32
        )
        attributes['cuda_matmul_allow_fp16'] = BackendAttribute(
            True,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            setCudaMatmulAllowFP16
        )
        attributes['cuda_matmul_allow_bf16'] = BackendAttribute(
            True,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            setCudaMatmulAllowBF16
        )
    else:
        attributes['cuda_matmul_allow_tf32'] = BackendAttribute(*disabled)
        attributes['cuda_matmul_allow_fp16'] = BackendAttribute(*disabled)
        attributes['cuda_matmul_allow_bf16'] = BackendAttribute(*disabled)
    if torch.backends.cudnn.is_available():
        attributes['cudnn_benchmark'] = BackendAttribute(
            True, lambda: torch.backends.cudnn.benchmark, setCudnnBenchmark
        )
        attributes['cudnn_deterministic'] = BackendAttribute(
            True, lambda: torch.backends.cudnn.deterministic,
            setCudnnDeterministic
        )
        attributes['cudnn_allow_tf32'] = BackendAttribute(
            True, lambda: torch.backends.cudnn.allow_tf32, setCudnnAllowTF32
        )
    else:
        attributes['cudnn_benchmark'] = BackendAttribute(*disabled)
        attributes['cudnn_deterministic'] = BackendAttribute(*disabled)
        attributes['cudnn_allow_tf32'] = BackendAttribute(*disabled)
    return attributes


def setCurrentDevice(device: int):
    torch.cuda.set_device(device)


def getDevice(device: Optional[int]):
    if device is None:
        return torch.device('cpu')
    return torch.device(f'cuda:{device}')


@functools.lru_cache()
def getCudaVersion() -> str:
    if torch.version.cuda is None:
        return 'Not available'
    return torch.version.cuda


@functools.lru_cache()
def getCudaDevices() -> Dict[int, Dict[str, Any]]:
    cudaPropertiesName = [
        'name', 'major', 'minor', 'total_memory', 'multi_processor_count'
    ]
    n = torch.cuda.device_count()
    ret: Dict[int, Dict[str, Any]] = {}
    for _ in range(n):
        properties = torch.cuda.get_device_properties(_)
        for item in cudaPropertiesName:
            ret.setdefault(_, {})[item] = properties.__getattribute__(item)
        ret[_]['total_memory'] /= 1024 ** 2
    return ret


def getGpuMemory():
    if not torch.cuda.is_available():
        return [0, 0, 0, 0]
    return [
        _ / 1024 / 1024
        for _ in [
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
            *torch.cuda.mem_get_info()
        ]
    ]
