import functools
import time
from typing import Any, Dict, Optional

import torch
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mkldnn
import torch.backends.mps
import torch.backends.openmp
import torch.backends.opt_einsum
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


def detectBackends():
    return [
        torch.cuda.is_available(),
        torch.backends.mps.is_available(),
        torch.backends.mkl.is_available(),
        torch.backends.mkldnn.is_available(),
        torch.backends.openmp.is_available()
    ], [
        torch.backends.cudnn.is_available(),
        torch.backends.opt_einsum.is_available()
    ], [
        (torch.backends.cudnn.enabled, setCudnnEnabled),
        (torch.backends.opt_einsum.enabled, setOpteinsumEnabled)
    ], [
        (torch.backends.cuda.matmul.allow_tf32, setCudaMatmulAllowTF32),
        (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            setCudaMatmulAllowFP16
        ), (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            setCudaMatmulAllowBF16
        )
    ], [
        (torch.backends.cudnn.allow_tf32, setCudnnAllowTF32),
        (torch.backends.cudnn.deterministic, setCudnnDeterministic),
        (torch.backends.cudnn.benchmark, setCudnnBenchmark)
    ]


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
    return [
        _ / 1024 / 1024
        for _ in [
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
            *torch.cuda.mem_get_info()
        ]
    ]
