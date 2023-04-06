import dataclasses
from typing import List, Optional

import torch.autograd.profiler_util


@dataclasses.dataclass
class PytorchProfileRecord():
    key: Optional[str]
    count: int
    node_id: int
    cpu_time_total: Optional[int]
    cuda_time_total: Optional[int]
    self_cpu_time_total: Optional[int]
    self_cuda_time_total: Optional[int]
    cpu_memory_usage: Optional[int]
    cuda_memory_usage: Optional[int]
    self_cpu_memory_usage: Optional[int]
    self_cuda_memory_usage: Optional[int]

    @classmethod
    def fromProfile(
        cls,
        profile: torch.autograd.profiler.profile,
        record: torch.autograd.profiler_util.FunctionEventAvg
    ):
        cpu = profile.use_cpu
        gpu = profile.use_cuda
        mem = profile.profile_memory
        return cls(
            key=record.key,
            count=record.count,
            node_id=record.node_id,
            cpu_time_total=record.cpu_time_total if cpu else None,
            cuda_time_total=record.cuda_time_total if gpu else None,
            self_cpu_time_total=record.self_cpu_time_total if cpu else None,
            self_cuda_time_total=record.self_cuda_time_total if gpu else None,
            cpu_memory_usage=record.cpu_memory_usage if mem else None,
            cuda_memory_usage=record.cuda_memory_usage if mem else None,
            self_cpu_memory_usage=record.self_cpu_memory_usage if mem else None,
            self_cuda_memory_usage=record.self_cuda_memory_usage if mem else None,
        )

    def toDict(self):
        return dataclasses.asdict(self)
