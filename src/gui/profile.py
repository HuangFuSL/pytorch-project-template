import functools
import pstats
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from ..common import PytorchProfileRecord, dumpData


@functools.lru_cache()
def cProfileAvailable():
    try:
        import cProfile
        return True
    except ImportError:
        return False


@functools.lru_cache()
def pyProfileAvailable():
    try:
        import profile
        return True
    except ImportError:
        return False


def toTimeString(time: Optional[float]):
    if time is None:
        return 'untracked'
    units = ['us', 'ms', 's']
    for unit in units:
        if time < 1000:
            return f'{time:.3f} {unit}'
        time /= 1000
    time *= 1000
    return f'{time:.3f} s'


def toMemoryString(memory: Optional[int]):
    if memory is None:
        return 'untracked'
    units = ['B', 'KB', 'MB', 'GB']
    memory_ = abs(memory / 8)
    flag = '-' if memory < 0 else ''
    for unit in units:
        if memory_ < 1024:
            return f'{flag}{memory_:.1f} {unit}'
        memory_ /= 1024
    memory_ *= 1024
    return f'{flag}{memory_:.1f} GB'


def calcAvg(record: Dict[str, Any], from_: str):
    count = record['count']
    if record[from_] is None:
        record['avg_' + from_] = None
    else:
        record['avg_' + from_] = record[from_] / count if count else 0.0


def getCustomDisplayItem(func: Callable[[Any], str]):
    class CustomDisplayItem(QTableWidgetItem):
        def __init__(self, value: Optional[Any]):
            super().__init__(func(value))
            if value is not None:
                self._value = value
            else:
                self._value = 0

        def __lt__(self, other: 'CustomDisplayItem'):
            return self._value < other._value

    return CustomDisplayItem


MemoryItem = getCustomDisplayItem(toMemoryString)
TimeItem = getCustomDisplayItem(toTimeString)


def transformPython(profile: Dict[str, pstats.FunctionProfile]):
    def toDict(profile: pstats.FunctionProfile):
        nameMap = {
            'ncalls': 'count', 'tottime': 't', 'percall_tottime': 'at',
            'cumtime': 'T', 'percall_cumtime': 'aT',
            'file_name': 'file', 'line_number': 'lineno'
        }
        return {v: getattr(profile, k) for k, v in nameMap.items()}
    records = [{'name': k, **toDict(v)} for k, v in profile.items()]
    for _ in records:
        calls = _['count']
        if '/' in calls:
            t, p = map(int, calls.split('/'))
        else:
            t, p = int(calls), int(calls)
        _['count'] = t
        _['pcount'] = p
    return records


def transformPytorch(profile: List[PytorchProfileRecord]):
    def toDict(record: PytorchProfileRecord):
        dictRecord = record.toDict()
        avgFields = [
            'cpu_time_total', 'cuda_time_total',
            'self_cpu_time_total', 'self_cuda_time_total'
        ]
        for _ in avgFields:
            calcAvg(dictRecord, _)
        nameMap = {
            'key': 'name', 'node_id': 'device', 'count': 'count',
            'cpu_time_total': 'CT', 'cuda_time_total': 'GT',
            'avg_cpu_time_total': 'ACT', 'avg_cuda_time_total': 'AGT',
            'self_cpu_time_total': 'ct', 'self_cuda_time_total': 'gt',
            'avg_self_cpu_time_total': 'act', 'avg_self_cuda_time_total': 'agt',
            'cpu_memory_usage': 'CM', 'cuda_memory_usage': 'GM',
            'self_cpu_memory_usage': 'cm', 'self_cuda_memory_usage': 'gm',
        }
        ret = {
            v: dictRecord[k]
            for k, v in nameMap.items()
            if dictRecord.get(k, None) is not None
        }
        return ret
    return [toDict(_) for _ in profile]


def savePythonProfile(
    profile: Dict[str, pstats.FunctionProfile], path: str, format: str
):
    dumpData(transformPython(profile), path, format)


def savePytorchProfile(
    profile: List[PytorchProfileRecord], path: str, format: str
):
    dumpData(transformPytorch(profile), path, format)


def drawPythonTable(
    profile: Dict[str, pstats.FunctionProfile],
    widget: QTableWidget
):
    records = transformPython(profile)
    headers = {
        'name': 300, 'count': 40, 'pcount': 40, 't': 40, 'at': 40,
        'T': 40, 'aT': 40, 'file': 600, 'lineno': 40
    }
    widget.clear()
    widget.setSortingEnabled(False)
    widget.setRowCount(len(records))
    widget.setColumnCount(len(headers))
    for i, header in enumerate(headers.values()):
        widget.setColumnWidth(i, header)
    widget.setHorizontalHeaderLabels([*headers])
    tCall, pCall = 0, 0
    for i, record in enumerate(records):
        for j, header in enumerate(headers):
            if header == 'lineno' and int(record[header]) == 0:
                record[header] = 'N/A'
            _ = QTableWidgetItem()
            if isinstance(record[header], (float, int)):
                _.setData(Qt.ItemDataRole.DisplayRole, record[header])
            else:
                _.setText(record[header])
            _.setFlags(_.flags() & ~Qt.ItemFlag.ItemIsEditable)
            widget.setItem(i, j, _)
        tCall += record['count']
        pCall += record['pcount']
    widget.setSortingEnabled(True)
    return tCall, pCall


def drawPytorchTable(profile: List[PytorchProfileRecord], widget: QTableWidget):
    records = transformPytorch(profile)
    longFields = ['name']
    timeFields = ['CT', 'GT', 'ACT', 'AGT', 'ct', 'gt', 'act', 'agt']
    memFields = ['CM', 'GM', 'cm', 'gm']
    headers = {
        k: 300 if k in longFields else 80 for k in records[0].keys()
    }
    widget.clear()
    widget.setSortingEnabled(False)
    widget.setRowCount(len(records))
    widget.setColumnCount(len(headers))
    for i, header in enumerate(headers.values()):
        widget.setColumnWidth(i, header)
    widget.setHorizontalHeaderLabels([*headers])
    for i, record in enumerate(records):
        for j, header in enumerate(headers):
            if isinstance(record[header], (float, int)):
                if header in timeFields:
                    _ = TimeItem(record[header])
                elif header in memFields:
                    _ = MemoryItem(record[header])
                else:
                    _ = QTableWidgetItem(str(record[header]))
            else:
                _ = QTableWidgetItem(record[header])
            _.setFlags(_.flags() & ~Qt.ItemFlag.ItemIsEditable)
            widget.setItem(i, j, _)
    widget.setSortingEnabled(True)
