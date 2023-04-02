import csv
import functools
import json
import pstats
from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


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


def transform(profile: Dict[str, pstats.FunctionProfile]):
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


def saveProfile(
    profile: Dict[str, pstats.FunctionProfile], path: str, format: str
):
    records = transform(profile)
    if format == 'csv':
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    elif format == 'json':
        with open(path, 'w') as f:
            json.dump(records, f)


def drawTable(profile: Dict[str, pstats.FunctionProfile], widget: QTableWidget):
    records = transform(profile)
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
