import csv
import json
from typing import Dict, List, Mapping, Sequence

from PySide6.QtCore import QObject, Signal, Slot


class QScalarStorage(QObject):

    update = Signal(name='update')

    def __init__(self, parent: QObject):
        super().__init__()
        self._scalars: Dict[str, List[float]] = {'_time': [], '_epoch': []}
        assert hasattr(parent, 'scalars')
        assert isinstance(parent.scalars, Signal)  # type: ignore
        parent.scalars.connect(self.onScalarReceived)  # type: ignore

    def __getitem__(self, key: str) -> Sequence[float]:
        return self._scalars[key]

    def _row_iter(self):
        for _ in range(max(map(len, self._scalars.values()))):
            yield {k: v[_] if _ < len(v) else None for k, v in self._scalars.items()}

    @property
    def content(self) -> Mapping[str, Sequence[float]]:
        return self._scalars

    @Slot(str, float, bool, name='scalars')
    def onScalarReceived(self, name: str, value: float, silent: bool = False):
        self._scalars.setdefault(name, []).append(value)
        if not silent:
            self.update.emit()

    def add(self, name: str, value: float, silent: bool = False):
        self._scalars.setdefault(name, []).append(value)
        if not silent:
            self.update.emit()

    def clear(self):
        self._scalars.clear()
        self.update.emit()

    def to_csv(self, path: str):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, self._scalars.keys())
            writer.writeheader()
            writer.writerows(self._row_iter())

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(list(self._row_iter()), f)
