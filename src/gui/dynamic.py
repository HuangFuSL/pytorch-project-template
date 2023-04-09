import importlib
import inspect
import typing as T
from types import ModuleType

import torch.optim
import torch.utils.data
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QComboBox, QTableWidget, QTableWidgetItem

from .. import data as data_def
from .. import model as model_def
from ..wrappers.dataloader import QDataLoader

U = T.TypeVar('U')


def detectClasses(
    module: ModuleType, reload: bool, *types: T.Type[U]
) -> T.Dict[str, T.Tuple[T.Type[U], inspect.Signature]]:
    objs = importlib.reload(module) if reload else module
    result: T.Dict[str, T.Tuple[T.Type[U], inspect.Signature]] = {}
    for name in dir(objs):
        if name.startswith('_'):
            continue
        obj = getattr(objs, name)
        if not isinstance(obj, type):
            continue
        if not any(issubclass(obj, type_) for type_ in types):
            continue
        obj = T.cast(T.Type[U], obj)
        result[name] = (obj, inspect.signature(obj.__init__))
    return result


def detectObjects(
    module: ModuleType, reload: bool, type: T.Type[U]
) -> T.Dict[str, U]:
    objs = importlib.reload(module) if reload else module
    result: T.Dict[str, U] = {}
    for name in dir(objs):
        if name.startswith('_'):
            continue
        obj = getattr(objs, name)
        if not isinstance(obj, type):
            continue
        result[name] = obj
    return result


def detectModels():
    return detectClasses(model_def, True, torch.nn.Module)


def detectOptimizers():
    return detectClasses(torch.optim, False, torch.optim.Optimizer)


def detectSchedulers():
    try:
        return detectClasses(
            torch.optim.lr_scheduler, False,
            torch.optim.lr_scheduler._LRScheduler,
            torch.optim.lr_scheduler.LRScheduler
        )
    except AttributeError:
        return detectClasses(
            torch.optim.lr_scheduler, False,
            torch.optim.lr_scheduler._LRScheduler
        )


def detectDataLoaders():
    return detectObjects(data_def, True, QDataLoader)


class ComboSelector(T.Generic[U]):

    def __init__(self, combo: QComboBox, refresh: T.Callable[[], T.Dict[str, U]]):
        self._combo = combo
        self._map: T.Dict[str, U] = {}
        self._busy = False
        self._refresh = refresh
        self._callback: T.Callable[[T.Any], None] = lambda _: None

    def callback(self, index: int):
        if not self._busy:
            self._callback(self.value)

    @property
    def key(self):
        return self._combo.currentText()

    @property
    def value(self):
        return self._map[self.key]

    @Slot()
    def refresh(self):
        self.refreshBusy = True
        self._combo.clear()
        self._map = self._refresh()
        self._combo.addItems(list(self._map))
        self.refreshBusy = False

    def setUpdateCallback(self, callback: T.Callable[[T.Any], None]):
        self._callback = callback
        self._combo.currentIndexChanged.connect(self.callback)

    def execute(self):
        self.callback(self._combo.currentIndex())


class ParameterTable():

    _ignored = set()

    def __init__(self, table: QTableWidget, ignore: T.Optional[T.Set[str]] = None):
        self._table = table
        self._params = {}
        if ignore is not None:
            self._ignored |= ignore
        self._signature: T.Optional[inspect.Signature] = None

    def refresh(self, signature: inspect.Signature):
        self._signature = signature
        self._params = {}
        self._table.clear()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(['Parameter', 'Type', 'Value'])
        new_table = []
        for i, (k, v) in enumerate(signature.parameters.items()):
            type_ = '(unknown)'
            if v.annotation is inspect._empty:
                if v.default is not None and v.default is not inspect._empty:
                    type_ = type(v.default).__name__
                    if v.default == 0 and type_ == 'int':
                        type_ = 'float'
            elif isinstance(v.annotation, type):
                type_ = v.annotation.__name__
            elif v.annotation._name == 'Optional':
                type_ = v.annotation.__args__[0].__name__
            new_table.append(
                (k, type_, str(v.default) if v.default is not inspect._empty else '')
            )

        def f(_):
            k, t, v = _
            return k not in self._ignored
        filtered = list(filter(f, new_table))
        self._table.setRowCount(len(filtered))
        for i, (k, t, v) in enumerate(filtered):
            self._table.setRowHeight(i, 20)
            item0 = QTableWidgetItem(k)
            item0.setFlags(item0.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 0, item0)
            item1 = QTableWidgetItem(t)
            item1.setFlags(item0.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 1, item1)
            item2 = QTableWidgetItem(v)
            self._table.setItem(i, 2, item2)

    def buildArgs(self):
        for _ in range(self._table.rowCount()):
            k = self._table.item(_, 0).text()
            v = eval(self._table.item(_, 2).text())
            self._params[k] = v
        return self._params


def buildDynamicClassSelector(
    combo: QComboBox, table: QTableWidget,
    loader: T.Callable[[], T.Dict[str, T.Tuple[T.Type[U], inspect.Signature]]],
    ignored_params: T.Optional[T.Set[str]] = None
) -> T.Tuple[ComboSelector[T.Tuple[T.Type[U], inspect.Signature]], ParameterTable]:
    selector = ComboSelector(combo, loader)
    selector.refresh()
    defaultIgnore = {'self', 'args', 'kwargs'}
    params = ParameterTable(table, defaultIgnore | (ignored_params or set()))
    try:
        params.refresh(selector.value[1])
    except KeyError:
        pass
    selector.setUpdateCallback(lambda _: params.refresh(_[1]))
    return selector, params


def buildDynamicObjectSelector(
    combo: QComboBox, loader: T.Callable[[], T.Dict[str, U]]
) -> ComboSelector[U]:
    selector = ComboSelector(combo, loader)
    selector.refresh()
    return selector
