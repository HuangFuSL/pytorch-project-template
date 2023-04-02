from typing import Any, Callable, List, Sequence

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QMainWindow,
                               QProgressBar, QRadioButton, QWidget)

from .. import PROJECT_NAME
from ..model import closure
from ..wrappers import QScalarStorage, QTrainingWorker
from . import dynamic, form_ui, profile, torch


def setupWidgets(*widgets, controller: QRadioButton | QCheckBox, reverse=False):
    def f():
        for w in widgets:
            w.setEnabled(controller.isChecked() ^ reverse)
    if isinstance(controller, QCheckBox):
        controller.stateChanged.connect(f)  # type: ignore
    elif isinstance(controller, QRadioButton):
        controller.toggled.connect(f)  # type: ignore
    f()


def setupBackendToggle(
    disabled: List[QCheckBox], enabled: List[QCheckBox],
    cudaGroup: List[QCheckBox], cudnnGroup: List[QCheckBox]
):
    d, e, f, cudaStatus, cudnnStatus = torch.detectBackends()
    for widget, _ in zip(disabled, d):
        widget.setChecked(_)
    for widget, _ in zip(enabled, e):
        widget.setEnabled(_)
    for widget, (_, setter) in zip(enabled, f):
        widget.setChecked(_)
        widget.stateChanged.connect(setter)  # type: ignore
    for widget, (_, setter) in zip(cudaGroup, cudaStatus):
        widget.setEnabled(d[0])
        if d[0]:
            widget.setChecked(_)
        widget.stateChanged.connect(setter)  # type: ignore
    for widget, (_, setter) in zip(cudnnGroup, cudnnStatus):
        widget.setEnabled(e[0] and f[0][0])
        if e[0] and f[0]:
            widget.setChecked(_)
        widget.stateChanged.connect(setter)  # type: ignore


def setWidgetAttribute(*widgets: QWidget, method: Callable[..., None], args: Sequence[Any]):
    for widget in widgets:
        method(widget, *args)


def refreshProgress(widget: QProgressBar, value: float, max_: float):
    while value < 100 and value > 0:
        value *= 100
        max_ *= 100
    widget.setMaximum(int(max_))
    widget.setValue(int(value))


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = form_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupTrainerPage()
        self.setupCudaPage()
        self.setupProfilingPage()
        self.setWindowTitle(PROJECT_NAME)

        self.profile = []
        self.epochStartTime = 0

        self.trainer = QTrainingWorker()
        self.scalars = QScalarStorage(self.trainer)
        self.trainer.ended.connect(self.on_trainer_ended)
        self.trainer.epochStart.connect(self.on_epochStart)
        self.trainer.epochEnd.connect(self.on_epochEnd)
        self.trainer.prof.connect(self.onProfileReceived)
        self.trainer.setClosure(closure)

    def setupTrainerPage(self):
        setupWidgets(
            self.ui.schedulerSelectCombo, self.ui.schedulerLabel1,
            self.ui.schedulerLabel2, self.ui.schedulerParamTable,
            controller=self.ui.schedulerToggle
        )
        setupWidgets(
            self.ui.modelSeedInput, self.ui.modelSeedLabel,
            controller=self.ui.modelInitToggle
        )
        setupWidgets(
            self.ui.modelPathInput, self.ui.modelPathButton,
            controller=self.ui.modelLoadToggle
        )

        self.modelSelector, self.modelParams = dynamic.buildDynamicClassSelector(
            self.ui.modelSelectCombo, self.ui.modelParamTable,
            dynamic.detectModels, {'self'}
        )
        self.ui.reloadModelButton.clicked.connect(self.modelSelector.refresh)
        self.optimSelector, self.optimParams = dynamic.buildDynamicClassSelector(
            self.ui.optimSelectCombo, self.ui.optimParamTable,
            dynamic.detectOptimizers, {'self', 'params'}
        )
        self.schedulerSelector, self.schedulerParams = dynamic.buildDynamicClassSelector(
            self.ui.schedulerSelectCombo, self.ui.schedulerParamTable,
            dynamic.detectSchedulers, {'self', 'optimizer'}
        )
        self.dataSelector = dynamic.buildDynamicObjectSelector(
            self.ui.dataSelectCombo, dynamic.detectDataLoaders
        )
        self.ui.reloadDataButton.clicked.connect(self.dataSelector.refresh)

    def setupCudaPage(self):
        setupBackendToggle([
            self.ui.cudaAvailableToggle,
            self.ui.mpsAvailableToggle,
            self.ui.mklAvailableToggle,
            self.ui.mkldnnAvailableToggle,
            self.ui.openmpAvailableToggle
        ], [
            self.ui.cudnnAvailableToggle,
            self.ui.opteinsumAvailableToggle
        ], [
            self.ui.cudaTF32Toggle,
            self.ui.cudaFP16Toggle,
            self.ui.cudaBF16Toggle
        ], [
            self.ui.cudnnTF32Toggle,
            self.ui.cudnnDetermToggle,
            self.ui.cudnnBenchToggle
        ])
        self.ui.cudaVersionLabel.setText(torch.getCudaVersion())
        cudaStatus = self.ui.cudaAvailableToggle.isChecked()
        cudaDevices = torch.getCudaDevices()
        setWidgetAttribute(
            self.ui.useCudaToggle, self.ui.cudaDevicePrompt,
            self.ui.cudaSelectCombo, self.ui.cudaVersionPrompt,
            self.ui.cudaVersionLabel, self.ui.cudaMetricsGroup,
            method=QWidget.setEnabled, args=[cudaStatus]
        )
        self.ui.useCudaToggle.setChecked(cudaStatus)
        self.ui.cudaSelectCombo.clear()
        for k, v in cudaDevices.items():
            text = f'[cuda:{k}]: {v["name"]} - {v["total_memory"]} MiB'
            self.ui.cudaSelectCombo.addItem(text)

        def refreshMetrics():
            a, r, f, t = torch.getGpuMemory()
            refreshProgress(self.ui.allocatedProgress, a, r)
            refreshProgress(self.ui.reservedProgress, r, t)
            refreshProgress(self.ui.memoryProgress, t - f, t)
        timer = QTimer(self)
        timer.setInterval(1000)
        timer.timeout.connect(refreshMetrics)  # type: ignore
        timer.start()
        refreshMetrics()

    def setupProfilingPage(self):
        cStatus = profile.cProfileAvailable()
        pyStatus = profile.pyProfileAvailable()
        if not cStatus and not pyStatus:
            self.ui.profileToggle.setEnabled(False)
        if not cStatus:
            self.ui.cProfileToggle.setEnabled(False)
            self.ui.pyProfileToggle.setChecked(True)
        else:
            setupWidgets(
                self.ui.cProfileToggle, controller=self.ui.profileToggle
            )
        if not pyStatus:
            self.ui.pyProfileToggle.setEnabled(False)
            self.ui.cProfileToggle.setChecked(True)
        else:
            setupWidgets(
                self.ui.pyProfileToggle, controller=self.ui.profileToggle
            )

        def onProfileChanged(index: int):
            if not self.profile:
                self.ui.profileStatsTable.clear()
                return
            prof, time = self.profile[index]
            self.ui.profileStatsTable.clear()
            tCall, pCall = profile.drawTable(prof, self.ui.profileStatsTable)
            summary = f'{tCall} function calls ({pCall} primitive calls) in {time:.3f} seconds'
            self.ui.profileSummaryLabel.setText(summary)

        def onSortChanged(index: int, order: Qt.SortOrder):
            self.ui.profileStatsTable.sortByColumn(index, order)

        header = self.ui.profileStatsTable.horizontalHeader()
        header.sortIndicatorChanged.connect(onSortChanged)
        self.ui.profileEpochCombo.currentIndexChanged.connect(onProfileChanged)

    def setupTrainParams(self, initializeModel: bool = True):
        # Super parameters
        epoch = self.ui.epochInput.value()
        batchSize = self.ui.batchSizeInput.value()
        shuffle = self.ui.shuffleInput.isChecked()
        enableScheduler = self.ui.schedulerToggle.isChecked()
        useCuda = self.ui.useCudaToggle.isChecked()
        cudaDevId = self.ui.cudaSelectCombo.currentIndex()
        if self.ui.profileToggle.isChecked():
            if self.ui.cProfileToggle.isChecked():
                profile = 'c'
            else:
                profile = 'py'
        else:
            profile = None

        # Model and optimizer
        if initializeModel:
            model = self.modelSelector.value[0](**self.modelParams.buildArgs())
            if self.ui.modelInitToggle.isChecked():
                if not self.ui.modelSeedInput.text():
                    torch.resetSeed()
                else:
                    torch.setSeed(int(self.ui.modelSeedInput.text()))
            if self.ui.modelLoadToggle.isChecked():
                torch.resetSeed()
                torch.loadModelDict(model, self.ui.modelPathInput.text())
        else:
            model = self.trainer.model
        optim = self.optimSelector.value[0]
        optimArgs = self.optimParams.buildArgs()
        if enableScheduler:
            scheduler = self.schedulerSelector.value[0]
            schedulerArgs = self.schedulerParams.buildArgs()
        dataloader = self.dataSelector.value.dataloader(batchSize, shuffle)
        dev = torch.getDevice(cudaDevId if useCuda else None)

        self.trainer.setModel(model)
        self.trainer.setEpoch(epoch)
        self.trainer.setDevice(dev)
        self.trainer.setOptimizer(optim, model.parameters(), **optimArgs)
        if enableScheduler:
            self.trainer.setScheduler(
                scheduler, **schedulerArgs  # type: ignore
            )
        self.trainer.setDataloader(dataloader)
        self.trainer.setProfile(profile)
        self.profile.clear()

        # UI element
        self.ui.buttonStart.setEnabled(False)
        self.ui.buttonSave.setEnabled(False)
        self.ui.buttonScalar.setEnabled(False)
        self.ui.buttonContinue.setEnabled(False)
        self.ui.trainProgress.setValue(0)
        self.ui.trainProgress.setMaximum(epoch)
        self.ui.profileEpochCombo.clear()
        self.ui.profileStatsTable.clear()
        self.ui.profileSummaryLabel.setText(
            'xxx function calls (yyy primitive calls) in zzz seconds'
        )

        if initializeModel:
            self.scalars.clear()
            self.trainer.resetEpoch()
        self.trainer.start()

    @Slot()
    def on_buttonStart_clicked(self):
        self.setupTrainParams()

    @Slot()
    def on_buttonContinue_clicked(self):
        self.setupTrainParams(False)

    @Slot()
    def on_modelLoadToggle_toggled(self):
        if not self.ui.modelLoadToggle.isChecked():
            return
        if not self.ui.modelPathInput.text():
            self.on_modelPathButton_clicked()

    @Slot()
    def on_modelPathButton_clicked(self):
        path = QFileDialog.getOpenFileName(
            self, 'Open model', '.', 'Model (*.pt *.pth)'
        )[0]
        self.ui.modelPathInput.setText(path)

    @Slot()
    def on_buttonSave_clicked(self):
        path = QFileDialog.getSaveFileName(
            self, 'Save model', '.', 'Model (*.pt *.pth)'
        )[0]
        if path:
            torch.saveModelDict(self.trainer.model, path)

    @Slot()
    def on_buttonScalar_clicked(self):
        filters = {'CSV (*.csv)': 'csv', 'JSON (*.json)': 'json'}
        dest, filter = QFileDialog.getSaveFileName(
            self, 'Save scalars', '.', ';;'.join(filters)
        )
        if not dest:
            return
        if filters[filter] == 'csv':
            self.scalars.to_csv(dest)
        elif filters[filter] == 'json':
            self.scalars.to_json(dest)

    @Slot(name='ended')
    def on_trainer_ended(self):
        self.ui.buttonStart.setEnabled(True)
        self.ui.buttonSave.setEnabled(True)
        self.ui.buttonScalar.setEnabled(True)
        self.ui.buttonContinue.setEnabled(True)

    @Slot(int, float)
    def on_epochStart(self, epoch: int, time: float):
        self.epochStartTime = time

    @Slot(int, float)
    def on_epochEnd(self, epoch: int, time: float):
        self.ui.trainProgress.setValue(epoch + 1)

    def onProfileReceived(self, prof, time):
        self.profile.append((prof, time))
        self.ui.profileEpochCombo.addItem(
            f'Epoch {len(self.profile)}: {time:.3f}s'
        )
        if len(self.profile) == 1:
            self.ui.profileEpochCombo.setCurrentIndex(0)

    @Slot()
    def on_buttonSaveProfile_clicked(self):
        filters = {'CSV (*.csv)': 'csv', 'JSON (*.json)': 'json'}
        dest, filter = QFileDialog.getSaveFileName(
            self, 'Save profile stats', '.', ';;'.join(filters)
        )
        if not dest:
            return
        index = self.ui.profileEpochCombo.currentIndex()
        profile.saveProfile(self.profile[index][0], dest, filters[filter])
