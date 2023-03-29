from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QCheckBox, QFileDialog, QMainWindow, QRadioButton

from .. import PROJECT_NAME
from ..model import closure
from ..wrappers import QScalarStorage, QTrainingWorker
from . import dynamic, form_ui, torch


def setupWidgets(*widgets, controller: QRadioButton | QCheckBox, reverse=False):
    def f():
        for w in widgets:
            w.setEnabled(controller.isChecked() ^ reverse)
    if isinstance(controller, QCheckBox):
        controller.stateChanged.connect(f)
    elif isinstance(controller, QRadioButton):
        controller.toggled.connect(f)
    f()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = form_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupWidgets()
        self.setupSelector()
        self.setWindowTitle(PROJECT_NAME)

        self.epochStartTime = 0

        self.trainer = QTrainingWorker()
        self.scalars = QScalarStorage(self.trainer)
        self.trainer.ended.connect(self.on_trainer_ended)
        self.trainer.epochStart.connect(self.on_epochStart)
        self.trainer.epochEnd.connect(self.on_epochEnd)
        self.trainer.setClosure(closure)

    def setupWidgets(self):
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

    def setupSelector(self):
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

    def setupTrainParams(self, initializeModel: bool = True):
        # Super parameters
        epoch = self.ui.epochInput.value()
        batchSize = self.ui.batchSizeInput.value()
        shuffle = self.ui.shuffleInput.isChecked()
        enableScheduler = self.ui.schedulerToggle.isChecked()

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
        scheduler = self.schedulerSelector.value[0]
        schedulerArgs = self.schedulerParams.buildArgs() if enableScheduler else {}
        dataloader = self.dataSelector.value.dataloader(batchSize, shuffle)

        self.trainer.setModel(model)
        self.trainer.model.train()
        self.trainer.setEpoch(epoch)
        self.trainer.setOptimizer(optim, model.parameters(), **optimArgs)
        if enableScheduler:
            self.trainer.setScheduler(scheduler, **schedulerArgs)
        self.trainer.setDataloader(dataloader)

        # UI element
        self.ui.buttonStart.setEnabled(False)
        self.ui.buttonSave.setEnabled(False)
        self.ui.buttonScalar.setEnabled(False)
        self.ui.buttonContinue.setEnabled(False)
        self.ui.trainProgress.setValue(0)
        self.ui.trainProgress.setMaximum(epoch)

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
