import sys
import subprocess
import argparse
from dataclasses import dataclass

from PyQt5 import QtCore, QtWidgets
import torch

import data_loader
import models
import train

DATASET_URLS = [
    'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing'
]

class LogEmitter(QtCore.QObject):
    message = QtCore.pyqtSignal(str)

def log_writer(emitter: LogEmitter, message: str):
    emitter.message.emit(message)
    print(message)

@dataclass
class ExperimentConfig:
    model: str
    dataset: str
    epochs: int
    lr: float
    hidden: int
    dropout: float
    weight_decay: float
    num_layers: int

class DownloadWorker(QtCore.QThread):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def run(self):
        for url in DATASET_URLS:
            cmd = ['gdown', url, '--folder']
            log_writer(self.emitter, f"Running: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log_writer(self.emitter, line.rstrip())
            proc.wait()
        log_writer(self.emitter, 'Download complete.')

class ExperimentWorker(QtCore.QThread):
    def __init__(self, cfg: ExperimentConfig, emitter: LogEmitter):
        super().__init__()
        self.cfg = cfg
        self.emitter = emitter

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_writer(self.emitter, f'Using device {device}')
        backup = sys.stdout

        class Stream:
            def write(self, msg):
                if msg.strip():
                    log_writer(self.emitter, msg.rstrip())
            def flush(self):
                pass

        sys.stdout = Stream()
        try:
            if self.cfg.model in ['BaselineGCN', 'GraphSAGE']:
                data, feat_dim, num_classes = data_loader.load_dataset_dgl(self.cfg.dataset, root='simple_data')
                data = data.to(device)
            else:
                data, feat_dim, num_classes = data_loader.load_dataset(self.cfg.dataset, root='simple_data')
                data = data.to(device)
            name = self.cfg.model.lower()
            if name == 'baselinegcn':
                model = models.BaselineGCN(feat_dim, self.cfg.hidden, num_classes, self.cfg.dropout)
            elif name == 'graphsage':
                model = models.GraphSAGE(feat_dim, self.cfg.hidden, num_classes, self.cfg.num_layers, self.cfg.dropout)
            elif name == 'tgat':
                model = models.TGAT(feat_dim, self.cfg.hidden, num_classes, num_layers=self.cfg.num_layers, dropout=self.cfg.dropout)
            elif name == 'tgn':
                model = models.TGN(data.num_nodes, self.cfg.hidden, 1, num_classes)
            elif name == 'agnnet':
                model = models.AGNNet(feat_dim, self.cfg.hidden, num_classes, dropout=self.cfg.dropout)
            else:
                log_writer(self.emitter, f'Unknown model {self.cfg.model}')
                return
            model = model.to(device)
            train.run_training_session(
                model,
                data,
                data if self.cfg.dataset != 'Reddit' else None,
                data if self.cfg.dataset != 'Reddit' else None,
                data if self.cfg.dataset != 'Reddit' else None,
                False,
                device,
                argparse.Namespace(
                    model=self.cfg.model,
                    dataset=self.cfg.dataset,
                    epochs=self.cfg.epochs,
                    lr=self.cfg.lr,
                    hidden_channels=self.cfg.hidden,
                    dropout=self.cfg.dropout,
                    weight_decay=self.cfg.weight_decay,
                    num_layers=self.cfg.num_layers,
                )
            )
        finally:
            sys.stdout = backup
            log_writer(self.emitter, 'Experiment complete.')

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GNN Experiments')
        layout = QtWidgets.QVBoxLayout(self)

        form_layout = QtWidgets.QFormLayout()
        self.model_box = QtWidgets.QComboBox()
        self.model_box.addItems(['BaselineGCN', 'GraphSAGE', 'TGAT', 'TGN', 'AGNNet'])
        form_layout.addRow('Model', self.model_box)

        self.dataset_box = QtWidgets.QComboBox()
        self.dataset_box.addItems(['OGB-Arxiv', 'Reddit', 'TGB-Wiki', 'MOOC'])
        form_layout.addRow('Dataset', self.dataset_box)

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(20)
        form_layout.addRow('Epochs', self.epochs_spin)

        self.lr_edit = QtWidgets.QLineEdit('0.01')
        form_layout.addRow('Learning Rate', self.lr_edit)

        self.hidden_spin = QtWidgets.QSpinBox()
        self.hidden_spin.setValue(64)
        form_layout.addRow('Hidden Channels', self.hidden_spin)

        self.dropout_edit = QtWidgets.QLineEdit('0.5')
        form_layout.addRow('Dropout', self.dropout_edit)

        self.weight_decay_edit = QtWidgets.QLineEdit('5e-4')
        form_layout.addRow('Weight Decay', self.weight_decay_edit)

        self.num_layers_spin = QtWidgets.QSpinBox()
        self.num_layers_spin.setRange(1, 10)
        self.num_layers_spin.setValue(2)
        form_layout.addRow('Num Layers', self.num_layers_spin)

        layout.addLayout(form_layout)

        self.download_btn = QtWidgets.QPushButton('Download Datasets')
        self.run_btn = QtWidgets.QPushButton('Run Experiment')
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.download_btn)
        btn_layout.addWidget(self.run_btn)
        layout.addLayout(btn_layout)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        self.emitter = LogEmitter()
        self.emitter.message.connect(self.append_log)

        self.download_btn.clicked.connect(self.start_download)
        self.run_btn.clicked.connect(self.start_experiment)

        self.download_worker = None
        self.experiment_worker = None

    def append_log(self, text: str):
        self.log_view.appendPlainText(text)

    def start_download(self):
        if self.download_worker is None or not self.download_worker.isRunning():
            self.log_view.clear()
            self.download_worker = DownloadWorker(self.emitter)
            self.download_worker.start()

    def start_experiment(self):
        if self.experiment_worker is None or not self.experiment_worker.isRunning():
            self.log_view.clear()
            cfg = ExperimentConfig(
                model=self.model_box.currentText(),
                dataset=self.dataset_box.currentText(),
                epochs=self.epochs_spin.value(),
                lr=float(self.lr_edit.text()),
                hidden=self.hidden_spin.value(),
                dropout=float(self.dropout_edit.text()),
                weight_decay=float(self.weight_decay_edit.text()),
                num_layers=self.num_layers_spin.value(),
            )
            self.experiment_worker = ExperimentWorker(cfg, self.emitter)
            self.experiment_worker.start()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
