import sys
import subprocess
import io
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QTextEdit, QComboBox, QSpinBox, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal

import torch
import argparse
import data_loader
import models
import train

DATASET_URLS = [
    'https://drive.google.com/drive/folders/131rtWfO1wKf7c-2nVgw65Tkpod3N0wbv?usp=sharing',
    'https://drive.google.com/drive/folders/1PE8LNwFMmjE_LQtUA2vcbQYk9Jh0ohGa?usp=sharing'
]

class StreamEmitter(io.TextIOBase):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
    def write(self, msg):
        if msg:
            self.signal.emit(msg)
    def flush(self):
        pass

class DownloadThread(QThread):
    message = pyqtSignal(str)
    def run(self):
        for url in DATASET_URLS:
            cmd = ['gdown', url, '--folder']
            self.message.emit(f"Running: {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                self.message.emit(line)
            proc.wait()
        self.message.emit('Download complete.\n')

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

class TrainThread(QThread):
    message = pyqtSignal(str)
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backup = sys.stdout
        sys.stdout = StreamEmitter(self.message)
        try:
            data, feat_dim, num_classes = data_loader.load_dataset(self.config.dataset, root='data')
            data = data.to(device)
            name = self.config.model.lower()
            if name == 'baselinegcn':
                model = models.BaselineGCN(feat_dim, self.config.hidden, num_classes, self.config.dropout)
            elif name == 'graphsage':
                model = models.GraphSAGE(feat_dim, self.config.hidden, num_classes, self.config.num_layers, self.config.dropout)
            elif name == 'tgat':
                model = models.TGAT(feat_dim, self.config.hidden, num_classes, num_layers=self.config.num_layers, dropout=self.config.dropout)
            elif name == 'tgn':
                model = models.TGN(data.num_nodes, self.config.hidden, 1, num_classes)
            elif name == 'agnnet':
                model = models.AGNNet(feat_dim, self.config.hidden, num_classes, dropout=self.config.dropout)
            else:
                self.message.emit(f'Unknown model {self.config.model}\n')
                return
            model = model.to(device)
            train.run_training_session(
                model,
                data,
                data if self.config.dataset != 'Reddit' else None,
                data if self.config.dataset != 'Reddit' else None,
                data if self.config.dataset != 'Reddit' else None,
                False,
                device,
                argparse.Namespace(
                    model=self.config.model,
                    dataset=self.config.dataset,
                    epochs=self.config.epochs,
                    lr=self.config.lr,
                    hidden_channels=self.config.hidden,
                    dropout=self.config.dropout,
                    weight_decay=self.config.weight_decay,
                    num_layers=self.config.num_layers,
                )
            )
        finally:
            sys.stdout = backup
            self.message.emit('Experiment complete.\n')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GNN Experiments')
        layout = QVBoxLayout(self)

        dl_group = QGroupBox('Dataset Download')
        dl_layout = QVBoxLayout()
        self.download_btn = QPushButton('Download Datasets')
        self.download_btn.clicked.connect(self.start_download)
        self.download_output = QTextEdit(); self.download_output.setReadOnly(True)
        dl_layout.addWidget(self.download_btn)
        dl_layout.addWidget(self.download_output)
        dl_group.setLayout(dl_layout)

        exp_group = QGroupBox('Run Experiment')
        form = QFormLayout()
        self.model_combo = QComboBox(); self.model_combo.addItems(['BaselineGCN','GraphSAGE','TGAT','TGN','AGNNet'])
        self.dataset_combo = QComboBox(); self.dataset_combo.addItems(['OGB-Arxiv','Reddit','TGB-Wiki','MOOC'])
        self.epoch_spin = QSpinBox(); self.epoch_spin.setRange(1,200); self.epoch_spin.setValue(20)
        self.lr_edit = QLineEdit('0.01')
        self.hidden_edit = QLineEdit('64')
        self.dropout_edit = QLineEdit('0.5')
        self.wd_edit = QLineEdit('5e-4')
        self.layers_spin = QSpinBox(); self.layers_spin.setRange(1,10); self.layers_spin.setValue(2)
        form.addRow('Model', self.model_combo)
        form.addRow('Dataset', self.dataset_combo)
        form.addRow('Epochs', self.epoch_spin)
        form.addRow('Learning Rate', self.lr_edit)
        form.addRow('Hidden Channels', self.hidden_edit)
        form.addRow('Dropout', self.dropout_edit)
        form.addRow('Weight Decay', self.wd_edit)
        form.addRow('Num Layers', self.layers_spin)
        self.run_btn = QPushButton('Run')
        self.run_btn.clicked.connect(self.start_experiment)
        self.exp_output = QTextEdit(); self.exp_output.setReadOnly(True)
        exp_layout = QVBoxLayout(); exp_layout.addLayout(form); exp_layout.addWidget(self.run_btn); exp_layout.addWidget(self.exp_output)
        exp_group.setLayout(exp_layout)

        layout.addWidget(dl_group)
        layout.addWidget(exp_group)
        self.setLayout(layout)

    def start_download(self):
        self.download_btn.setEnabled(False)
        self.download_output.clear()
        self.dl_thread = DownloadThread()
        self.dl_thread.message.connect(self.download_output.insertPlainText)
        self.dl_thread.finished.connect(lambda: self.download_btn.setEnabled(True))
        self.dl_thread.start()

    def start_experiment(self):
        self.run_btn.setEnabled(False)
        self.exp_output.clear()
        config = ExperimentConfig(
            model=self.model_combo.currentText(),
            dataset=self.dataset_combo.currentText(),
            epochs=self.epoch_spin.value(),
            lr=float(self.lr_edit.text()),
            hidden=int(self.hidden_edit.text()),
            dropout=float(self.dropout_edit.text()),
            weight_decay=float(self.wd_edit.text()),
            num_layers=self.layers_spin.value(),
        )
        self.train_thread = TrainThread(config)
        self.train_thread.message.connect(self.exp_output.insertPlainText)
        self.train_thread.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.train_thread.start()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
