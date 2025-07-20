import subprocess
import threading
import sys
from dataclasses import dataclass
from flask import Flask, render_template, request, redirect, url_for
import torch
import argparse
import data_loader
import models
import train

app = Flask(__name__)

DATASET_URLS = [
    'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing'
]

# In-memory logs for simplicity
download_log = []
experiment_log = []

download_thread = None
experiment_thread = None


def log_writer(log_list, message):
    log_list.append(message)
    print(message)


def run_download():
    download_log.clear()
    for url in DATASET_URLS:
        cmd = ['gdown', url, '--folder']
        log_writer(download_log, f"Running: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log_writer(download_log, line.rstrip())
        proc.wait()
    log_writer(download_log, 'Download complete.')


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


def run_experiment(cfg: ExperimentConfig):
    experiment_log.clear()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_writer(experiment_log, f'Using device {device}')
    backup = sys.stdout

    class Stream:
        def write(self, msg):
            if msg.strip():
                log_writer(experiment_log, msg.rstrip())
        def flush(self):
            pass

    sys.stdout = Stream()
    try:
        data, feat_dim, num_classes = data_loader.load_dataset(cfg.dataset, root='simple_data')
        data = data.to(device)
        name = cfg.model.lower()
        if name == 'baselinegcn':
            model = models.BaselineGCN(feat_dim, cfg.hidden, num_classes, cfg.dropout)
        elif name == 'graphsage':
            model = models.GraphSAGE(feat_dim, cfg.hidden, num_classes, cfg.num_layers, cfg.dropout)
        elif name == 'tgat':
            model = models.TGAT(feat_dim, cfg.hidden, num_classes, num_layers=cfg.num_layers, dropout=cfg.dropout)
        elif name == 'tgn':
            model = models.TGN(data.num_nodes, cfg.hidden, 1, num_classes)
        elif name == 'agnnet':
            model = models.AGNNet(feat_dim, cfg.hidden, num_classes, dropout=cfg.dropout)
        else:
            log_writer(experiment_log, f'Unknown model {cfg.model}')
            return
        model = model.to(device)
        train.run_training_session(
            model,
            data,
            data if cfg.dataset != 'Reddit' else None,
            data if cfg.dataset != 'Reddit' else None,
            data if cfg.dataset != 'Reddit' else None,
            False,
            device,
            argparse.Namespace(
                model=cfg.model,
                dataset=cfg.dataset,
                epochs=cfg.epochs,
                lr=cfg.lr,
                hidden_channels=cfg.hidden,
                dropout=cfg.dropout,
                weight_decay=cfg.weight_decay,
                num_layers=cfg.num_layers,
            )
        )
    finally:
        sys.stdout = backup
        log_writer(experiment_log, 'Experiment complete.')


def ensure_download_thread():
    global download_thread
    if download_thread is None or not download_thread.is_alive():
        download_thread = threading.Thread(target=run_download, daemon=True)
        download_thread.start()


def ensure_experiment_thread(cfg):
    global experiment_thread
    if experiment_thread is None or not experiment_thread.is_alive():
        experiment_thread = threading.Thread(target=run_experiment, args=(cfg,), daemon=True)
        experiment_thread.start()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/download', methods=['POST'])
def download():
    ensure_download_thread()
    return redirect(url_for('view_log', task='download'))


@app.route('/run', methods=['POST'])
def run():
    cfg = ExperimentConfig(
        model=request.form['model'],
        dataset=request.form['dataset'],
        epochs=int(request.form['epochs']),
        lr=float(request.form['lr']),
        hidden=int(request.form['hidden']),
        dropout=float(request.form['dropout']),
        weight_decay=float(request.form['weight_decay']),
        num_layers=int(request.form['num_layers']),
    )
    ensure_experiment_thread(cfg)
    return redirect(url_for('view_log', task='experiment'))



@app.route('/logs/<task>')
def view_log(task):
    log = '\n'.join(download_log if task == 'download' else experiment_log)
    return render_template('log.html', log=log, task=task)

if __name__ == '__main__':
    app.run(debug=True)
