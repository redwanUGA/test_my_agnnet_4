@echo off
setlocal enabledelayedexpansion

REM AGNNet Ablation Study Runner (Windows)
REM This script runs a compact set of ablations isolating key components of AGNNet.
REM Results/logs are saved under logs/ with a timestamp; each run prints config.

set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG_FILE=%LOG_DIR%\run_ablation_agnnet_%TS%.txt"
powershell -NoProfile -Command "Start-Transcript -Path '%LOG_FILE%' | Out-Null"

REM Ensure datasets exist (uses Google Drive folder from README)
if not exist "simple_data" (
  echo [INFO] Downloading datasets to simple_data/ …
  gdown "https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing" --folder
)

REM Default hyperparameters for the ablation baseline
if "%EPOCHS%"=="" set "EPOCHS=50"
set "HIDDEN=128"
set "LAYERS=3"
set "DROPOUT=0.25"
set "LR=0.003"
set "WD=0.0005"
set "TAU=0.9"
set "K=8"

REM Datasets to evaluate. Reddit is sampled, others are full-batch.
set DATASETS=OGB-Arxiv Reddit TGB-Wiki MOOC

REM Ablation variants (name -> flag set):
REM 1) baseline: full AGNNet
REM 2) no_pred_subgraph: ablate predictive subgraph selection (use full graph)
REM 3) no_k_anneal: fix k (disable annealing)
REM 4) no_soft_topk: hard top-k without soft attention cap
REM 5) no_self_loops: remove forced self-loops in subgraph
REM 6) no_edge_threshold: ensure thresholding is disabled (edge_threshold=0.0)

for %%D in (%DATASETS%) do (
  echo [!DATE! !TIME!] ===== Dataset: %%D =====

  REM 1) Baseline
  echo [!DATE! !TIME!] AGNNet/base ds=%%D
  python main.py ^
    --model AGNNet ^
    --dataset %%D ^
    --epochs %EPOCHS% ^
    --hidden-channels %HIDDEN% ^
    --num-layers %LAYERS% ^
    --dropout %DROPOUT% ^
    --tau %TAU% ^
    --k %K% ^
    --k-anneal ^
    --k-min 2 ^
    --k-max %K% ^
    --soft-topk ^
    --ffn-expansion 2.0 ^
    --optimizer adamw ^
    --lr %LR% ^
    --weight-decay %WD% ^
    --lr-schedule cosine ^
    --warmup-epochs 500 ^
    --label-smoothing 0.05 ^
    --edge-threshold 0.0

  REM 2) No predictive subgraph
  echo [!DATE! !TIME!] AGNNet/ablate:no_pred_subgraph ds=%%D
  python main.py ^
    --model AGNNet ^
    --dataset %%D ^
    --epochs %EPOCHS% ^
    --hidden-channels %HIDDEN% ^
    --num-layers %LAYERS% ^
    --dropout %DROPOUT% ^
    --tau %TAU% ^
    --k %K% ^
    --k-anneal ^
    --k-min 2 ^
    --k-max %K% ^
    --soft-topk ^
    --ffn-expansion 2.0 ^
    --optimizer adamw ^
    --lr %LR% ^
    --weight-decay %WD% ^
    --lr-schedule cosine ^
    --warmup-epochs 500 ^
    --label-smoothing 0.05 ^
    --edge-threshold 0.0 ^
    --disable-pred-subgraph

  REM 3) No k annealing (fixed k)
  echo [!DATE! !TIME!] AGNNet/ablate:no_k_anneal ds=%%D
  python main.py ^
    --model AGNNet ^
    --dataset %%D ^
    --epochs %EPOCHS% ^
    --hidden-channels %HIDDEN% ^
    --num-layers %LAYERS% ^
    --dropout %DROPOUT% ^
    --tau %TAU% ^
    --k %K% ^
    --ffn-expansion 2.0 ^
    --optimizer adamw ^
    --lr %LR% ^
    --weight-decay %WD% ^
    --lr-schedule cosine ^
    --warmup-epochs 500 ^
    --label-smoothing 0.05 ^
    --edge-threshold 0.0

  REM 4) No soft top-k (hard cap only)
  echo [!DATE! !TIME!] AGNNet/ablate:no_soft_topk ds=%%D
  python main.py ^
    --model AGNNet ^
    --dataset %%D ^
    --epochs %EPOCHS% ^
    --hidden-channels %HIDDEN% ^
    --num-layers %LAYERS% ^
    --dropout %DROPOUT% ^
    --tau %TAU% ^
    --k %K% ^
    --k-anneal ^
    --k-min 2 ^
    --k-max %K% ^
    --ffn-expansion 2.0 ^
    --optimizer adamw ^
    --lr %LR% ^
    --weight-decay %WD% ^
    --lr-schedule cosine ^
    --warmup-epochs 500 ^
    --label-smoothing 0.05

  REM 5) No self loops
  echo [!DATE! !TIME!] AGNNet/ablate:no_self_loops ds=%%D
  python main.py ^
    --model AGNNet ^
    --dataset %%D ^
    --epochs %EPOCHS% ^
    --hidden-channels %HIDDEN% ^
    --num-layers %LAYERS% ^
    --dropout %DROPOUT% ^
    --tau %TAU% ^
    --k %K% ^
    --k-anneal ^
    --k-min 2 ^
    --k-max %K% ^
    --soft-topk ^
    --ffn-expansion 2.0 ^
    --optimizer adamw ^
    --lr %LR% ^
    --weight-decay %WD% ^
    --lr-schedule cosine ^
    --warmup-epochs 500 ^
    --label-smoothing 0.05 ^
    --no-self-loops

  REM 6) No edge threshold (explicit)
  echo [!DATE! !TIME!] AGNNet/ablate:no_edge_threshold ds=%%D
  python main.py ^
    --model AGNNet ^
    --dataset %%D ^
    --epochs %EPOCHS% ^
    --hidden-channels %HIDDEN% ^
    --num-layers %LAYERS% ^
    --dropout %DROPOUT% ^
    --tau %TAU% ^
    --k %K% ^
    --k-anneal ^
    --k-min 2 ^
    --k-max %K% ^
    --soft-topk ^
    --ffn-expansion 2.0 ^
    --optimizer adamw ^
    --lr %LR% ^
    --weight-decay %WD% ^
    --lr-schedule cosine ^
    --warmup-epochs 500 ^
    --label-smoothing 0.05 ^
    --edge-threshold 0.0
)

echo [DONE] Ablation experiments completed.
powershell -NoProfile -Command "Stop-Transcript" >NUL
endlocal
