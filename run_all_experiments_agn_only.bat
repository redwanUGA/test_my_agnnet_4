@echo off
setlocal enabledelayedexpansion

REM Runs ONLY AGNNet across datasets with recommended modifications and k/tau sweeps.

set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG_FILE=%LOG_DIR%\run_all_experiments_agn_only_%TS%.txt"
powershell -NoProfile -Command "Start-Transcript -Path '%LOG_FILE%' | Out-Null"

REM Ensure datasets
if not exist "simple_data" (
  echo [INFO] Downloading datasets to simple_data/ …
  gdown "https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing" --folder
)

if "%EPOCHS%"=="" set "EPOCHS=50"
set "SAVE_DIR=saved_models_agn"
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

set DATASETS=OGB-Arxiv Reddit TGB-Wiki MOOC
set KS=2 4 8 16 32
set TAUS=0.7 0.9 1.2

for %%D in (%DATASETS%) do (
  for %%K in (%KS%) do (
    for %%T in (%TAUS%) do (
      echo [!DATE! !TIME!] AGNNet ds=%%D k=%%K tau=%%T
      python main.py ^
        --model AGNNet ^
        --dataset %%D ^
        --epochs %EPOCHS% ^
        --hidden-channels 128 ^
        --num-layers 3 ^
        --dropout 0.25 ^
        --tau %%T ^
        --k %%K ^
        --k-anneal ^
        --k-min 2 ^
        --k-max %%K ^
        --soft-topk ^
        --ffn-expansion 2.0 ^
        --optimizer adamw ^
        --lr 0.003 ^
        --weight-decay 0.0005 ^
        --lr-schedule cosine ^
        --warmup-epochs 500 ^
        --label-smoothing 0.05
    )
  )
  REM Ablation: disable predictive-subgraph once per dataset
  echo [!DATE! !TIME!] AGNNet ds=%%D (ablation)
  python main.py ^
    --model AGNNet ^
    --dataset %%D ^
    --epochs %EPOCHS% ^
    --hidden-channels 128 ^
    --num-layers 3 ^
    --dropout 0.25 ^
    --tau 0.9 ^
    --k 8 ^
    --k-anneal ^
    --k-min 2 ^
    --k-max 8 ^
    --soft-topk ^
    --ffn-expansion 2.0 ^
    --optimizer adamw ^
    --lr 0.003 ^
    --weight-decay 0.0005 ^
    --lr-schedule cosine ^
    --warmup-epochs 500 ^
    --label-smoothing 0.05 ^
    --disable-pred-subgraph
)

echo [DONE] AGNNet-only experiments completed.

powershell -NoProfile -Command "Stop-Transcript" >NUL
endlocal
