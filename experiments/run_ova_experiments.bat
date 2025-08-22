@echo off
setlocal enabledelayedexpansion

REM Run One-vs-All (OVA) experiments with per-class SMOTE across models and datasets.

REM -----------------------
REM Logging (portable)
REM -----------------------
set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG_FILE=%LOG_DIR%\run_ova_experiments_%TS%.txt"
powershell -NoProfile -Command "Start-Transcript -Path '%LOG_FILE%' | Out-Null"

REM -----------------------
REM 1) Download datasets if missing
REM -----------------------
if not exist "simple_data" (
  echo [INFO] Downloading datasets to simple_data/ …
  gdown "https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing" --folder
)

REM -----------------------
REM 2) Experiment settings
REM -----------------------
if "%EPOCHS%"=="" set "EPOCHS=20"

set MODELS=BaselineGCN GraphSAGE GAT TGAT TGN AGNNet
set DATASETS=OGB-Arxiv Reddit TGB-Wiki MOOC

REM -----------------------
REM 3) Run OVA-SMOTE training for each model/dataset
REM -----------------------
for %%M in (%MODELS%) do (
  for %%D in (%DATASETS%) do (
    echo [!DATE! !TIME!] OVA-SMOTE: model=%%M dataset=%%D
    python ..\backend\main.py --model %%M --dataset %%D --epochs %EPOCHS% --ova-smote
  )
)

echo [DONE] OVA experiments completed.

powershell -NoProfile -Command "Stop-Transcript" >NUL
endlocal
