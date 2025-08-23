@echo off
setlocal enabledelayedexpansion

REM Local-only execution: remote orchestration removed

REM -----------------------
REM Logging (portable)
REM -----------------------
set "LOG_DIR=..\results\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%i"
set "LOG_FILE=%LOG_DIR%\run_all_experiments_%TIMESTAMP%.txt"
powershell -NoProfile -Command "Start-Transcript -Path '%LOG_FILE%' | Out-Null"

REM -----------------------
REM 1) Download datasets if missing
REM -----------------------
if not exist "..\simple_data" (
  echo [INFO] Downloading datasets to ..\simple_data\ …
  pushd ..
  gdown "https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing" --folder
  popd
)

REM -----------------------
REM 2) Experiment settings
REM -----------------------
REM Default epochs (ensure at least 20)
if "%SEARCH_EPOCHS%"=="" set "SEARCH_EPOCHS=20"
if "%EPOCHS%"=="" set "EPOCHS=20"
set "SAVE_DIR=..\results\saved_models"
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM -----------------------
REM 3) Run hyperparameter search (if needed) and training
REM -----------------------
for %%D in (OGB-Arxiv Reddit TGB-Wiki MOOC) do (
  for %%M in (BaselineGCN GraphSAGE GAT TGAT TGN AGNNet) do (
    if not exist "%SAVE_DIR%\%%M_%%D.pt" (
      set NEED_SEARCH=1
    ) else if not exist "%SAVE_DIR%\%%M_%%D_params.json" (
      set NEED_SEARCH=1
    ) else (
      set NEED_SEARCH=0
    )

    if "!NEED_SEARCH!"=="1" (
      echo [!DATE! !TIME!] Hyperparameter search for model=%%M dataset=%%D
      python ..\backend\hyperparameter_search.py --model %%M --dataset %%D --epochs %SEARCH_EPOCHS% --save-dir %SAVE_DIR%
    ) else (
      echo [!DATE! !TIME!] Using existing model/config for model=%%M dataset=%%D
    )

    echo [!DATE! !TIME!] Training model=%%M dataset=%%D
    python ..\backend\main.py --model %%M --dataset %%D --epochs %EPOCHS% --load-model "%SAVE_DIR%\%%M_%%D.pt" --config "%SAVE_DIR%\%%M_%%D_params.json"
  )
)

echo [DONE] All experiments completed.

powershell -NoProfile -Command "Stop-Transcript" >NUL

endlocal
