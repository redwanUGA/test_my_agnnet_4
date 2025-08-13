@echo off
setlocal enabledelayedexpansion

REM ssh -p 28082 root@88.196.156.207 -L 8080:localhost:8080

REM Remote configuration (edit placeholders or set env vars)
set "REMOTE_HOST=%REMOTE_HOST%"
if "%REMOTE_HOST%"=="" set "REMOTE_HOST=88.196.156.207"
set "REMOTE_USER=%REMOTE_USER%"
if "%REMOTE_USER%"=="" set "REMOTE_USER=root"
set "REMOTE_PORT=%REMOTE_PORT%"
if "%REMOTE_PORT%"=="" set "REMOTE_PORT=28082"
set "REMOTE_DIR=%REMOTE_DIR%"
if "%REMOTE_DIR%"=="" set "REMOTE_DIR=~/agnnet_remote"

REM If we are executing locally (default), perform remote orchestration. The remote
REM invocation sets RUN_REMOTE=1 to skip this block and run the actual experiments.
if not "%RUN_REMOTE%"=="1" (
  if "%REMOTE_HOST%"=="YOUR.SERVER.IP" (
    echo Please set REMOTE_HOST/REMOTE_USER (or edit placeholders in run_all_experiments.bat).
    exit /b 1
  )
  set "REPO_DIR=%cd%"
  set "REMOTE=%REMOTE_USER%@%REMOTE_HOST%"

  echo [LOCAL] Preparing remote directory at %REMOTE%:%REMOTE_DIR%
  ssh -p %REMOTE_PORT% %REMOTE% "mkdir -p '%REMOTE_DIR%' && [ -n \"%REMOTE_DIR%\" ] && [ \"%REMOTE_DIR%\" != \"/\" ] && rm -rf \"%REMOTE_DIR%\"/*"

  echo [LOCAL] Copying repository to remote...
  scp -P %REMOTE_PORT% -r "%REPO_DIR%\*" "%REMOTE%:%REMOTE_DIR%"

  echo [LOCAL] Running experiments on remote...
  ssh -p %REMOTE_PORT% %REMOTE% "cd '%REMOTE_DIR%' && python3 -m pip install -r requirements.txt && RUN_REMOTE=1 bash run_all_experiments.sh"

  echo [LOCAL] Fetching results back to local machine...
  if not exist logs mkdir logs
  if not exist saved_models mkdir saved_models
  scp -P %REMOTE_PORT% -r "%REMOTE%:%REMOTE_DIR%/logs" .
  scp -P %REMOTE_PORT% -r "%REMOTE%:%REMOTE_DIR%/saved_models" .

  echo [LOCAL] Done.
  exit /b 0
)

REM -----------------------
REM Logging (portable)
REM -----------------------
set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%i"
set "LOG_FILE=%LOG_DIR%\run_all_experiments_%TIMESTAMP%.txt"
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
if "%SEARCH_EPOCHS%"=="" set "SEARCH_EPOCHS=1"
if "%EPOCHS%"=="" set "EPOCHS=2"
set "SAVE_DIR=saved_models"
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM -----------------------
REM 3) Run hyperparameter search (if needed) and training
REM -----------------------
for %%M in (BaselineGCN GraphSAGE GAT TGAT TGN AGNNet) do (
  for %%D in (OGB-Arxiv Reddit TGB-Wiki MOOC) do (
    if not exist "%SAVE_DIR%\%%M_%%D.pt" (
      set NEED_SEARCH=1
    ) else if not exist "%SAVE_DIR%\%%M_%%D_params.json" (
      set NEED_SEARCH=1
    ) else (
      set NEED_SEARCH=0
    )

    if "!NEED_SEARCH!"=="1" (
      echo [!DATE! !TIME!] Hyperparameter search for model=%%M dataset=%%D
      python hyperparameter_search.py --model %%M --dataset %%D --epochs %SEARCH_EPOCHS% --save-dir %SAVE_DIR%
    ) else (
      echo [!DATE! !TIME!] Using existing model/config for model=%%M dataset=%%D
    )

    echo [!DATE! !TIME!] Training model=%%M dataset=%%D
    python main.py --model %%M --dataset %%D --epochs %EPOCHS% --load-model "%SAVE_DIR%\%%M_%%D.pt" --config "%SAVE_DIR%\%%M_%%D_params.json"
  )
)

echo [DONE] All experiments completed.

powershell -NoProfile -Command "Stop-Transcript" >NUL

endlocal
