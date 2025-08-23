@echo off
setlocal enabledelayedexpansion

REM Usage: run_param_scaling_ova.bat <MODEL> <DATASET> <EPOCHS> <HIDDEN> <NUM_LAYERS> <HEADS>
if "%~6"=="" (
  echo Usage: %~nx0 MODEL DATASET EPOCHS HIDDEN NUM_LAYERS HEADS
  exit /b 2
)

set "MODEL=%~1"
set "DATASET=%~2"
set "EPOCHS=%~3"
set "HIDDEN=%~4"
set "NUM_LAYERS=%~5"
set "HEADS=%~6"

REM Anchor to project root (this script is in experiments\)
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

REM Ensure datasets present (optional, skip if already downloaded)
if not exist "%PROJECT_ROOT%\simple_data" (
  echo [INFO] Datasets folder not found at %PROJECT_ROOT%\simple_data. If needed, run the dataset download step from README.
)

REM Run backend/main.py with OVA-SMOTE and hyperparameter overrides
python "%PROJECT_ROOT%\backend\main.py" ^
  --model "%MODEL%" ^
  --dataset "%DATASET%" ^
  --epochs %EPOCHS% ^
  --hidden-channels %HIDDEN% ^
  --num-layers %NUM_LAYERS% ^
  --heads %HEADS% ^
  --ova-smote

endlocal
