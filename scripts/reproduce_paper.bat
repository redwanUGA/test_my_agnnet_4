@echo off
setlocal ENABLEDELAYEDEXPANSION

REM One-command entrypoint to reproduce paper artifacts on Windows.
REM Steps: (a) ensure data present; (b) run experiments; (c) write logs to results\; (d) produce results\summary.json.

set ROOT=%~dp0..
cd /d %ROOT%

REM Prepare output directories
if not exist results mkdir results
if not exist results\logs mkdir results\logs
if not exist results\configs mkdir results\configs

REM Timestamp for log file
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do (
  set mm=%%a
  set dd=%%b
  set yyyy=%%c
)
set hh=%time:~0,2%
set mn=%time:~3,2%
set ss=%time:~6,2%
set TS=%yyyy%%mm%%dd%_%hh%%mn%%ss%
set LOG=results\logs\reproduce_%TS%.txt

REM Ensure datasets exist (simple_data folder). If missing, attempt to download via gdown.
if not exist simple_data (
  echo [reproduce] simple_data\ not found. Attempting download using gdown...>> %LOG%
  echo Please see DOWNLOAD_INSTRUCTIONS.md for details.>> %LOG%
  python -m pip install --quiet gdown >> %LOG% 2>&1
  python -m gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder --output simple_data >> %LOG% 2>&1
)

REM Run experiments (adjust as needed to match Figures/Tables mapping in README)
if exist experiments\run_all_experiments_agn_only.bat (
  call experiments\run_all_experiments_agn_only.bat >> %LOG% 2>&1
) else if exist experiments\run_all_experiments.bat (
  call experiments\run_all_experiments.bat >> %LOG% 2>&1
) else (
  echo No comprehensive experiments batch script found. Running a default baseline sanity.>> %LOG%
  python backend\main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 5 >> %LOG% 2>&1
)

REM Produce summary.json from logs
python scripts\make_summary.py results\logs results\summary.json >> %LOG% 2>&1

echo Reproduction complete. See %LOG% and results\summary.json
exit /b 0
