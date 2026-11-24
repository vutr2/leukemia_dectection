@echo off
REM One-time setup on Windows. Creates .venv and installs pinned packages.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3.12 -m venv .venv || py -3 -m venv .venv
) else (
  python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo ERROR: could not create .venv. Install Python 3.12 from python.org and run this again.
  pause
  exit /b 1
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-notebooks-lock.txt
.venv\Scripts\python run_local.py notebooks
.venv\Scripts\python run_local.py check
echo.
echo Setup complete. Next:  .venv\Scripts\python run_local.py v27
pause
