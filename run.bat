@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import numpy, matplotlib" 2>nul || (
  echo Устанавливаю зависимости...
  %PY% -m pip install -r requirements.txt
)
%PY% app.py
if errorlevel 1 pause
