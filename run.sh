#!/usr/bin/env bash
# Запуск приложения (Linux/macOS). Нужен python3 с tkinter.
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
$PY -c "import numpy, matplotlib" 2>/dev/null || $PY -m pip install -r requirements.txt
exec $PY app.py "$@"
