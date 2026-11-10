#!/usr/bin/env bash
# Снимок окна приложения под виртуальным дисплеем (Xvfb + ImageMagick).
# Использование: tools/screenshot.sh <tab> <out.png> [доп. аргументы app.py]
set -e
cd "$(dirname "$0")/.."
TAB="${1:-training}"; OUT="${2:-screenshots/$TAB.png}"; shift 2 || true
PY=".venv/bin/python"; [ -x "$PY" ] || PY=python3
xvfb-run -a -s "-screen 0 1440x900x24" "$PY" app.py --tab "$TAB" --screenshot "$OUT" --geometry 1400x860+10+10 "$@"
