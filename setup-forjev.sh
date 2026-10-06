#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -m venv .venv-forjev
.venv-forjev/bin/python -m pip install -r requirements-forjev.txt
.venv-forjev/bin/python -m pip install --no-deps -e .
if [ ! -f .forjev.env ]; then
    cp forjev.env.example .forjev.env
    chmod 600 .forjev.env
fi
echo 'Installed. Run: bash restart-forjev.sh start'
