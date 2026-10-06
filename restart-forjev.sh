#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ ! -x .venv-forjev/bin/python ]; then
    echo 'Run bash setup-forjev.sh first.' >&2
    exit 1
fi
set -a
source "${FORJEV_ENV_FILE:-.forjev.env}"
set +a
exec .venv-forjev/bin/python -m openjev.forjev_service "${@:-status}"
