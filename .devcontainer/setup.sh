#!/usr/bin/env bash
# One-time setup for the Codespace / dev container.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Installing PropSimulator and its dependencies..."
# Editable, so edits to propsim/ take effect without reinstalling.
#   api -> FastAPI + uvicorn for the web interface
#   dev -> pytest, plus SciPy, which is used only as a test oracle
python -m pip install --upgrade pip >/dev/null
python -m pip install -e ".[api,dev]"

echo
echo "Checking the install..."
python - <<'PY'
import propsim
from propsim import PropagationEngine  # noqa: F401
print(f"  propsim {propsim.__version__} imports cleanly")
PY
python -m pytest tests/test_core_physics.py -q

cat <<'MESSAGE'

PropSimulator is ready.

  The web interface starts automatically on port 8000 and Codespaces opens
  it for you. To drive it by hand instead:

    bash scripts/serve.sh          # foreground, Ctrl-C to stop
    bash scripts/serve.sh --stop   # stop a background server

  Command line:

    propsim --tx-lat 40.4 --tx-lon -3.7 --tx-name Madrid \
            --rx-lat 51.5 --rx-lon -0.1 --rx-name London \
            --f107 140 --kp 2 --reliability

  Tests (the full suite takes about 70 seconds):

    pytest -q

MESSAGE
