#!/usr/bin/env bash
# Start the PropSimulator web interface.
#
# Binds 0.0.0.0 rather than 127.0.0.1: in a Codespace or any container the
# port forwarder reaches the process from outside the network namespace, and
# a server bound to loopback is invisible to it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The dev container image provides "python"; a bare Debian or a Mac may only
# have "python3". Pick whichever exists so the script is not tied to one image.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        echo "no python interpreter found on PATH" >&2
        exit 1
    fi
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOG="${PROPSIM_LOG:-/tmp/propsim-server.log}"
PIDFILE="/tmp/propsim-server.${PORT}.pid"

usage() {
    cat <<'USAGE'
Usage: scripts/serve.sh [--background | --stop | --status]

  (no flag)      run in the foreground; Ctrl-C stops it
  --background   start detached, logging to /tmp/propsim-server.log
  --stop         stop a background server on this port
  --status       report whether anything is listening

Environment: PORT (default 8000), HOST (default 0.0.0.0)
USAGE
}

# Whether something already holds the port. Uses Python rather than lsof or
# ss so it works on any image without extra packages.
port_in_use() {
    "$PYTHON" - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    probe.bind(("0.0.0.0", port))
except OSError:
    sys.exit(0)      # in use
finally:
    probe.close()
sys.exit(1)          # free
PY
}

public_url() {
    if [[ -n "${CODESPACE_NAME:-}" && -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]]; then
        echo "https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
    else
        echo "http://localhost:${PORT}"
    fi
}

case "${1:---foreground}" in
    -h|--help)
        usage
        exit 0
        ;;
    --status)
        if port_in_use; then
            echo "PropSimulator is listening on port ${PORT}: $(public_url)"
        else
            echo "nothing is listening on port ${PORT}"
        fi
        exit 0
        ;;
    --stop)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            kill "$(cat "$PIDFILE")"
            rm -f "$PIDFILE"
            echo "stopped the background server on port ${PORT}"
        else
            rm -f "$PIDFILE"
            echo "no background server of ours is running on port ${PORT}"
        fi
        exit 0
        ;;
    --background)
        if port_in_use; then
            # Already up, most likely from a previous attach. Not an error.
            echo "PropSimulator is already running: $(public_url)"
            exit 0
        fi
        setsid nohup "$PYTHON" -m uvicorn api.main:app \
            --host "$HOST" --port "$PORT" >"$LOG" 2>&1 </dev/null &
        server_pid=$!
        echo "$server_pid" >"$PIDFILE"
        # Wait for the port rather than assuming success, so a startup failure
        # is reported here instead of being discovered later as an unexplained
        # connection refused. Waiting out the full timeout on a process that
        # has already died is pointless, so the loop also watches the child.
        for _ in $(seq 1 30); do
            if port_in_use; then
                echo "PropSimulator is running: $(public_url)"
                echo "  log: ${LOG}"
                exit 0
            fi
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        rm -f "$PIDFILE"
        echo "the server did not come up; last lines of ${LOG}:" >&2
        tail -n 20 "$LOG" >&2 || true
        exit 1
        ;;
    --foreground)
        if port_in_use; then
            echo "Port ${PORT} is already in use: $(public_url)" >&2
            echo "Stop it with 'scripts/serve.sh --stop', or set PORT to something else." >&2
            exit 1
        fi
        echo "PropSimulator starting: $(public_url)"
        exec "$PYTHON" -m uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
