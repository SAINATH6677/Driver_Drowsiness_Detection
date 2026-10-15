#!/usr/bin/env bash
# DriveGuard AI Execution Wrapper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/drowsiness_yawn.py" "$@"
