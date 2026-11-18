#!/bin/sh
# Start ML Atlas on this computer (macOS and Linux). Needs Python 3.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then exec python3 serve.py "$@"; fi
if command -v python >/dev/null 2>&1; then exec python serve.py "$@"; fi
echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and run this again."
