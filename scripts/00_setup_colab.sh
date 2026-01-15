#!/usr/bin/env bash
set -euo pipefail
pip -q install -r requirements.txt
pip -q install -e .
pip -q install fastapi uvicorn pydantic requests
echo "✅ setup done"
