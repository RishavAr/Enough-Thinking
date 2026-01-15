#!/usr/bin/env bash
set -euo pipefail

# Create DB
python -m enough_thinking.mcp.make_db

# Start server in background
uvicorn enough_thinking.mcp.server:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
sleep 2

# Calendar query
python -m enough_thinking.mcp.agent_client --server http://127.0.0.1:8000   --question "Look at my calendar and tell me my next 3 events and whether any overlap."

# Expense summary
python -m enough_thinking.mcp.agent_client --server http://127.0.0.1:8000   --question "From my expenses table, summarize total spend per category."

# Stop server
kill $SERVER_PID
echo "✅ MCP demo complete"
