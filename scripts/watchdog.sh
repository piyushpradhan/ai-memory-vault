#!/usr/bin/env bash
# ponytail: server can hang (alive but unresponsive) without crashing, which
# launchd's KeepAlive won't catch. Kill it on a failed health check so
# KeepAlive relaunches a fresh process.
curl -sf --max-time 10 http://127.0.0.1:8420/health >/dev/null && exit 0
pkill -9 -f ai-memory/.venv/bin/aimemory-server
