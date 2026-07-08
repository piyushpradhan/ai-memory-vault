#!/usr/bin/env bash
set -e

AGENT_DIR="$HOME/Library/LaunchAgents"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGS_DIR="$PROJ_DIR/logs"

echo "=== ai-memory macOS setup ==="

# Create logs directory
mkdir -p "$LOGS_DIR"

# Copy launchd plists
cp "$PROJ_DIR/scripts/com.aimemory.server.plist" "$AGENT_DIR/"
cp "$PROJ_DIR/scripts/com.aimemory.reindex.plist" "$AGENT_DIR/"

# Unload if already loaded
launchctl unload "$AGENT_DIR/com.aimemory.server.plist" 2>/dev/null || true
launchctl unload "$AGENT_DIR/com.aimemory.reindex.plist" 2>/dev/null || true

# Load agents
launchctl load "$AGENT_DIR/com.aimemory.server.plist"
launchctl load "$AGENT_DIR/com.aimemory.reindex.plist"

echo "Server started (port 8420) — will auto-restart on crash/reboot."
echo "Re-index runs every 5 min — catches edits made in Obsidian."
echo ""
echo "Check status:  curl http://localhost:8420/health"
echo "View logs:     tail -f $LOGS_DIR/server.log"
echo ""
echo "To stop:"
echo "  launchctl unload $AGENT_DIR/com.aimemory.server.plist"
echo "  launchctl unload $AGENT_DIR/com.aimemory.reindex.plist"
