#!/usr/bin/env bash
# Installs (or removes) the aimemory MCP server into every supported AI client
# found on this machine:
#   - Codex            ~/.codex/config.toml
#   - Claude Code      ~/.claude.json
#   - Claude Desktop   ~/Library/Application Support/Claude/claude_desktop_config.json
#   - Hermes           ~/.hermes/config.yaml
#
# Pi is intentionally skipped — it has no built-in MCP support by design.
#
# Safe to re-run: edits are idempotent and every existing config is backed up
# (with a .bak-<timestamp> suffix) before it is touched.
#
# Usage:
#   scripts/setup-mcp.sh             # install / refresh
#   scripts/setup-mcp.sh --remove    # uninstall from all clients
#   scripts/setup-mcp.sh --url http://host:8420/mcp
set -euo pipefail

MCP_URL="http://localhost:8420/mcp"
MODE="install"

usage() {
  cat <<EOF
Usage: $0 [--remove] [--url URL]

  --remove     Remove the aimemory MCP server from all clients.
  --url URL    MCP endpoint (default: $MCP_URL)
  -h, --help   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove) MODE="remove"; shift ;;
    --url)    MCP_URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

NPX="$(command -v npx || true)"

echo "=== aimemory MCP setup ==="
echo "mode:    $MODE"
echo "endpoint: $MCP_URL"
if [[ -n "$NPX" ]]; then
  echo "npx:     $NPX (for Claude Desktop bridge)"
else
  echo "npx:     not found (Claude Desktop step will be skipped)"
fi

HEALTH_URL="${MCP_URL%/mcp}/health"
if curl -sf --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "server:  UP"
else
  echo "server:  not reachable at $HEALTH_URL"
  echo "         (start it first with scripts/setup-macos.sh — config will still be written)"
fi
echo

python3 - "$MODE" "$MCP_URL" "$NPX" <<'PYEOF'
import json, os, sys, pathlib, datetime, shutil

mode, mcp_url, npx_path = sys.argv[1], sys.argv[2], sys.argv[3]
SERVER = "aimemory"
results = []

def record(label, status, detail=""):
    results.append((label, status, detail))
    tag = "ok" if status == "ok" else ("skip" if status == "skip" else "ERR")
    line = f"  [{tag:>4}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)

def backup(path):
    if not os.path.exists(path):
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cand = f"{path}.bak-{ts}"
    i = 0
    while os.path.exists(cand):
        i += 1
        cand = f"{path}.bak-{ts}.{i}"
    shutil.copy2(path, cand)
    return cand

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    prev_mode = os.stat(path).st_mode if os.path.exists(path) else None
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    if prev_mode is not None:
        os.chmod(path, prev_mode)

# ---------- Claude Code (~/.claude.json) ----------
def claude_code():
    p = os.path.expanduser("~/.claude.json")
    if not os.path.exists(p):
        record("claude-code", "skip", f"{p} not found")
        return
    d = load_json(p)
    servers = d.setdefault("mcpServers", {})
    if mode == "remove":
        if SERVER in servers:
            b = backup(p)
            del servers[SERVER]
            save_json(p, d)
            record("claude-code", "ok", f"removed (backup {b})")
        else:
            record("claude-code", "skip", "aimemory not present")
    else:
        want = {"type": "http", "url": mcp_url}
        if servers.get(SERVER) == want:
            record("claude-code", "ok", "already configured")
        else:
            b = backup(p)
            servers[SERVER] = want
            save_json(p, d)
            record("claude-code", "ok", f"set http server (backup {b})")

# ---------- Claude Desktop ----------
def claude_desktop():
    p = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
    if mode != "remove" and not npx_path:
        record("claude-desktop", "skip", "npx not found (desktop needs npx for mcp-remote bridge)")
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = load_json(p) if os.path.exists(p) else {}
    servers = d.setdefault("mcpServers", {})
    if mode == "remove":
        if SERVER in servers:
            b = backup(p) if os.path.exists(p) else None
            del servers[SERVER]
            save_json(p, d)
            record("claude-desktop", "ok", f"removed (backup {b})" if b else "removed")
        else:
            record("claude-desktop", "skip", "aimemory not present")
    else:
        want = {"command": npx_path, "args": ["-y", "mcp-remote", mcp_url]}
        if servers.get(SERVER) == want:
            record("claude-desktop", "ok", "already configured")
        else:
            b = backup(p) if os.path.exists(p) else None
            servers[SERVER] = want
            save_json(p, d)
            record("claude-desktop", "ok", f"set mcp-remote bridge (backup {b})" if b else "set mcp-remote bridge")

# ---------- Codex (~/.codex/config.toml) ----------
def _toml_find_table(lines, header):
    for i, l in enumerate(lines):
        if l.strip() == header:
            return i
    return None

def _toml_table_end(lines, start):
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("["):
            return j
    return len(lines)

def codex():
    p = os.path.expanduser("~/.codex/config.toml")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    header = f"[mcp_servers.{SERVER}]"
    lines = pathlib.Path(p).read_text().splitlines() if os.path.exists(p) else []
    start = _toml_find_table(lines, header)
    if mode == "remove":
        if start is None:
            record("codex", "skip", "aimemory not present")
            return
        b = backup(p)
        end = _toml_table_end(lines, start)
        del lines[start:end]
        pathlib.Path(p).write_text("\n".join(lines).rstrip() + "\n")
        record("codex", "ok", f"removed (backup {b})")
        return
    block = [header, f'url = "{mcp_url}"']
    if start is not None:
        end = _toml_table_end(lines, start)
        if lines[start:end] == block:
            record("codex", "ok", "already configured")
            return
        b = backup(p)
        new_lines = lines[:start] + block + lines[end:]
    else:
        b = backup(p) if os.path.exists(p) else None
        if lines and lines[-1].strip() != "":
            lines.append("")
        new_lines = lines + block
    pathlib.Path(p).write_text("\n".join(new_lines) + "\n")
    record("codex", "ok", f"set http server (backup {b})" if b else "set http server")

# ---------- Hermes (~/.hermes/config.yaml) ----------
def _yaml_find_top_key(lines, key):
    for i, l in enumerate(lines):
        if l.startswith(key + ":"):
            return i
    return None

def _yaml_block_end(lines, start):
    """End (exclusive) of a top-level mapping's children: first non-blank
    line at column 0 after `start`."""
    for j in range(start + 1, len(lines)):
        s = lines[j]
        if s.strip() != "" and not s[0].isspace():
            return j
    return len(lines)

def _yaml_strip_child(block, name, indent):
    """Remove a child mapping `name:` (at `indent` spaces) and its deeper
    children from a list of lines; return the filtered list."""
    out, i, n = [], 0, len(block)
    needle = " " * indent + name + ":"
    while i < n:
        if block[i].rstrip() == needle:
            i += 1
            child = " " * (indent + 2)
            while i < n and block[i].startswith(child) and block[i].strip() != "":
                i += 1
            continue
        out.append(block[i])
        i += 1
    return out

def _yaml_child_url(block, name, indent):
    """Return the `url:` value of a child mapping, or None if absent."""
    needle = " " * indent + name + ":"
    for i, l in enumerate(block):
        if l.rstrip() == needle:
            for j in range(i + 1, len(block)):
                s = block[j]
                if s.startswith(" " * (indent + 2)) and s.strip() != "":
                    if s.strip().startswith("url:"):
                        return s.strip()[len("url:"):].strip().strip('"')
                else:
                    break
            return None
    return None

def hermes():
    p = os.path.expanduser("~/.hermes/config.yaml")
    if not os.path.exists(p):
        record("hermes", "skip", f"{p} not found (hermes not installed?)")
        return
    lines = pathlib.Path(p).read_text().splitlines()
    ms_idx = _yaml_find_top_key(lines, "mcp_servers")
    if mode == "remove":
        if ms_idx is None:
            record("hermes", "skip", "no mcp_servers section")
            return
        end = _yaml_block_end(lines, ms_idx)
        block = lines[ms_idx + 1:end]
        if not any(l.rstrip() == "  aimemory:" for l in block):
            record("hermes", "skip", "aimemory not present")
            return
        b = backup(p)
        stripped = _yaml_strip_child(block, "aimemory", 2)
        if all(l.strip() == "" for l in stripped):
            final = lines[:ms_idx] + lines[end:]
        else:
            final = lines[:ms_idx + 1] + stripped + lines[end:]
        pathlib.Path(p).write_text("\n".join(final).rstrip() + "\n")
        record("hermes", "ok", f"removed (backup {b})")
        return
    aim_block = ["  aimemory:", f'    url: "{mcp_url}"']
    if ms_idx is None:
        b = backup(p)
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append("mcp_servers:")
        lines.extend(aim_block)
        pathlib.Path(p).write_text("\n".join(lines) + "\n")
        record("hermes", "ok", f"added mcp_servers.aimemory (backup {b})")
        return
    end = _yaml_block_end(lines, ms_idx)
    if _yaml_child_url(lines[ms_idx + 1:end], "aimemory", 2) == mcp_url:
        record("hermes", "ok", "already configured")
        return
    b = backup(p)
    block = _yaml_strip_child(lines[ms_idx + 1:end], "aimemory", 2)
    final = lines[:ms_idx + 1] + aim_block + block + lines[end:]
    pathlib.Path(p).write_text("\n".join(final) + "\n")
    record("hermes", "ok", f"set mcp_servers.aimemory (backup {b})")

for fn in (claude_code, claude_desktop, codex, hermes):
    try:
        fn()
    except Exception as e:
        record(fn.__name__, "error", f"{type(e).__name__}: {e}")

errors = [r for r in results if r[1] == "error"]
print()
if errors:
    print(f"Completed with {len(errors)} error(s). See above.")
    sys.exit(1)
if mode == "remove":
    print("Done. Restart each client to drop the removed server:")
    print("  claude code    — restart your CLI session")
    print("  claude desktop — quit & reopen the app")
    print("  codex          — start `codex` (config is read on launch)")
    print("  hermes         — auto-reloads, or run /reload-mcp")
else:
    print("Done. Restart each client to pick up the new MCP server:")
    print("  claude code    — restart your CLI session")
    print("  claude desktop — quit & reopen the app")
    print("  codex          — start `codex` (config is read on launch)")
    print("  hermes         — auto-reloads, or run /reload-mcp")
    print("  (pi has no built-in MCP support — skipped by design)")
PYEOF
