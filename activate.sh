#!/bin/bash
# dptb-pilot One-Click Startup Script

############################
# 1. Port occupancy check（50001–50003）
############################
for port in 50001 50002 50003; do
  if netstat -lnt | grep -q "[.:]${port}[[:space:]]"; then
    echo "DeePTB-Pilot-Startup-Script:[ERROR] Port ${port} is already in use. Aborting."
    exit 1
  fi
done

echo "DeePTB-Pilot-Startup-Script:[INFO] Ports 50001–50003 are free."

############################
# 2. Activate mcp-server
############################
# Move to repository to access knowledge database
cd /dptb-pilot

if ! command -v uv >/dev/null 2>&1; then
  echo "DeePTB-Pilot-Startup-Script:[ERROR] uv is required but was not found."
  exit 1
fi

echo "DeePTB-Pilot-Startup-Script:[INFO] Python project source verification:"
uv run --project /dptb-pilot python -c "import dptb_pilot.server.app as app; print(app.__file__); print([route.path for route in app.app.routes if route.path == '/api/execution-config'])"

nohup uv run --project /dptb-pilot dptb-tools > mcp.out 2>&1 &

echo "DeePTB-Pilot-Startup-Script:[INFO] dptb-tools started, waiting for port 50001..."

############################
# 3. Listen on port 50001 (up to 120 seconds)
############################
timeout=240
elapsed=0

while true; do
  if netstat -lnt | grep -q "[.:]50001[[:space:]]"; then
    break
  fi

  if [ "$elapsed" -ge "$timeout" ]; then
    echo "DeePTB-Pilot-Startup-Script:[ERROR] Timeout: port 50001 did not respond within ${timeout}s."
    exit 1
  fi

  if [ $((elapsed % 30)) -eq 0 ] && [ "$elapsed" -ne 0 ]; then
    echo "DeePTB-Pilot-Startup-Script:[INFO] Waiting for port 50001... (${elapsed}s elapsed)"
  fi

  sleep 1
  elapsed=$((elapsed + 1))
done

############################
# 4. Activate API server and Frontend
############################
echo "DeePTB-Pilot-Startup-Script:[INFO] MCP Service is ready. Activate API server and Frontend..."

exec uv run --project /dptb-pilot dptb-pilot
