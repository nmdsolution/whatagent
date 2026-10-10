#!/usr/bin/env bash
# Runs ON THE SERVER, from the deploy directory, once the workflow has
# copied the code and written .env there. Builds and (re)starts the bot.
#
# Environment: BOT_PORT (host port the bot is published on, default 8100).
set -euo pipefail

export BOT_PORT="${BOT_PORT:-8100}"
COMPOSE="docker compose -f docker-compose.prod.yml"

echo "=========================================="
echo "Deploying whatagent — $(date)"
echo "=========================================="

if [ ! -s .env ]; then
  echo "ERROR: .env is missing in $(pwd)."
  exit 1
fi

# Refuse to start if another program already owns the port (the bot's own
# container is fine: compose replaces it).
if [ -z "$($COMPOSE ps -q bot)" ] && ss -ltn | awk '{print $4}' | grep -q ":${BOT_PORT}\$"; then
  echo "ERROR: port ${BOT_PORT} is already in use on this server. Change BOT_PORT in the workflow."
  exit 1
fi

$COMPOSE build bot
$COMPOSE up -d --remove-orphans

for attempt in $(seq 1 20); do
  container_id="$($COMPOSE ps -q bot)"
  status="missing"
  if [ -n "$container_id" ]; then
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id")"
  fi
  if [ "$status" = "healthy" ]; then
    $COMPOSE ps
    docker image prune -f
    echo "=========================================="
    echo "Deployment completed! Bot listening on port ${BOT_PORT}."
    echo "=========================================="
    exit 0
  fi
  echo "Waiting for bot ($status), attempt $attempt/20..."
  sleep 5
done

echo "ERROR: bot did not become healthy in time. Last logs:"
$COMPOSE logs --tail=100 bot
$COMPOSE ps
exit 1
