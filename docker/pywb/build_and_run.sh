#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p wacz

if [[ ! -f wacz/t4_spike.wacz ]]; then
  if [[ -f ../../archive/wacz/t4_spike.wacz ]]; then
    cp ../../archive/wacz/t4_spike.wacz wacz/
  else
    echo "ERROR: no t4_spike.wacz found. Run archive/run_crawl.sh first." >&2
    exit 1
  fi
fi

docker build -t opencua-pywb:t4 .

docker rm -f opencua-pywb 2>/dev/null || true
docker run -d --name opencua-pywb -p 8080:8080 opencua-pywb:t4

sleep 3
docker logs opencua-pywb | tail -20

echo ""
echo "pywb proxy listening on http://localhost:8080"
echo "Test a replayed page:  http://localhost:8080/t4/2024/https://www.thriftbooks.com/"
echo "Or use as HTTP proxy:  curl -x http://localhost:8080 https://www.thriftbooks.com/"
