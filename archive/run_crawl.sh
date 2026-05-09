#!/usr/bin/env bash
set -euo pipefail

# T2 spike crawl using wget+warc inside a Docker container, then wacz packaging.
# Why not browsertrix-crawler: its bundled Redis hits a Lua script bug in this
# environment (ERR value is not an integer). WebstaurantStore + DDG HTML are
# both server-rendered, so wget captures them faithfully without JS execution.

cd "$(dirname "$0")"
mkdir -p wacz

VOL_NAME="opencua-crawl-vol"
docker volume create "$VOL_NAME" >/dev/null

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15'

echo "=== Step 1: wget+warc capture ==="
docker run --rm \
  --platform linux/amd64 \
  -v "$VOL_NAME:/data" \
  -v "$PWD/seed_lists:/seeds:ro" \
  alpine:latest \
  sh -c "
    set -e
    apk add --no-cache wget ca-certificates >/dev/null
    cd /data
    rm -f t2_spike.warc.gz t2_spike.warc t2_spike.cdx
    wget \
      --user-agent='$UA' \
      --warc-file=t2_spike \
      --warc-cdx \
      --no-warc-compression \
      --recursive --level=1 \
      --no-parent --no-directories \
      --span-hosts \
      --domains=duckduckgo.com,html.duckduckgo.com,www.webstaurantstore.com,webstaurantstore.com \
      --reject='*.css,*.js,*.png,*.jpg,*.jpeg,*.gif,*.svg,*.ico,*.woff,*.woff2,*.ttf,*.eot,*.webp,*.mp4,*.mp3,*.pdf,*.zip' \
      --tries=2 --timeout=20 --waitretry=3 \
      -o wget.log \
      -i /seeds/spike_seeds.txt || true
    echo
    echo '=== WARC produced ==='
    ls -lh t2_spike.warc t2_spike.cdx 2>/dev/null || ls -lh
    echo
    echo '=== wget log tail ==='
    tail -30 wget.log
  "

echo ""
echo "=== Step 2: convert WARC to WACZ ==="
docker run --rm \
  --platform linux/amd64 \
  -v "$VOL_NAME:/data" \
  python:3.12-slim \
  sh -c "
    set -e
    pip install --quiet wacz
    cd /data
    rm -f t2_spike.wacz
    wacz create \
      --output t2_spike.wacz \
      --text \
      --detect-pages \
      t2_spike.warc
    ls -lh t2_spike.wacz
  "

echo ""
echo "=== Step 3: copy WACZ out ==="
docker run --rm \
  --platform linux/amd64 \
  -v "$VOL_NAME:/data:ro" \
  -v "$PWD/wacz:/out" \
  alpine \
  sh -c 'cp /data/t2_spike.wacz /out/ && ls -lh /out/'

echo ""
echo "WACZ written to: archive/wacz/t2_spike.wacz"
