#!/bin/sh
# backup_archive.py を完了するまで繰り返す (カメラに繋がらない間は 5 分おきに再試行)
cd "$(dirname "$0")"
while :; do
  .venv/bin/python -u backup_archive.py >> backup_archive.log 2>&1
  tail -1 backup_archive.log | grep -q '^done' && break
  echo "$(date '+%F %T') retry in 5 min" >> backup_archive.log
  sleep 300
done
