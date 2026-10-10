"""
PiCam アーカイブの元画像 (1920x1080, 再エンコードなし) を手元に保存する

カメラ側の保持期間は約 7 日 (古い日から消える) なので、優先度順に取得する:
  1. feeding/events.csv の給餌・作業イベント (A/B) の前後 (既定: 10 分前〜90 分後。給餌後の濁りが戻るまで)
  2. 各日の日中 (06-18 時)、古い日から
  3. 残りの時間帯 (夜間)、古い日から

  .venv/bin/python backup_archive.py                 # 全部 (途中で止めても再実行で続きから)
  .venv/bin/python backup_archive.py --no-night      # 夜間を除く
  ./backup_loop.sh                                   # 繋がらない間は 5 分おきに再試行し、終わるまで続ける

出力: archive/YYYY-MM-DD/HH-MM-SS.jpg (取得済みのファイルはスキップ)
"""
import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dataset_tools import Archive


def tsec(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="archive")
    ap.add_argument("--events", default="feeding/events.csv")
    ap.add_argument("--before-min", type=int, default=10)
    ap.add_argument("--after-min", type=int, default=90)
    ap.add_argument("--day-hours", default="06-18")
    ap.add_argument("--no-night", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    arc = Archive()
    days = arc.days()
    index = {}  # day -> 全画像メタ
    for day in days:
        hours = arc.get("/api/calendar/hours", date=day).json()["hours"]
        index[day] = [dict(m, day=day) for h in hours for m in arc.images(day, h)]
    print(f"archive: {', '.join(f'{d} ({len(index[d])})' for d in days)}", flush=True)

    # ---- 優先度ごとの取得リスト
    windows = []
    if os.path.exists(args.events):
        with open(args.events, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["category"].startswith(("A_", "B_")) and r["date"] in index:
                    windows.append((r["date"], tsec(r["start"]) - args.before_min * 60,
                                    tsec(r["end"]) + args.after_min * 60))
    h0, h1 = map(int, args.day_hours.split("-"))
    tiers = [
        ("events", [m for d in days for m in index[d]
                    if any(d == wd and a <= tsec(m["time"]) <= b for wd, a, b in windows)]),
        ("daytime", [m for d in days for m in index[d] if h0 <= int(m["time"][:2]) <= h1]),
    ]
    if not args.no_night:
        tiers.append(("night", [m for d in days for m in index[d]]))

    def path(m):
        return os.path.join(args.out, m["day"], m["time"].replace(":", "-") + ".jpg")

    def fetch(m):
        p = path(m)
        for k in range(4):
            try:
                data = arc.get(m["url"]).content
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p + ".part", "wb") as f:
                    f.write(data)
                os.replace(p + ".part", p)
                return len(data)
            except Exception:
                time.sleep(2 * (k + 1))
        return -1

    for name, metas in tiers:
        todo = [m for m in metas if not os.path.exists(path(m))]
        todo = list({path(m): m for m in todo}.values())
        print(f"[{name}] {len(metas)} images, {len(todo)} to fetch", flush=True)
        t0, nbytes, fail, streak = time.time(), 0, 0, 0
        with ThreadPoolExecutor(args.workers) as ex:
            for n, size in enumerate(ex.map(fetch, todo), 1):
                if size < 0:
                    fail += 1
                    streak += 1
                    if streak >= 20:  # カメラに繋がらない。続きは再実行で (取得済みはスキップされる)
                        print(f"[{name}] {streak} consecutive failures, camera unreachable -> exit", flush=True)
                        os._exit(2)
                else:
                    nbytes += size
                    streak = 0
                if n % 500 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"[{name}] {n}/{len(todo)} {n/el:.2f} img/s {nbytes/1e9:.2f} GB fail={fail} "
                          f"eta {(len(todo)-n)/(n/el)/60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
