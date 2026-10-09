"""
過去画像から「人が手前水槽で作業 (給餌など) している」画像を抽出する

段階1 (粗探索): 指定間隔で画像を取得し、YOLO の person が水槽水面 (少し広げた範囲) に重なるものを候補とする
段階2 (詳細化): 候補時刻の前後を 5 秒間隔の全画像で調べ、該当フレームを保存する

  .venv/bin/python find_feeding.py                       # 全日 06-18 時, 1 分間隔
  .venv/bin/python find_feeding.py --days 2026-10-08 --step-min 2

出力: feeding/hits.csv, feeding/images/*.jpg (検出枠付き), feeding/raw/*.jpg (元画像), feeding/contact_*.jpg
"""
import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from dataset_tools import Archive
from foam import FoamSegmenter
from stream_segmentation import DEVICE

SIZE = (1280, 720)


class PersonNearTank:
    def __init__(self, roi_mask, weights="yolo11m-seg.pt", conf=0.35, margin=60):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        # 水面 ROI を広げた範囲 = 水槽の縁 + 周囲 (縁から身を乗り出す/手を伸ばす人を拾う)
        self.zone = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2, margin * 2)))
        self.surface = roi_mask

    def __call__(self, img):
        r = self.model.predict(img, conf=self.conf, classes=[0], device=DEVICE, verbose=False, retina_masks=True)[0]
        best = None
        if r.masks is None:
            return None
        for i, (conf, box) in enumerate(zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist())):
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [r.masks.xy[i].astype(np.int32)], 255)
            area = max(int((m > 0).sum()), 1)
            near = float(((m > 0) & (self.zone > 0)).sum()) / area      # 人の何割が水槽まわりにあるか
            over = int(((m > 0) & (self.surface > 0)).sum())            # 水面上に重なる画素 (手・腕・上体)
            if near < 0.05:
                continue
            score = near + over / 2000.0 + conf
            if best is None or score > best["score"]:
                best = {"score": round(score, 3), "conf": round(conf, 3), "near": round(near, 3),
                        "over_px": over, "box": [int(v) for v in box], "poly": r.masks.xy[i].astype(np.int32)}
        return best


def draw(img, hit, label):
    out = img.copy()
    cv2.polylines(out, [hit["poly"]], True, (0, 255, 255), 2, cv2.LINE_AA)
    x1, y1, x2, y2 = hit["box"]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(paths, dst, cols=5, w=320):
    if not paths:
        return
    tiles = []
    for p in paths:
        im = cv2.imread(p)
        t = cv2.resize(im, (w, w * im.shape[0] // im.shape[1]))
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(dst, cv2.vconcat([cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", help="カンマ区切り (既定: 全日)")
    ap.add_argument("--hours", default="06-18", help="開始-終了 時 (両端含む)")
    ap.add_argument("--step-min", type=int, default=1, help="段階1の間隔 [分]")
    ap.add_argument("--window-sec", type=int, default=60, help="段階2で候補の前後に調べる秒数")
    ap.add_argument("--out", default="feeding")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    for d in ("images", "raw"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    arc = Archive()
    days = args.days.split(",") if args.days else arc.days()
    h0, h1 = map(int, args.hours.split("-"))
    roi = FoamSegmenter("foam_roi.json").roi_mask((SIZE[1], SIZE[0]))
    det = PersonNearTank(roi)

    def fetch(meta):
        for k in range(3):
            try:
                return meta, cv2.resize(arc.image(meta["url"]), SIZE, interpolation=cv2.INTER_AREA)
            except Exception:
                time.sleep(2 * (k + 1))
        return meta, None

    # ---- 一覧取得
    index = {}  # day -> 全画像メタ (時刻順)
    for day in days:
        avail = arc.get("/api/calendar/hours", date=day).json()["hours"]
        index[day] = [dict(m, day=day) for h in avail if h0 <= int(h) <= h1 for m in arc.images(day, h)]

    def sample(lst, step_sec):
        out, nxt = [], None
        for m in lst:
            t = sum(int(x) * k for x, k in zip(m["time"].split(":"), (3600, 60, 1)))
            if nxt is None or t >= nxt:
                out.append(m)
                nxt = t - t % step_sec + step_sec
        return out

    def tsec(m):
        return sum(int(x) * k for x, k in zip(m["time"].split(":"), (3600, 60, 1)))

    coarse = [m for d in days for m in sample(index[d], args.step_min * 60)]
    print(f"[stage1] {len(coarse)} images", flush=True)

    hits, seen = {}, set()

    def run(metas, stage):
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            for n, (meta, img) in enumerate(ex.map(fetch, metas), 1):
                key = (meta["day"], meta["time"])
                seen.add(key)
                if img is not None:
                    hit = det(img)
                    if hit:
                        name = f"{meta['day']}_{meta['time'].replace(':', '')}"
                        cv2.imwrite(os.path.join(args.out, "raw", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"),
                                    draw(img, hit, f"{meta['day']} {meta['time']}"), [cv2.IMWRITE_JPEG_QUALITY, 85])
                        hits[key] = {"name": name, "date": meta["day"], "time": meta["time"], "stage": stage,
                                     **{k: v for k, v in hit.items() if k != "poly"}}
                if n % 100 == 0 or n == len(metas):
                    el = time.time() - t0
                    print(f"[stage{stage}] {n}/{len(metas)} hits={len(hits)} "
                          f"{n/el:.2f} img/s, eta {(len(metas)-n)/(n/el)/60:.0f} min", flush=True)

    run(coarse, 1)

    # ---- 段階2: 候補の前後を全フレームで
    fine = []
    for (day, tm) in sorted(hits):
        t = tsec({"time": tm})
        fine += [m for m in index[day] if abs(tsec(m) - t) <= args.window_sec and (day, m["time"]) not in seen]
    uniq = {(m["day"], m["time"]): m for m in fine}
    fine = [uniq[k] for k in sorted(uniq)]
    print(f"[stage2] {len(fine)} images around {len(hits)} candidates", flush=True)
    if fine:
        run(fine, 2)

    rows = [hits[k] for k in sorted(hits)]
    keys = ["name", "date", "time", "stage", "score", "conf", "near", "over_px", "box"]
    with open(os.path.join(args.out, "hits.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    # 日ごとのコンタクトシート
    for day in days:
        ps = [os.path.join(args.out, "images", r["name"] + ".jpg") for r in rows if r["date"] == day]
        contact_sheet(ps[:: max(1, len(ps) // 40)], os.path.join(args.out, f"contact_{day}.jpg"))
    print(f"done: {len(rows)} frames with a person near the tank -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
