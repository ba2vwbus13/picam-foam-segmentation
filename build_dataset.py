"""
泡セグメンテーション学習用データセットを自動作成する (疑似ラベル)

条件: 10:00-15:00 / 10 分間隔 / 参照画角に位置合わせできた画像 / 泡の割合 15-80%
ラベル: 0=水面外, 1=水面 (泡以外), 2=泡   ※既存の画像処理 (foam.py) の出力をそのまま使う
分割: 日単位 (最終日=test, その前日=val, 残り=train)

  .venv/bin/python build_dataset.py
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from dataset_tools import Archive, ViewAligner
from foam import FoamSegmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--ref", default="samples/full/f1.jpg", help="ROI を定義した参照フレーム")
    ap.add_argument("--roi", default="foam_roi.json")
    ap.add_argument("--hours", default="10,11,12,13,14")
    ap.add_argument("--step-min", type=int, default=10)
    ap.add_argument("--min-inliers", type=int, default=200)
    ap.add_argument("--ratio-range", default="0.15,0.80")
    args = ap.parse_args()

    lo, hi = map(float, args.ratio_range.split(","))
    for d in ("images", "masks", "overlays"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    ref = cv2.imread(args.ref)
    foam = FoamSegmenter(args.roi)
    aligner = ViewAligner(ref, foam.roi_mask(ref.shape))
    arc = Archive()
    days = arc.days()
    split = {d: "train" for d in days}
    split[days[-1]], split[days[-2]] = "test", "val"

    # 候補: 各時間の 00,10,20... 分に最も近い画像
    cands = []
    for day in days:
        for hour in args.hours.split(","):
            ims = arc.images(day, hour)
            for m in range(0, 60, args.step_min):
                target = f"{hour}:{m:02d}:00"
                near = [i for i in ims if i["time"] >= target]
                if near:
                    cands.append((day, near[0]))
    print(f"candidates: {len(cands)}")

    def fetch(c):
        try:
            return c, arc.image(c[1]["url"])
        except Exception as ex:
            print("fetch failed", c[1]["url"], ex)
            return c, None

    rows = []
    with ThreadPoolExecutor(6) as ex:
        for (day, meta), bgr in ex.map(fetch, cands):
            name = f"{day}_{meta['time'].replace(':', '')}"
            row = {"name": name, "date": day, "time": meta["time"], "split": split[day], "status": ""}
            if bgr is None:
                row["status"] = "fetch_failed"
                rows.append(row)
                continue
            img, roi, info = aligner(bgr)
            row.update(inliers=info["inliers"], shift_px=round(info.get("shift_px", -1), 1))
            if roi is None or info["inliers"] < args.min_inliers or info.get("roi_kept", 0) < 0.95:
                row["status"] = "align_failed"
                rows.append(row)
                continue
            fm, roi, ratio, t, _ = foam.segment(img, roi)
            row.update(foam_ratio=round(ratio, 4), L_threshold=t)
            if not lo <= ratio <= hi:
                row["status"] = "ratio_out_of_range"
                rows.append(row)
                continue
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[roi > 0] = 1
            mask[fm > 0] = 2
            cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(args.out, "masks", name + ".png"), mask)
            ov = img.copy()
            ov[mask == 1] = (0.6 * ov[mask == 1] + 0.4 * np.array([255, 255, 0])).astype(np.uint8)
            ov[mask == 2] = (0.4 * ov[mask == 2] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out, "overlays", name + ".jpg"), cv2.resize(ov, (640, 360)))
            row["status"] = "ok"
            rows.append(row)

    keys = ["name", "date", "time", "split", "status", "inliers", "shift_px", "foam_ratio", "L_threshold"]
    with open(os.path.join(args.out, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["name"]))
    ok = [r for r in rows if r["status"] == "ok"]
    print("status:", {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})})
    print("split:", {s: sum(r["split"] == s for r in ok) for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
