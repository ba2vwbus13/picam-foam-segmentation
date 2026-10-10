"""
「水面外 / 水 / 泡 / 魚群」4 クラスの学習用データセットを作成する (疑似ラベル + 手修正ラベル)

画像の出どころ (いずれも 1280x720 にそろえる):
  - dataset/images/       泡モデル用の通常時 204 枚
  - feeding/raw/          水槽のそばに人がいる 1,563 枚 (給餌中を含む)
  - archive/YYYY-MM-DD/   元画像のバックアップ (backup_archive.py)。給餌イベント前後を --archive-step 秒ごとに採用
疑似ラベル:
  - 水面: foam_roi.json の楕円を ViewAligner で各画像の画角に合わせたもの
  - 泡: foam.py の閾値処理 (明るさの下限 --foam-l-min を追加して濁った水の誤検出を抑える)
  - 魚群: fish.py の閾値処理
  - 人 (YOLO) は水面外扱い
手修正ラベル: labels/manual/<name>.png があれば疑似ラベルの代わりに使う (値は 0-3)
分割: 日単位 (--val-days / --test-days、残りが train)

  .venv/bin/python build_dataset_fish.py
出力: dataset_fish/{images,masks,overlays}/, dataset_fish/meta.csv
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from dataset_tools import ViewAligner
from fish import FishSchoolSegmenter

CLASSES = ["background", "water", "foam", "fish"]
COLORS = {1: (255, 255, 0), 2: (255, 0, 255), 3: (0, 255, 0)}
SIZE = (1280, 720)


def tsec(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def collect(args):
    """(name, path, source) の一覧。同じ時刻は先に見つかったものを使う"""
    items = {}
    for p in sorted(glob.glob("dataset/images/*.jpg")):
        items.setdefault(os.path.basename(p)[:-4], (p, "dataset"))
    for p in sorted(glob.glob("feeding/raw/*.jpg")):
        items.setdefault(os.path.basename(p)[:-4], (p, "feeding"))
    # アーカイブ元画像: 給餌・作業イベント (A/B) の前後を間引いて採用
    windows = []
    with open(args.events, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["category"].startswith(("A_", "B_")):
                windows.append((r["date"], tsec(r["start"]) - 600, tsec(r["end"]) + 5400))
    for day_dir in sorted(glob.glob("archive/*")):
        day = os.path.basename(day_dir)
        last = -10 ** 9
        for p in sorted(glob.glob(os.path.join(day_dir, "*.jpg"))):
            t = tsec(os.path.basename(p)[:-4].replace("-", ":"))
            if t - last < args.archive_step or not any(d == day and a <= t <= b for d, a, b in windows):
                continue
            last = t
            items.setdefault(f"{day}_{os.path.basename(p)[:-4].replace('-', '')}", (p, "archive"))
    return [(n, p, s) for n, (p, s) in sorted(items.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_fish")
    ap.add_argument("--ref", default="samples/full/f1.jpg", help="ROI を定義した参照フレーム")
    ap.add_argument("--events", default="feeding/events.csv")
    ap.add_argument("--archive-step", type=int, default=60, help="アーカイブ画像の採用間隔 [秒]")
    ap.add_argument("--manual", default="labels/manual")
    ap.add_argument("--foam-l-min", type=int, default=120)
    ap.add_argument("--min-inliers", type=int, default=200)
    ap.add_argument("--val-days", default="2026-10-08")
    ap.add_argument("--test-days", default="2026-10-09")
    args = ap.parse_args()

    for d in ("images", "masks", "overlays"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    seg = FishSchoolSegmenter(foam_l_min=args.foam_l_min)
    ref = cv2.imread(args.ref)
    aligner = ViewAligner(ref, seg.foam.roi_mask(ref.shape))
    val, test = set(args.val_days.split(",")), set(args.test_days.split(","))

    meta_path = os.path.join(args.out, "meta.csv")
    done = {}
    if os.path.exists(meta_path):  # 再実行時は作成済みの画像を再利用 (手修正ラベルは毎回反映)
        done = {r["name"]: r for r in csv.DictReader(open(meta_path))}

    items = collect(args)
    print(f"{len(items)} images: " + str({s: sum(i[2] == s for i in items) for s in ("dataset", "feeding", "archive")}),
          flush=True)
    rows = []
    for k, (name, path, source) in enumerate(items, 1):
        day = name[:10]
        split = "test" if day in test else "val" if day in val else "train"
        manual = os.path.join(args.manual, name + ".png")
        row = dict(done.get(name) or {"name": name, "date": day, "time": name[11:], "source": source})
        row["split"] = split
        mpath = os.path.join(args.out, "masks", name + ".png")
        if name in done and row.get("status") == "ok" and os.path.exists(mpath) and not os.path.exists(manual):
            rows.append(row)
            continue
        img, roi, info = aligner(cv2.imread(path))
        row.update(inliers=info["inliers"], shift_px=round(info.get("shift_px", -1), 1))
        if roi is None or info["inliers"] < args.min_inliers or info.get("roi_kept", 0) < 0.95:
            row["status"] = "align_failed"
            rows.append(row)
            continue
        if os.path.exists(manual):
            mask = cv2.imread(manual, cv2.IMREAD_GRAYSCALE)
            row["label"] = "manual"
        else:
            r = seg(img, roi=roi)
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[roi > 0] = 1
            mask[r["foam"] > 0] = 2
            mask[r["fish"] > 0] = 3
            mask[(r["person"] > 0)] = 0
            row["label"] = "pseudo"
        water = np.isin(mask, (1, 2, 3)).sum()
        row.update(foam_ratio=round((mask == 2).sum() / max(water, 1), 4),
                   fish_ratio=round((mask == 3).sum() / max(water, 1), 4), status="ok")
        cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(mpath, mask)
        ov = img.copy()
        for c, col in COLORS.items():
            sel = mask == c
            ov[sel] = (0.55 * ov[sel] + 0.45 * np.array(col)).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, "overlays", name + ".jpg"), cv2.resize(ov, (640, 360)),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(row)
        if k % 100 == 0 or k == len(items):
            print(f"{k}/{len(items)}", flush=True)

    keys = ["name", "date", "time", "source", "split", "status", "label", "inliers", "shift_px",
            "foam_ratio", "fish_ratio"]
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print("status:", {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})})
    print("split:", {s: sum(r["split"] == s for r in ok) for s in ("train", "val", "test")},
          "with fish:", {s: sum(r["split"] == s and float(r["fish_ratio"]) > 0.02 for r in ok)
                         for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
