"""
「赤いコップを持った人」が写っている画像を抜き出してコピーする (給餌シーンの抽出)

判定: YOLO の person マスクを少し広げた範囲の中に、一定面積以上の赤い領域 (コップ) があること。
      背景の赤い物 (バルブ・ホースリール・クランプ) は人の近くにないので除外される。

  .venv/bin/python find_red_cup.py                                   # feeding/feeding_clear -> feeding/red_cup
  .venv/bin/python find_red_cup.py --src feeding/raw --dst feeding/red_cup_all
"""
import argparse
import csv
import glob
import os
import shutil

import cv2
import numpy as np

from stream_segmentation import DEVICE


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (((h <= 8) | (h >= 172)) & (s >= 120) & (v >= 70)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="feeding/feeding_clear")
    ap.add_argument("--dst", default="feeding/red_cup")
    ap.add_argument("--min-area", type=int, default=200, help="コップとみなす赤領域の最小面積 [px, 1280x720 換算]")
    ap.add_argument("--margin", type=int, default=40, help="人マスクを広げる幅 [px]")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO("yolo11m-seg.pt")
    os.makedirs(args.dst, exist_ok=True)
    os.makedirs(args.dst + "_debug", exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.margin * 2 + 1,) * 2)

    rows = []
    paths = sorted(glob.glob(os.path.join(args.src, "*.jpg")))
    # 背景の赤い物 (バルブ・ホースリールなど): 半数以上の画像で同じ位置が赤い画素を除外する
    acc = None
    for p in paths[:: max(1, len(paths) // 150)]:
        m = red_mask(cv2.imread(p)).astype(np.float32) / 255
        acc = m if acc is None else acc + m
    static = (acc / len(paths[:: max(1, len(paths) // 150)]) > 0.5).astype(np.uint8) * 255
    static = cv2.dilate(static, np.ones((9, 9), np.uint8))
    print(f"static red pixels: {int((static > 0).sum())}", flush=True)
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        scale = (img.shape[0] * img.shape[1]) / (1280 * 720)
        red = red_mask(img)
        if static.shape == red.shape:
            red[static > 0] = 0
        r = model.predict(img, conf=args.conf, classes=[0], device=DEVICE, verbose=False, retina_masks=True)[0]
        best = {"area": 0, "box": None}
        if r.masks is not None:
            person = np.zeros(img.shape[:2], np.uint8)
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(person, [poly.astype(np.int32)], 255)
            zone = cv2.dilate(person, kernel)
            n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_and(red, zone))
            for k in range(1, n):
                if st[k, cv2.CC_STAT_AREA] > best["area"]:
                    x, y, w, h = st[k, :4]
                    best = {"area": int(st[k, cv2.CC_STAT_AREA]), "box": (int(x), int(y), int(w), int(h))}
        hit = best["area"] >= args.min_area * scale
        name = os.path.basename(p)
        rows.append({"file": name, "red_cup": int(hit), "red_area_px": best["area"]})
        if hit:
            shutil.copy(p, os.path.join(args.dst, name))
        if best["box"]:
            dbg = img.copy()
            x, y, w, h = best["box"]
            cv2.rectangle(dbg, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 0) if hit else (0, 0, 255), 3)
            cv2.putText(dbg, f"red {best['area']}px {'CUP' if hit else 'no'}", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hit else (0, 0, 255), 2)
            cv2.imwrite(os.path.join(args.dst + "_debug", name), cv2.resize(dbg, (640, 360)))
        if i % 50 == 0 or i == len(paths):
            print(f"{i}/{len(paths)} hits={sum(r['red_cup'] for r in rows)}", flush=True)

    with open(os.path.join(args.dst, "red_cup.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"done: {sum(r['red_cup'] for r in rows)} / {len(rows)} images copied -> {args.dst}/")


if __name__ == "__main__":
    main()
