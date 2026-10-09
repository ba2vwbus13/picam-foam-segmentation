"""
給餌時に水面へ集まる魚群 (黒く写る領域) のセグメンテーション

手順 (泡と同じく画像処理のみ):
1. 水面 ROI (foam_roi.json の楕円を少し内側に縮めたもの)
2. 泡 (明るい画素) と人 (YOLO person マスク) を除外
3. 「暗い (Lab の L < l_max) かつ 黄色みが少ない (b-128 < b_max)」画素 = 魚の背中
   茶色い水は明るく黄色みが強いので除外される
4. 魚画素の密度が高い領域 (density > dens_min) を魚群とし、小領域を除去

  .venv/bin/python fish.py --src feeding/red_cup --dst feeding/fish
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from foam import FoamSegmenter


class FishSchoolSegmenter:
    def __init__(self, roi_path="foam_roi.json", l_max=68, b_max=9, dens_min=0.5, win=41, min_area=2000,
                 exclude_person=True):
        self.foam = FoamSegmenter(roi_path)
        self.l_max, self.b_max, self.dens_min, self.win, self.min_area = l_max, b_max, dens_min, win, min_area
        self.person_model = None
        if exclude_person:
            from ultralytics import YOLO
            self.person_model = YOLO("yolo11m-seg.pt")

    def person_mask(self, bgr):
        m = np.zeros(bgr.shape[:2], np.uint8)
        if self.person_model is None:
            return m
        from stream_segmentation import DEVICE
        r = self.person_model.predict(bgr, conf=0.3, classes=[0], device=DEVICE, verbose=False, retina_masks=True)[0]
        if r.masks is not None:
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        return cv2.dilate(m, np.ones((21, 21), np.uint8))

    def __call__(self, bgr):
        roi = cv2.erode(self.foam.roi_mask(bgr.shape), np.ones((15, 15), np.uint8))
        foam, _, foam_ratio, _, _ = self.foam.segment(bgr, roi)
        person = self.person_mask(bgr)
        lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (9, 9), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
        L, b = lab[..., 0], lab[..., 2] - 128
        valid = (roi > 0) & (person == 0)
        dark = ((L < self.l_max) & (b < self.b_max) & valid & (foam == 0)).astype(np.float32)
        w = (self.win, self.win)
        dens = cv2.blur(dark, w) / np.maximum(cv2.blur(valid.astype(np.float32), w), 1e-3)
        fish = ((dens > self.dens_min) & valid).astype(np.uint8) * 255
        fish = cv2.morphologyEx(fish, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        n, lab_, st, _ = cv2.connectedComponentsWithStats(fish)
        fish = np.isin(lab_, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= self.min_area]).astype(np.uint8) * 255
        ratio = float((fish > 0).sum()) / max(int((roi > 0).sum()), 1)
        m = cv2.moments(fish, binaryImage=True)
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else None
        return {"fish": fish, "foam": foam, "roi": roi, "person": person, "fish_ratio": ratio,
                "foam_ratio": foam_ratio, "centroid": centroid}

    @staticmethod
    def draw(bgr, r):
        out = bgr.copy()
        sel = r["foam"] > 0
        out[sel] = (0.5 * out[sel] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        cs, _ = cv2.findContours(r["fish"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        g = out.copy()
        cv2.drawContours(g, cs, -1, (0, 255, 0), -1)
        out = cv2.addWeighted(g, 0.45, out, 0.55, 0)
        cv2.drawContours(out, cs, -1, (0, 255, 0), 2)
        rc, _ = cv2.findContours(r["roi"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, rc, -1, (0, 255, 255), 1)
        if r["centroid"]:
            cx, cy = map(int, r["centroid"])
            cv2.circle(out, (cx, cy), 9, (0, 0, 0), -1)
            cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)
        txt = f"fish {r['fish_ratio']*100:.1f}%  foam {r['foam_ratio']*100:.1f}%"
        cv2.putText(out, txt, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="feeding/red_cup")
    ap.add_argument("--dst", default="feeding/fish")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.dst, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "masks"), exist_ok=True)
    seg = FishSchoolSegmenter()
    rows = []
    paths = sorted(glob.glob(os.path.join(args.src, "*.jpg")))
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        r = seg(img)
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(args.dst, "overlays", name + ".jpg"), seg.draw(img, r), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(os.path.join(args.dst, "masks", name + ".png"), r["fish"])
        cx, cy = r["centroid"] or (None, None)
        rows.append({"name": name, "fish_ratio": round(r["fish_ratio"], 4), "foam_ratio": round(r["foam_ratio"], 4),
                     "cx": None if cx is None else round(cx, 1), "cy": None if cy is None else round(cy, 1)})
        if i % 40 == 0 or i == len(paths):
            print(f"{i}/{len(paths)}", flush=True)
    with open(os.path.join(args.dst, "fish.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"done -> {args.dst}/")


if __name__ == "__main__":
    main()
