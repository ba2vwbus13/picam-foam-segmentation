"""
水槽水面の白い泡のセグメンテーション

1. 水面 ROI (楕円) を推定
   SegFormer の tank 領域内にある暗いブロブ (沈殿物) の凸包に楕円をフィット。
   カメラ固定前提なので一度求めたら JSON に保存して再利用する。
   自動推定がずれる場合は --calibrate で内側の縁を5点以上クリックして手動設定。
2. ROI 内で「明るい (Lab の L が Otsu 閾値以上) かつ 低彩度」を泡とみなす。
"""
import json
import os
from collections import deque

import cv2
import numpy as np


class FoamSegmenter:
    def __init__(self, roi_path="foam_roi.json", shrink=0.97, sat_max=60, sem=None, trail=60):
        self.roi_path = roi_path
        self.shrink = shrink
        self.sat_max = sat_max
        self.sem = sem  # 自動 ROI 推定に使う SemanticSegmenter (遅延生成)
        self.ellipse = None
        self.size = None
        self.mask = None
        self.trail = deque(maxlen=trail)  # 重心の軌跡
        if os.path.exists(roi_path):
            d = json.load(open(roi_path))
            self.ellipse = (tuple(d["center"]), tuple(d["axes"]), d["angle"])
            self.size = tuple(d["size"])

    # ---------------------------------------------------------- ROI
    def save_roi(self, shape):
        (cx, cy), (a, b), ang = self.ellipse
        self.size = (shape[1], shape[0])
        json.dump({"center": [cx, cy], "axes": [a, b], "angle": ang, "size": list(self.size)},
                  open(self.roi_path, "w"), indent=2)
        print(f"[foam] ROI saved -> {self.roi_path}")

    def estimate_roi(self, bgr):
        if self.sem is None:
            from stream_segmentation import SemanticSegmenter
            self.sem = SemanticSegmenter("nvidia/segformer-b2-finetuned-ade-512-512")
        h, w = bgr.shape[:2]
        seg = self.sem(bgr)
        ids = {v: k for k, v in self.sem.labels.items()}
        tank = np.isin(seg, [ids[n] for n in ("tank", "water", "swimming pool") if n in ids]).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(tank)
        if n <= 1:
            raise RuntimeError("tank 領域が見つかりません。--calibrate で手動設定してください")
        tank = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        inner = cv2.erode(tank, np.ones((15, 15), np.uint8))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = ((gray > 30) & (gray < 135) & (inner > 0)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
        # 画像端に接するもの (水槽外壁の影など) を除外
        good = [i for i in range(1, n)
                if st[i, 4] > 150 and st[i, 0] > 2 and st[i, 1] > 2
                and st[i, 0] + st[i, 2] < w - 2 and st[i, 1] + st[i, 3] < h - 2]
        pts = np.column_stack(np.where(np.isin(lab, good)))[:, ::-1].astype(np.int32)
        if len(pts) < 5:
            raise RuntimeError("沈殿物が少なく ROI を推定できません。--calibrate で手動設定してください")
        self.ellipse = cv2.fitEllipse(cv2.convexHull(pts))
        self.save_roi(bgr.shape)

    def calibrate(self, bgr):
        """水面の内側の縁をクリック (5点以上)。Enter で確定、r でやり直し、Esc で中止"""
        pts, win = [], "calibrate: click inner rim (>=5 pts), Enter=OK r=reset Esc=cancel"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, lambda e, x, y, *_: pts.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
        while True:
            v = bgr.copy()
            for p in pts:
                cv2.circle(v, p, 4, (0, 0, 255), -1)
            if len(pts) >= 5:
                cv2.ellipse(v, cv2.fitEllipse(np.array(pts, np.int32)), (0, 255, 255), 2)
            cv2.imshow(win, v)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10) and len(pts) >= 5:
                break
            if k == ord("r"):
                pts.clear()
            if k == 27:
                cv2.destroyWindow(win)
                return False
        cv2.destroyWindow(win)
        self.ellipse = cv2.fitEllipse(np.array(pts, np.int32))
        self.shrink = 1.0
        self.save_roi(bgr.shape)
        return True

    def roi_mask(self, shape):
        h, w = shape[:2]
        if self.mask is not None and self.mask.shape == (h, w):
            return self.mask
        (cx, cy), (a, b), ang = self.ellipse
        sx, sy = w / self.size[0], h / self.size[1]  # 解像度が変わった場合に追従
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, ((cx * sx, cy * sy), (a * sx * self.shrink, b * sy * self.shrink), ang), 255, -1)
        self.mask = m
        return m

    # ---------------------------------------------------------- inference
    def __call__(self, bgr):
        if self.ellipse is None:
            self.estimate_roi(bgr)
        roi = self.roi_mask(bgr.shape)
        L = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0], (5, 5), 0)
        S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
        t, _ = cv2.threshold(L[roi > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foam = ((L > t) & (S < self.sat_max) & (roi > 0)).astype(np.uint8) * 255
        foam = cv2.morphologyEx(foam, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        foam = cv2.morphologyEx(foam, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        ratio = float((foam > 0).sum()) / max(int((roi > 0).sum()), 1)
        # 泡の重心 (泡マスク全体の面積重心)
        m = cv2.moments(foam, binaryImage=True)
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else None
        if centroid:
            self.trail.append(centroid)
        return foam, roi, ratio, float(t), centroid

    def draw(self, bgr, result, color=(255, 0, 255)):
        foam, roi, ratio, t, centroid = result
        out = bgr.copy()
        sel = foam > 0
        out[sel] = (0.4 * out[sel] + 0.6 * np.array(color)).astype(np.uint8)
        cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 255, 255), 2)
        info = {"foam_ratio": round(ratio, 4), "L_threshold": t, "cx": None, "cy": None, "dx": None, "dy": None}
        if centroid:
            cx, cy = centroid
            # 水面 (ROI 楕円) 中心からのずれ: 画像の横/縦方向それぞれ、楕円の半幅/半高で正規化 (-1〜1, 右・下が正)
            (ex, ey), (a, b), ang = self.ellipse_px(bgr.shape)
            th = np.deg2rad(ang)
            rx = np.hypot(a / 2 * np.cos(th), b / 2 * np.sin(th))
            ry = np.hypot(a / 2 * np.sin(th), b / 2 * np.cos(th))
            u, v = (cx - ex) / rx, (cy - ey) / ry
            info.update(cx=round(cx, 1), cy=round(cy, 1), dx=round(float(u), 3), dy=round(float(v), 3))
            pts = np.array(self.trail, np.int32)
            if len(pts) > 1:
                cv2.polylines(out, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(out, (int(ex), int(ey)), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.line(out, (int(ex), int(ey)), (int(cx), int(cy)), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(out, (int(cx), int(cy)), 9, (0, 0, 0), -1)
            cv2.circle(out, (int(cx), int(cy)), 6, (0, 255, 0), -1)
            label = f"centroid ({cx:.0f},{cy:.0f})"
            cv2.putText(out, label, (int(cx) + 12, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, label, (int(cx) + 12, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        txt = f"foam {ratio*100:.1f}% of surface"
        cv2.putText(out, txt, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out, info

    def ellipse_px(self, shape):
        (cx, cy), (a, b), ang = self.ellipse
        sx, sy = shape[1] / self.size[0], shape[0] / self.size[1]
        return (cx * sx, cy * sy), (a * sx, b * sy), ang
