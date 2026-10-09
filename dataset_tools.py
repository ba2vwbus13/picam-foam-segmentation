"""
泡セグメンテーション学習用データセット作成ツール

- PiCam アーカイブから画像を取得 (.env の接続情報を使用)
- 手前水槽の水面を YOLO11-seg の tank(sink) マスクから楕円推定 (日ごとに画角が変わるため画像ごとに推定)
- 既存の画像処理 (foam.py) で泡マスクを自動生成 = 疑似ラベル
"""
import os
import sys

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_segmentation import DEVICE, ENV_FILE, load_env_file  # noqa: E402


class Archive:
    def __init__(self, env_file=ENV_FILE):
        load_env_file(env_file)
        self.base = os.environ["PICAM_URL"].rsplit("/stream", 1)[0]
        self.auth = (os.environ["PICAM_USER"], os.environ["PICAM_PASS"])

    def get(self, path, **params):
        r = requests.get(self.base + path, params=params, auth=self.auth, timeout=30)
        r.raise_for_status()
        return r

    def days(self):
        return sorted(self.get("/api/calendar/days").json()["days"])

    def images(self, day, hour):
        return self.get("/api/calendar/images", date=day, hour=hour, interval=0).json()["images"]

    def image(self, url):
        return cv2.imdecode(np.frombuffer(self.get(url).content, np.uint8), cv2.IMREAD_COLOR)


class TankFinder:
    """手前 (画像下寄りで最大) の水槽上面を YOLO-seg で検出し、内側水面の楕円を返す"""

    TANK_CLASSES = ("sink", "toilet", "bowl", "tub")

    def __init__(self, weights="yolo11m-seg.pt"):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def __call__(self, bgr, shrink=0.88):
        h, w = bgr.shape[:2]
        r = self.model.predict(bgr, conf=0.1, device=DEVICE, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            return None
        best, score = None, -1
        for i, c in enumerate(r.boxes.cls.tolist()):
            if r.names[int(c)] not in self.TANK_CLASSES:
                continue
            poly = r.masks.xy[i]
            if len(poly) < 5:
                continue
            area = cv2.contourArea(poly.astype(np.float32))
            cy = poly[:, 1].mean() / h
            s = area * (0.5 + cy)  # 大きく、画面下寄り (= 手前) を優先
            if s > score:
                best, score = poly, s
        if best is None:
            return None
        (cx, cy), (a, b), ang = cv2.fitEllipse(best.astype(np.float32))
        return (cx, cy), (a * shrink, b * shrink), ang


class ViewAligner:
    """参照フレーム (ROI を定義した画角) に対する射影変換を ORB 特徴点で推定し、ROI マスクを対象画像へ写す"""

    def __init__(self, ref_bgr, ref_roi_mask, size=(1280, 720)):
        self.size = size
        self.orb = cv2.ORB_create(5000)
        self.ref_gray = cv2.cvtColor(cv2.resize(ref_bgr, size), cv2.COLOR_BGR2GRAY)
        # 水槽の水面は毎回変わるので特徴点を取らない (静止した構造物だけで合わせる)
        feat_mask = cv2.bitwise_not(cv2.dilate(ref_roi_mask, np.ones((61, 61), np.uint8)))
        self.kp, self.des = self.orb.detectAndCompute(self.ref_gray, feat_mask)
        self.ref_roi = ref_roi_mask
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, bgr):
        """戻り値: (1280x720 画像, ROI マスク, 情報 dict) / 失敗時は (画像, None, 情報)"""
        img = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        info = {"inliers": 0}
        if des is None or len(kp) < 50:
            return img, None, info
        pairs = self.bf.knnMatch(self.des, des, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            return img, None, info
        src = np.float32([self.kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            return img, None, info
        info["inliers"] = int(inl.sum())
        w, h = self.size
        roi = cv2.warpPerspective(self.ref_roi, H, (w, h), flags=cv2.INTER_NEAREST)
        info["roi_kept"] = float((roi > 0).sum()) / max(int((self.ref_roi > 0).sum()), 1)
        corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H)
        info["shift_px"] = float(np.abs(corners.reshape(-1, 2) - [[0, 0], [w, 0], [w, h], [0, h]]).max())
        return img, roi, info
