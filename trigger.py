"""
給餌開始トリガー

給餌システムは「待機 → 給餌中」の 2 状態で動き、給餌中だけ魚群の狙い点を出力する。
給餌中に入るきっかけ (トリガー) は差し替えられるようにしてある:

  RedCupTrigger  : 開発・評価用。赤いコップを持った人が水槽に近づいたら給餌開始 (今回のデータの給餌方法)
  SwitchTrigger  : 実運用用。ロボットアームの電源スイッチ ON で給餌開始。
                   アーム側の I/F が決まるまでは、信号ファイルの内容 (ON/OFF) で代用する
                   (例: echo ON > feeding_switch.txt)。GPIO・PLC・ROS などに置き換える場合は active() を実装し直す
  AlwaysTrigger  : 常に給餌中 (トリガーなしで狙い点を出し続ける。デバッグ用)

  .venv/bin/python stream_segmentation.py --mode fish --fish-model models/fish_segformer --trigger redcup
  .venv/bin/python stream_segmentation.py --mode fish --fish-model models/fish_segformer --trigger switch --switch-file feeding_switch.txt
"""
import os
import time

import cv2
import numpy as np

from find_red_cup import red_mask


class AlwaysTrigger:
    name = "always"

    def __call__(self, bgr):
        return True, {}


class SwitchTrigger:
    """ロボットアームの電源スイッチ ON = 給餌開始 (信号ファイルで代用)"""
    name = "switch"

    def __init__(self, path="feeding_switch.txt"):
        self.path = path

    def active(self):
        try:
            with open(self.path) as f:
                return f.read().strip().upper().startswith("ON")
        except OSError:
            return False

    def __call__(self, bgr):
        return self.active(), {}


class RedCupTrigger:
    """赤いコップを持った人が水槽に近づいた = 給餌開始

    1 フレームの判定: YOLO の人マスクが水槽の周り (水面 ROI を margin 広げた範囲) にかかり、
    かつ人マスクを少し広げた範囲に一定面積以上の赤い領域 (コップ) がある。
    背景の赤い物 (バルブ・ホースリール等) は、長時間赤いままの画素として学習して除外する
    (最初のフレームの赤い画素で初期化し、時定数 static_tau 秒の移動平均で更新)。
    on_frames 回続けて成立したら給餌開始、off_sec 秒成立しなければ給餌終了 (待機に戻る)。
    """
    name = "redcup"

    def __init__(self, roi_mask, weights="yolo11m-seg.pt", conf=0.3, min_area=200, near_margin=60,
                 person_margin=40, on_frames=2, off_sec=120.0, static_tau=600.0, clock=time.time):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.min_area = conf, min_area
        self.zone = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_margin * 2 + 1,) * 2))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (person_margin * 2 + 1,) * 2)
        self.on_frames, self.off_sec, self.clock = on_frames, off_sec, clock
        self.static, self.static_tau, self.last_t = None, static_tau, None  # 背景の赤い画素の出現率
        self.streak, self.last_hit, self.feeding = 0, -1e9, False

    def detect(self, bgr):
        """1 フレームで「赤いコップを持った人が水槽のそば」か。戻り値: (成立, 情報)"""
        from stream_segmentation import DEVICE
        h, w = bgr.shape[:2]
        scale = h * w / (1280 * 720)
        zone = self.zone if self.zone.shape == (h, w) else cv2.resize(self.zone, (w, h), interpolation=cv2.INTER_NEAREST)
        red = (red_mask(bgr) > 0).astype(np.float32)
        now = self.clock()
        if self.static is None or self.static.shape != red.shape:
            self.static = red.copy()
        else:
            a = 1.0 - np.exp(-max(now - self.last_t, 0.0) / self.static_tau)
            self.static = (1 - a) * self.static + a * red
        self.last_t = now
        red[cv2.dilate((self.static > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
        r = self.model.predict(bgr, conf=self.conf, classes=[0], device=DEVICE, verbose=False, retina_masks=True)[0]
        info = {"cup_area": 0, "cup_box": None, "person_near": False}
        if r.masks is None:
            return False, info
        for poly in r.masks.xy:
            if len(poly) < 3:
                continue
            person = np.zeros((h, w), np.uint8)
            cv2.fillPoly(person, [poly.astype(np.int32)], 255)
            area = max(int((person > 0).sum()), 1)
            if ((person > 0) & (zone > 0)).sum() / area < 0.05:  # 水槽のそばにいない人
                continue
            info["person_near"] = True
            cup = (red > 0) & (cv2.dilate(person, self.kernel) > 0)
            n, lab, st, _ = cv2.connectedComponentsWithStats(cup.astype(np.uint8))
            for k in range(1, n):
                if st[k, cv2.CC_STAT_AREA] > info["cup_area"]:
                    info["cup_area"] = int(st[k, cv2.CC_STAT_AREA])
                    info["cup_box"] = tuple(int(v) for v in st[k, :4])
        return info["cup_area"] >= self.min_area * scale, info

    def __call__(self, bgr):
        hit, info = self.detect(bgr)
        now = self.clock()
        self.streak = self.streak + 1 if hit else 0
        if hit:
            self.last_hit = now
        if not self.feeding and self.streak >= self.on_frames:
            self.feeding = True
        elif self.feeding and now - self.last_hit > self.off_sec:
            self.feeding = False
        info["hit"] = hit
        return self.feeding, info


def make_trigger(kind, roi_mask=None, switch_file="feeding_switch.txt", **kw):
    if kind in (None, "none", "always"):
        return AlwaysTrigger()
    if kind == "switch":
        return SwitchTrigger(switch_file)
    if kind == "redcup":
        if roi_mask is None:
            from foam import FoamSegmenter
            roi_mask = FoamSegmenter("foam_roi.json").roi_mask((720, 1280))
        return RedCupTrigger(roi_mask, **kw)
    raise ValueError(f"unknown trigger: {kind}")


def draw_state(out, feeding, trigger_name, info=None):
    """画面左上に 待機/給餌中 とトリガー種別を表示 (赤いコップの検出位置も枠で示す)"""
    if info and info.get("cup_box"):
        x, y, w, h = info["cup_box"]
        cv2.rectangle(out, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 255, 255), 2)
    from PIL import Image, ImageDraw, ImageFont
    from fish import LEGEND_FONT
    k = out.shape[1] / 1280
    label = {"redcup": "赤いコップ", "switch": "アーム電源", "always": "常時"}.get(trigger_name, trigger_name)
    text = f"{'給餌中' if feeding else '待機中'}  (トリガー: {label})"
    try:
        font = ImageFont.truetype(LEGEND_FONT, int(22 * k))
    except OSError:
        font, text = ImageFont.load_default(), ("FEEDING" if feeding else "STANDBY") + f" (trigger: {trigger_name})"
    x0, y0 = int(10 * k), int(44 * k)
    tw = int(font.getlength(text))
    box = out[y0:y0 + int(34 * k), x0:x0 + tw + int(20 * k)]
    color = np.array((0, 140, 0) if feeding else (90, 90, 90), np.uint8)
    box[:] = (0.25 * box + 0.75 * color).astype(np.uint8)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x0 + int(10 * k), y0 + int(17 * k)), text, font=font, fill=(255, 255, 255), anchor="lm")
    out[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out
