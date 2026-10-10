"""
給餌時に水面へ集まる魚群 (黒く写る領域) のセグメンテーション

手順 (泡と同じく画像処理のみ):
1. 水面 ROI (foam_roi.json の楕円を少し内側に縮めたもの)
2. 泡 (明るい画素) と人 (YOLO person マスク) を除外
3. 「暗い (Lab の L < l_max) かつ 黄色みが少ない (b-128 < b_max)」画素 = 魚の背中
   茶色い水は明るく黄色みが強いので除外される
4. 魚画素の密度が高い領域 (density > dens_min) を魚群とし、小領域を除去

  .venv/bin/python fish.py --src feeding/red_cup --dst feeding/fish                    # 閾値処理
  .venv/bin/python fish.py --model models/fish_segformer --src 画像またはフォルダ --dst 出力先  # 学習モデル
  .venv/bin/python fish.py --model models/fish_segformer --src archive/2026-10-09 --start 15-45 --end 16-30 \
      --video fish_1009.mp4 --dst out_1009                                              # 連続画像 -> 動画
出力: <dst>/overlays/ (魚群・最大の群れ・狙い点を描いた画像), <dst>/masks/ (魚群マスク), <dst>/fish.csv
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from foam import FoamSegmenter


def aim_point(fish, dark, win=121):
    """給餌の狙い点 = 最大の群れの中で魚画素が最も密な点

    全体の重心は群れが分かれている/曲がっていると魚のいない所に落ちるため使わない。
    密度は大きめの窓 (餌が広がる範囲の目安) で平均し、群れの縁に寄りすぎないよう
    群れの内側ほど高くなる距離変換を掛ける。
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(fish)
    if n <= 1:
        return None, None
    largest = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    inside = largest > 0
    dens = cv2.blur(dark * inside, (win, win))
    dist = cv2.distanceTransform(largest, cv2.DIST_L2, 5)
    score = dens * np.minimum(dist / (win / 2), 1.0)
    y, x = np.unravel_index(np.argmax(np.where(inside, score, -1)), score.shape)
    return largest, (float(x), float(y))


LEGEND_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
LEGEND_ITEMS = [("aim", "狙い点 (最大の群れで最も密な点)"), ("centroid", "重心 (魚群全体)"),
                ("fish", "魚群"), ("largest", "最大の群れ"), ("foam", "泡"), ("roi", "水面")]
_legend_cache = {}


def legend_patch(width):
    """凡例 (BGR 画像と不透明度マスク)。画像の横幅に合わせた大きさで作り、使い回す"""
    if width in _legend_cache:
        return _legend_cache[width]
    from PIL import Image, ImageDraw, ImageFont
    k = width / 1280
    fs, row, icon, pad = int(17 * k), int(28 * k), int(36 * k), int(10 * k)
    try:
        font = ImageFont.truetype(LEGEND_FONT, fs)
    except OSError:  # macOS 以外: 日本語フォントが無ければ英語の凡例
        font = None
    labels = [t for _, t in LEGEND_ITEMS] if font else ["aim point", "centroid", "fish school", "largest school", "foam", "water"]
    font = font or ImageFont.load_default()
    tw = max(int(font.getlength(t)) for t in labels)
    w, h = pad * 3 + icon + tw, pad * 2 + row * len(LEGEND_ITEMS)
    img = np.full((h, w, 3), 40, np.uint8)
    for i, (kind, _) in enumerate(LEGEND_ITEMS):
        cx, cy = pad + icon // 2, pad + row * i + row // 2
        s = max(1, int(k + 0.5))
        if kind == "aim":
            r = int(9 * k)
            for c, t in (((0, 0, 0), 4 * s), ((0, 0, 255), 2 * s)):
                cv2.circle(img, (cx, cy), r, c, t)
                cv2.line(img, (cx - int(14 * k), cy), (cx + int(14 * k), cy), c, t)
                cv2.line(img, (cx, cy - int(12 * k)), (cx, cy + int(12 * k)), c, t)
        elif kind == "centroid":
            cv2.circle(img, (cx, cy), int(7 * k), (0, 0, 0), -1)
            cv2.circle(img, (cx, cy), int(5 * k), (160, 160, 160), -1)
        else:
            x0, x1, y0, y1 = cx - icon // 2, cx + icon // 2, cy - int(8 * k), cy + int(8 * k)
            if kind == "fish":
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), -1)
            elif kind == "foam":
                cv2.rectangle(img, (x0, y0), (x1, y1), (220, 80, 220), -1)
            elif kind == "largest":
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2 * s)
            elif kind == "roi":
                cv2.ellipse(img, (cx, cy), (icon // 2, int(8 * k)), 0, 0, 360, (0, 255, 255), s)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for i, t in enumerate(labels):
        d.text((pad * 2 + icon, pad + row * i + row // 2), t, font=font, fill=(255, 255, 255), anchor="lm")
    _legend_cache[width] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return _legend_cache[width]


def draw_legend(out, top=None):
    """右上に半透明の凡例を描く (top: 上端の y。既定は画像高さの 6%)"""
    p = legend_patch(out.shape[1])
    h, w = p.shape[:2]
    m = max(6, out.shape[1] // 128)
    y0 = int(out.shape[0] * 0.06) if top is None else top
    x0 = out.shape[1] - w - m
    if x0 < 0 or y0 + h > out.shape[0]:
        return out
    roi = out[y0:y0 + h, x0:x0 + w]
    out[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(p, 0.8, roi, 0.2, 0)
    return out


class FishSchoolSegmenter:
    def __init__(self, roi_path="foam_roi.json", l_max=68, b_max=9, dens_min=0.5, win=41, min_area=2000,
                 exclude_person=True, foam_l_min=0):
        self.foam = FoamSegmenter(roi_path, l_min=foam_l_min)
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

    def __call__(self, bgr, roi=None, person=None):
        """roi: 水面マスク (省略時は foam_roi.json の楕円)。日ごとの画角ずれは ViewAligner で求めた ROI を渡す"""
        if roi is None:
            roi = self.foam.roi_mask(bgr.shape)
        roi = cv2.erode(roi, np.ones((15, 15), np.uint8))
        foam, _, foam_ratio, _, _ = self.foam.segment(bgr, roi)
        if person is None:
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
        largest, aim = aim_point(fish, dark)
        return {"fish": fish, "foam": foam, "roi": roi, "person": person, "fish_ratio": ratio,
                "foam_ratio": foam_ratio, "centroid": centroid, "largest": largest, "aim": aim}

    @staticmethod
    def draw(bgr, r, legend=True):
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
        if r.get("largest") is not None:
            lc, _ = cv2.findContours(r["largest"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, lc, -1, (255, 255, 255), 2)
        if r["centroid"]:  # 全体の重心 (参考、灰色)
            cx, cy = map(int, r["centroid"])
            cv2.circle(out, (cx, cy), 7, (0, 0, 0), -1)
            cv2.circle(out, (cx, cy), 5, (160, 160, 160), -1)
        if r.get("aim"):  # 狙い点 (赤い照準)
            ax, ay = map(int, r["aim"])
            for c, t in (((0, 0, 0), 5), ((0, 0, 255), 2)):
                cv2.circle(out, (ax, ay), 14, c, t)
                cv2.line(out, (ax - 22, ay), (ax + 22, ay), c, t)
                cv2.line(out, (ax, ay - 22), (ax, ay + 22), c, t)
        txt = f"fish {r['fish_ratio']*100:.1f}%  foam {r['foam_ratio']*100:.1f}%"
        cv2.putText(out, txt, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return draw_legend(out) if legend else out


class FishModelSegmenter:
    """学習済み 4 クラス SegFormer (水面外/水/泡/魚群, train_foam.py --classes ...) による版。ROI・閾値の設定は不要"""

    def __init__(self, model_dir="models/fish_segformer", min_area=2000):
        import torch
        from train_foam import DEVICE, INFER_SIZE, load_model, to_tensor
        self.torch, self.device, self.size_in, self.to_tensor = torch, DEVICE, INFER_SIZE, to_tensor
        self.model = load_model(model_dir).eval()
        self.min_area = min_area

    def __call__(self, bgr):
        torch = self.torch
        h, w = bgr.shape[:2]
        x = self.to_tensor(cv2.resize(bgr, self.size_in, interpolation=cv2.INTER_AREA))[None].to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values=x).logits
            pred = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                                   align_corners=False).argmax(1)[0].cpu().numpy()
        roi = (pred >= 1).astype(np.uint8) * 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(roi)  # 水面は最大の連結成分だけ
        if n > 1:
            roi = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
        foam = ((pred == 2) & (roi > 0)).astype(np.uint8) * 255
        fish = ((pred == 3) & (roi > 0)).astype(np.uint8) * 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(fish)
        fish = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= self.min_area]).astype(np.uint8) * 255
        water = max(int((roi > 0).sum()), 1)
        m = cv2.moments(fish, binaryImage=True)
        largest, aim = aim_point(fish, (fish > 0).astype(np.float32), win=max(15, w * 121 // 1280) | 1)
        return {"fish": fish, "foam": foam, "roi": roi, "fish_ratio": float((fish > 0).sum()) / water,
                "foam_ratio": float((foam > 0).sum()) / water,
                "centroid": (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else None,
                "largest": largest, "aim": aim}


class LiveFish:
    """ライブ用: 魚群の推定 + 狙い点の安定化 (直近フレームの中央値。見失っても hold 秒は保持)

    info の aim_dx, aim_dy は水面の楕円中心からのずれを半径で正規化した値 (-1〜1、右・下が正、画像座標)
    """

    def __init__(self, model_dir=None, n=5, hold=1.0):
        from collections import deque
        self.seg = FishModelSegmenter(model_dir) if model_dir else FishSchoolSegmenter()
        self.hist = deque(maxlen=n)
        self.hold, self.last_t, self.aim = hold, 0.0, None

    def __call__(self, bgr):
        import time
        r = self.seg(bgr)
        now = time.time()
        if r["aim"]:
            self.hist.append(r["aim"])
            self.last_t = now
        elif now - self.last_t > self.hold:
            self.hist.clear()
        self.aim = tuple(np.median(np.array(self.hist), axis=0)) if self.hist else None
        r["aim_raw"], r["aim"] = r["aim"], self.aim
        return r

    def draw(self, bgr, r):
        out = FishSchoolSegmenter.draw(bgr, r)
        info = {"fish_ratio": round(r["fish_ratio"], 4), "aim_x": None, "aim_y": None, "aim_dx": None, "aim_dy": None}
        cs, _ = cv2.findContours(r["roi"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if r["aim"] and cs and len(max(cs, key=cv2.contourArea)) >= 5:
            (ex, ey), (a, b), ang = cv2.fitEllipse(max(cs, key=cv2.contourArea))
            th = np.deg2rad(ang)
            rx, ry = np.hypot(a / 2 * np.cos(th), b / 2 * np.sin(th)), np.hypot(a / 2 * np.sin(th), b / 2 * np.cos(th))
            ax, ay = map(float, r["aim"])
            info.update(aim_x=round(ax, 1), aim_y=round(ay, 1),
                        aim_dx=round(float((ax - ex) / rx), 3), aim_dy=round(float((ay - ey) / ry), 3))
        return out, info


VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")


def frames(src, start=None, end=None, step=1):
    """(名前, 画像) を順に返す。src = 画像 1 枚 / 画像フォルダ (ファイル名順) / 動画ファイル
    start, end: ファイル名の先頭と比較する範囲 (例 "15-45", "16-30")。動画では秒数"""
    if os.path.isfile(src) and src.lower().endswith(VIDEO_EXT):
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        k = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = k / fps
            if (start is None or t >= float(start)) and (end is None or t <= float(end)) and k % step == 0:
                yield f"{os.path.splitext(os.path.basename(src))[0]}_{k:06d}", img
            k += 1
        return
    if os.path.isfile(src):
        paths = [src]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(src, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
        names = [os.path.basename(p) for p in paths]
        paths = [p for p, n in zip(paths, names) if (start is None or n >= start) and (end is None or n[:len(end)] <= end)]
    for p in paths[::step]:
        yield os.path.splitext(os.path.basename(p))[0], cv2.imread(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="feeding/red_cup", help="画像 1 枚 / 画像フォルダ / 動画ファイル")
    ap.add_argument("--dst", help="出力先 (既定: fish_out/<入力のフォルダ名やファイル名>)")
    ap.add_argument("--model", help="学習済み 4 クラスモデル (例: models/fish_segformer)。未指定時は閾値処理")
    ap.add_argument("--start", help="処理範囲の開始 (画像: ファイル名の先頭と比較 例 15-45 / 動画: 秒)")
    ap.add_argument("--end", help="処理範囲の終了 (同上)")
    ap.add_argument("--step", type=int, default=1, help="N 枚に 1 枚だけ処理する")
    ap.add_argument("--video", help="結果を連続再生できる mp4 に書き出す (例: out.mp4)。指定時は overlays/ を書かない")
    ap.add_argument("--fps", type=float, default=4.0, help="--video のフレームレート (アーカイブ 5 秒間隔なら 4 で 20 倍速)")
    ap.add_argument("--width", type=int, default=1280, help="--video の横幅")
    ap.add_argument("--smooth", type=int, default=1, help="狙い点を直近 N フレームの中央値にする (連続画像向け)")
    args = ap.parse_args()
    if not args.dst:  # 既存の結果 (feeding/fish など) を上書きしないよう、入力ごとに別フォルダ
        args.dst = os.path.join("fish_out", os.path.splitext(os.path.basename(os.path.normpath(args.src)))[0])
    os.makedirs(os.path.join(args.dst, "masks"), exist_ok=True)
    if not args.video:
        os.makedirs(os.path.join(args.dst, "overlays"), exist_ok=True)
    seg = FishModelSegmenter(args.model) if args.model else FishSchoolSegmenter()
    from collections import deque
    hist = deque(maxlen=max(1, args.smooth))
    writer, rows = None, []
    for i, (name, img) in enumerate(frames(args.src, args.start, args.end, args.step), 1):
        r = seg(img)
        if r["aim"]:
            hist.append(r["aim"])
        else:
            hist.clear()
        if args.smooth > 1 and hist:
            r["aim"] = tuple(float(v) for v in np.median(np.array(hist), axis=0))
        out = FishSchoolSegmenter.draw(img, r)
        cv2.imwrite(os.path.join(args.dst, "masks", name + ".png"), r["fish"])
        if args.video:
            h = args.width * out.shape[0] // out.shape[1]
            frame = cv2.resize(out, (args.width, h), interpolation=cv2.INTER_AREA)
            for c, t in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                cv2.putText(frame, name, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, t, cv2.LINE_AA)
            if writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
                from stream_segmentation import open_video_writer
                writer = open_video_writer(args.video, args.fps, (args.width, h))
            writer.write(frame)
        else:
            cv2.imwrite(os.path.join(args.dst, "overlays", name + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cx, cy = r["centroid"] or (None, None)
        ax, ay = r["aim"] or (None, None)
        rows.append({"name": name, "fish_ratio": round(r["fish_ratio"], 4), "foam_ratio": round(r["foam_ratio"], 4),
                     "cx": None if cx is None else round(cx, 1), "cy": None if cy is None else round(cy, 1),
                     "aim_x": None if ax is None else round(ax, 1), "aim_y": None if ay is None else round(ay, 1)})
        if i % 100 == 0:
            print(f"{i} frames", flush=True)
    if writer:
        writer.release()
    if not rows:
        raise SystemExit("対象の画像がありません (--src / --start / --end を確認)")
    with open(os.path.join(args.dst, "fish.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} frames -> {args.dst}/" + (f", {args.video}" if args.video else ""))


if __name__ == "__main__":
    main()
