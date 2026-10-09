"""
PiCam MJPEG ストリームに対するリアルタイム・セグメンテーション

  semantic : SegFormer (ADE20K 150クラス) によるセマンティックセグメンテーション [既定]
  instance : YOLO11-seg (COCO 80クラス) によるインスタンスセグメンテーション
  both     : 両方を重ねて表示
  foam     : 水槽水面の白い泡を抽出し、水面に占める泡の割合を算出 (foam.py)

認証情報は環境変数で渡す (HTTP Basic):
  export PICAM_URL=https://<camera-host>/stream PICAM_USER=xxx PICAM_PASS=yyy
  .venv/bin/python stream_segmentation.py
  .venv/bin/python stream_segmentation.py --mode both --save out.mp4
  .venv/bin/python stream_segmentation.py --mode foam --csv foam_log.csv

ローカル画像で試す場合:
  .venv/bin/python stream_segmentation.py --image sample.jpg
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np
import requests
import torch

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env_file(path):
    """KEY=VALUE 形式のファイルを環境変数に読み込む (既に設定済みの環境変数が優先)"""
    if not os.path.exists(path):
        return False
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        os.environ.setdefault(k, v)
    return True
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- stream
class MJPEGReader(threading.Thread):
    """multipart/x-mixed-replace の JPEG を読み続け、最新フレームだけ保持する"""

    def __init__(self, url, auth):
        super().__init__(daemon=True)
        self.url, self.auth = url, auth
        self.frame = None
        self.seq = 0  # 受信フレーム番号 (重複処理の判定用)
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            try:
                with requests.get(self.url, auth=self.auth, stream=True, timeout=10) as r:
                    r.raise_for_status()
                    buf = b""
                    for chunk in r.iter_content(chunk_size=16384):
                        if not self.running:
                            return
                        buf += chunk
                        while True:
                            s = buf.find(b"\xff\xd8")
                            e = buf.find(b"\xff\xd9", s + 2)
                            if s < 0 or e < 0:
                                break
                            jpg, buf = buf[s:e + 2], buf[e + 2:]
                            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                with self.lock:
                                    self.frame = img
                                    self.seq += 1
                        if len(buf) > 5_000_000:
                            buf = b""
            except requests.HTTPError as ex:
                print(f"[stream] HTTP error: {ex}")
                if ex.response is not None and ex.response.status_code == 401:
                    print("[stream] 認証失敗: PICAM_USER / PICAM_PASS を確認してください")
                    self.running = False
                    return
                time.sleep(3)
            except Exception as ex:
                print(f"[stream] reconnecting: {ex}")
                time.sleep(3)

    def latest(self):
        with self.lock:
            return self.seq, (None if self.frame is None else self.frame.copy())


# ---------------------------------------------------------------- models
class SemanticSegmenter:
    def __init__(self, model_name):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.proc = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(DEVICE).eval()
        self.labels = self.model.config.id2label
        rng = np.random.default_rng(0)
        self.palette = rng.integers(40, 255, size=(len(self.labels), 3), dtype=np.uint8)

    @torch.no_grad()
    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.proc(images=rgb, return_tensors="pt").to(DEVICE)
        logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=bgr.shape[:2], mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].cpu().numpy().astype(np.int32)

    def draw(self, bgr, seg, alpha=0.5, min_ratio=0.005):
        color = self.palette[seg][..., ::-1]
        out = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)
        ids, counts = np.unique(seg, return_counts=True)
        total = seg.size
        stats = sorted(((c / total, i) for i, c in zip(ids, counts) if c / total >= min_ratio), reverse=True)
        # 各クラス領域の重心付近にラベル
        for ratio, cid in stats:
            mask = (seg == cid).astype(np.uint8)
            n, _, st, cen = cv2.connectedComponentsWithStats(mask)
            if n <= 1:
                continue
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            cx, cy = map(int, cen[k])
            put_label(out, self.labels[int(cid)], (cx, cy), tuple(int(v) for v in self.palette[cid][::-1]))
        # 凡例 (面積比 上位)
        y = 20
        for ratio, cid in stats[:15]:
            c = tuple(int(v) for v in self.palette[cid][::-1])
            cv2.rectangle(out, (8, y - 12), (22, y + 2), c, -1)
            put_text(out, f"{self.labels[int(cid)]} {ratio*100:.1f}%", (28, y))
            y += 20
        return out, [(self.labels[int(c)], r) for r, c in stats]


class InstanceSegmenter:
    def __init__(self, weights, conf, classes=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        # クラス名 -> ID に変換 (例: ["person"] -> [0])。None なら全クラス
        ids = {v: k for k, v in self.model.names.items()}
        self.classes = [ids[c] for c in classes] if classes else None

    def __call__(self, bgr):
        return self.model.predict(bgr, conf=self.conf, classes=self.classes, device=DEVICE, verbose=False)[0]

    # 泡 (マゼンタ) と紛れない色 (BGR): シアン, 黄, 緑, 橙, 青, 白
    COLORS = [(255, 255, 0), (0, 255, 255), (0, 255, 0), (0, 140, 255), (255, 80, 0), (255, 255, 255)]

    def draw(self, bgr, res, alpha=0.35, show_box=False):
        """インスタンスマスクを塗り + 輪郭で描画 (既定ではバウンディングボックスは描かない)"""
        out = bgr.copy()
        names = []
        if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
            return out, names
        h, w = out.shape[:2]
        for i, (cls, conf, box) in enumerate(zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())):
            name = res.names[int(cls)]
            names.append(name)
            color = self.COLORS[i % len(self.COLORS)]
            # マスク輪郭は元画像座標のポリゴンで得られる
            poly = res.masks.xy[i].astype(np.int32)
            if len(poly) < 3:
                continue
            fill = out.copy()
            cv2.fillPoly(fill, [poly], color)
            out = cv2.addWeighted(fill, alpha, out, 1 - alpha, 0)
            cv2.polylines(out, [poly], True, color, 2, cv2.LINE_AA)
            if show_box:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            top = tuple(poly[poly[:, 1].argmin()])
            put_label(out, f"{name} {conf:.2f}", (int(top[0]), max(int(top[1]) - 6, 14)), color)
        return out, names


# ---------------------------------------------------------------- util
def put_text(img, text, org, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def put_label(img, text, center, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = center[0] - w // 2, center[1]
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + 3), color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def process(frame, sem, ins, foam=None):
    # 描画順: semantic (背景の塗り分け) -> foam -> instance (YOLO) 。推論は常に元フレームに対して行う
    out, info = frame, {}
    if sem:
        out, info["semantic"] = sem.draw(out, sem(frame))
    if foam:
        out, info["foam"] = foam.draw(out, foam(frame))
    if ins:
        out, info["instance"] = ins.draw(out, ins(frame))
    return out, info


def main():
    # 接続情報 (PICAM_URL / PICAM_USER / PICAM_PASS) は .env から読む
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=ENV_FILE)
    env_file = pre.parse_known_args()[0].env_file
    load_env_file(env_file)

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--url", default=os.environ.get("PICAM_URL"),
                    help="ストリーム URL (未指定時は .env / 環境変数の PICAM_URL)")
    ap.add_argument("--mode", default="semantic",
                    help="semantic / instance / foam をカンマ区切りで組み合わせ可 (例: foam,instance)。both = semantic,instance")
    ap.add_argument("--seg-model", default="nvidia/segformer-b2-finetuned-ade-512-512",
                    help="例: nvidia/segformer-b0-finetuned-ade-512-512 (軽量) / -b4- (高精度)")
    ap.add_argument("--yolo", default="yolo11m-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--yolo-classes", help="YOLO で検出するクラスをカンマ区切りで限定 (例: person)")
    ap.add_argument("--image", help="ストリームの代わりにローカル画像を処理")
    ap.add_argument("--save", help="結果を mp4 (ストリーム) / 画像 (--image) で保存")
    ap.add_argument("--snapshot-dir", default="snapshots", help="'s' キーで保存する先")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--foam-model", help="学習済み泡モデルのフォルダ (例: models/foam_segformer)。指定時は閾値処理の代わりに使用")
    ap.add_argument("--roi", default="foam_roi.json", help="foam モードの水面 ROI 保存先")
    ap.add_argument("--calibrate", action="store_true", help="foam モードの水面 ROI をクリックで手動設定")
    ap.add_argument("--reset-roi", action="store_true", help="保存済み ROI を破棄して自動推定し直す")
    ap.add_argument("--csv", help="foam モードで泡の割合を時系列 CSV に追記")
    ap.add_argument("--save-fps", type=float, default=15.0,
                    help="保存動画の fps。実時間に合わせて間引き/複製するので再生速度は実時間と一致")
    ap.add_argument("--csv-interval", type=float, default=1.0, help="CSV 記録間隔 [秒]")
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    modes = set(args.mode.replace("both", "semantic,instance").split(","))
    if not modes <= {"semantic", "instance", "foam"}:
        ap.error(f"未知のモード: {modes - {'semantic', 'instance', 'foam'}}")
    args.mode = "+".join(m for m in ("semantic", "foam", "instance") if m in modes)  # 表示・ファイル名用
    sem = SemanticSegmenter(args.seg_model) if "semantic" in modes else None
    ins = InstanceSegmenter(args.yolo, args.conf, args.yolo_classes.split(",") if args.yolo_classes else None) if "instance" in modes else None
    foam = None
    if "foam" in modes:
        from foam import FoamSegmenter
        if args.reset_roi and os.path.exists(args.roi):
            os.remove(args.roi)
        if args.foam_model:
            from foam import FoamModelSegmenter
            foam = FoamModelSegmenter(args.foam_model)
        else:
            foam = FoamSegmenter(args.roi)

    if args.image:
        frame = cv2.imread(args.image)
        if foam and args.calibrate:
            foam.calibrate(frame)
        out, info = process(frame, sem, ins, foam)
        for k, v in info.items():
            print(k, v)
        dst = args.save or os.path.splitext(args.image)[0] + f"_{args.mode}.jpg"
        cv2.imwrite(dst, out)
        print("saved:", dst)
        return

    user, pw = os.environ.get("PICAM_USER"), os.environ.get("PICAM_PASS")
    if not user or not pw:
        raise SystemExit(f"PICAM_USER / PICAM_PASS が未設定です。{env_file} に記入するか環境変数で指定してください")
    # kill (SIGTERM) でも動画を正しく閉じて終了する
    import signal
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    if not args.url:
        raise SystemExit(f"ストリーム URL が未設定です。{env_file} の PICAM_URL か --url で指定してください")
    reader = MJPEGReader(args.url, (user, pw))
    reader.start()

    writer, last_seq, t0, n, last_log, next_write = None, 0, time.time(), 0, 0.0, 0.0
    os.makedirs(args.snapshot_dir, exist_ok=True)
    try:
        while reader.running or reader.frame is not None:
            seq, frame = reader.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                if not reader.running:
                    break
                continue
            last_seq = seq
            if foam and args.calibrate:
                foam.calibrate(frame)
                args.calibrate = False
            out, info = process(frame, sem, ins, foam)
            if args.csv and "foam" in info and time.time() - last_log >= args.csv_interval:
                header = "timestamp,foam_ratio,cx,cy,dx,dy\n"
                if os.path.exists(args.csv) and open(args.csv).readline() != header:
                    # 旧形式 (重心列なし) のログは退避
                    os.rename(args.csv, os.path.splitext(args.csv)[0] + time.strftime("_old_%Y%m%d%H%M%S.csv"))
                new = not os.path.exists(args.csv)
                fi = info["foam"]
                with open(args.csv, "a") as f:
                    if new:
                        f.write(header)
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{fi['foam_ratio']},"
                            f"{fi['cx']},{fi['cy']},{fi['dx']},{fi['dy']}\n")
                last_log = time.time()
            n += 1
            fps = n / (time.time() - t0)
            put_text(out, f"{args.mode}  {fps:.1f} fps  ({DEVICE})", (out.shape[1] - 260, 20))

            if args.save:
                now = time.time()
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.save_fps,
                                             (out.shape[1], out.shape[0]))
                    next_write = now
                rec = out.copy()
                put_text(rec, time.strftime("%Y-%m-%d %H:%M:%S"), (out.shape[1] - 200, out.shape[0] - 15))
                # 実時間に合わせて書き込む (処理が速ければ間引き、遅ければ同じフレームを複製)
                while next_write <= now:
                    writer.write(rec)
                    next_write += 1.0 / args.save_fps
            if not args.no_window:
                cv2.imshow("PiCam segmentation (q: quit, s: snapshot)", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    p = os.path.join(args.snapshot_dir, time.strftime("%Y%m%d_%H%M%S") + f"_{args.mode}.jpg")
                    cv2.imwrite(p, out)
                    print("snapshot:", p, info)
            elif n % 10 == 0:
                print(f"{fps:.1f} fps", info)
    except KeyboardInterrupt:
        pass
    finally:
        reader.running = False
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
