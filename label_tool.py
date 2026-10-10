"""
疑似ラベルの手修正ツール (ブラウザで塗る)

  .venv/bin/python label_tool.py --pick 30     # 修正対象を選んで labels/queue.txt に書く (初回のみ)
  .venv/bin/python label_tool.py               # http://localhost:8765 を開いて修正

- 初期表示は dataset_fish/masks の疑似ラベル。保存すると labels/manual/<name>.png (値 0-3) に書き出す
- 保存後に build_dataset_fish.py を再実行すると、そのラベルが疑似ラベルの代わりに使われる
- 外部には公開しない (localhost のみで待ち受け)
"""
import argparse
import base64
import csv
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "dataset_fish")
MANUAL = os.path.join(ROOT, "labels", "manual")
QUEUE = os.path.join(ROOT, "labels", "queue.txt")


def pick(n, seed=0):
    """日・給餌の有無が偏らないように選ぶ。魚群ありを半分、残りは通常時/濁り時"""
    rows = [r for r in csv.DictReader(open(os.path.join(DATA, "meta.csv"))) if r["status"] == "ok"]
    rng = random.Random(seed)
    days = sorted({r["date"] for r in rows})
    fish_days = sorted({r["date"] for r in rows if float(r["fish_ratio"]) > 0.05})
    out = []
    for i in range(n):
        want_fish = i % 2 == 0
        day = fish_days[(i // 2) % len(fish_days)] if want_fish else days[(i // 2) % len(days)]
        cand = [r for r in rows if r["date"] == day and (float(r["fish_ratio"]) > 0.05) == want_fish
                and r["name"] not in out]
        cand = cand or [r for r in rows if r["date"] == day and r["name"] not in out]
        if cand:
            out.append(rng.choice(cand)["name"])
    os.makedirs(os.path.dirname(QUEUE), exist_ok=True)
    with open(QUEUE, "w") as f:
        f.write("\n".join(sorted(out)) + "\n")
    print(f"{len(out)} images -> {QUEUE}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send(200, open(os.path.join(ROOT, "label_tool.html"), "rb").read(), "text/html; charset=utf-8")
        if path == "/api/queue":
            names = [l.strip() for l in open(QUEUE) if l.strip()]
            items = [{"name": n, "done": os.path.exists(os.path.join(MANUAL, n + ".png"))} for n in names]
            return self.send(200, json.dumps(items).encode(), "application/json")
        kind, _, name = path.strip("/").partition("/")
        name = os.path.basename(name)  # パスの外に出ないように
        if kind == "image":
            p = os.path.join(DATA, "images", name + ".jpg")
            if os.path.exists(p):
                return self.send(200, open(p, "rb").read(), "image/jpeg")
        if kind == "mask":
            p = os.path.join(MANUAL, name + ".png")
            if not os.path.exists(p):
                p = os.path.join(DATA, "masks", name + ".png")
            if os.path.exists(p):
                return self.send(200, open(p, "rb").read(), "image/png")
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        kind, _, name = self.path.strip("/").partition("/")
        name = os.path.basename(name)
        if kind != "save" or not os.path.exists(os.path.join(DATA, "images", name + ".jpg")):
            return self.send(404, b"not found", "text/plain")
        body = self.rfile.read(int(self.headers["Content-Length"]))
        png = base64.b64decode(body.split(b",", 1)[-1])
        import cv2
        import numpy as np
        rgba = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        mask = rgba[..., 2] if rgba.ndim == 3 else rgba  # R チャンネル = クラス番号
        mask = np.clip(mask, 0, 3).astype(np.uint8)
        os.makedirs(MANUAL, exist_ok=True)
        cv2.imwrite(os.path.join(MANUAL, name + ".png"), mask)
        self.send(200, b"ok", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, help="修正対象を N 枚選んで labels/queue.txt を作り直す")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.pick:
        pick(args.pick)
        return
    print(f"http://localhost:{args.port}  (Ctrl+C で終了)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
