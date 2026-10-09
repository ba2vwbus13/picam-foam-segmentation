"""
泡セグメンテーション (SegFormer) の学習と評価

  .venv/bin/python train_foam.py                 # 学習 -> models/foam_segformer/
  .venv/bin/python train_foam.py --eval-only     # test セットで評価のみ

クラス: 0=水面外, 1=水面 (泡以外), 2=泡
"""
import argparse
import csv
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["background", "surface", "foam"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
INFER_SIZE = (768, 432)  # 推論時の入力サイズ (W, H)


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


class FoamDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, train, crop=512):
        rows = list(csv.DictReader(open(os.path.join(root, "meta.csv"))))
        self.names = [r["name"] for r in rows if r["status"] == "ok" and r["split"] == split]
        self.root, self.train, self.crop = root, train, crop

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = cv2.imread(os.path.join(self.root, "images", n + ".jpg"))
        mask = cv2.imread(os.path.join(self.root, "masks", n + ".png"), cv2.IMREAD_GRAYSCALE)
        if self.train:
            # 拡大縮小 + ランダムクロップ
            s = random.uniform(max(0.6, (self.crop + 1) / img.shape[0]), 1.0)  # 短辺がクロップ以上になる範囲
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            c = self.crop
            y = random.randint(0, img.shape[0] - c)
            x = random.randint(0, img.shape[1] - c)
            img, mask = img[y:y + c, x:x + c], mask[y:y + c, x:x + c]
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            # 照明変化への耐性: 明るさ・コントラスト・色味・ぼかし
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-6, 6)) % 180
            hsv[..., 1] *= random.uniform(0.7, 1.3)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            img = img * random.uniform(0.7, 1.3) + random.uniform(-25, 25)
            img = np.clip(img, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            img = cv2.resize(img, INFER_SIZE, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, INFER_SIZE, interpolation=cv2.INTER_NEAREST)
        return to_tensor(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(mask)).long()


def load_model(path_or_name, new_head=False):
    kw = dict(num_labels=3, id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)},
              ignore_mismatched_sizes=True) if new_head else {}
    return SegformerForSemanticSegmentation.from_pretrained(path_or_name, **kw).to(DEVICE)


@torch.no_grad()
def predict(model, x, size):
    logits = model(pixel_values=x).logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False).argmax(1)


@torch.no_grad()
def evaluate(model, ds):
    model.eval()
    inter, union = np.zeros(3), np.zeros(3)
    ratio_err = []
    for i in range(len(ds)):
        x, y = ds[i]
        p = predict(model, x[None].to(DEVICE), y.shape)[0].cpu()
        for c in range(3):
            inter[c] += ((p == c) & (y == c)).sum().item()
            union[c] += ((p == c) | (y == c)).sum().item()
        # 泡の割合 = 泡 / (水面 + 泡)
        def ratio(m):
            s = ((m == 1) | (m == 2)).sum().item()
            return (m == 2).sum().item() / max(s, 1)
        ratio_err.append(abs(ratio(p) - ratio(y)))
    iou = inter / np.maximum(union, 1)
    return {"iou_" + c: round(float(v), 4) for c, v in zip(CLASSES, iou)} | {
        "miou": round(float(iou.mean()), 4), "foam_ratio_mae": round(float(np.mean(ratio_err)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--base", default="nvidia/segformer-b0-finetuned-ade-512-512")
    ap.add_argument("--out", default="models/foam_segformer")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0), random.seed(0), np.random.seed(0)

    val, test = FoamDataset(args.data, "val", False), FoamDataset(args.data, "test", False)
    if args.eval_only:
        model = load_model(args.out)
        print("test:", evaluate(model, test))
        return

    train = FoamDataset(args.data, "train", True)
    print(f"device={DEVICE} train={len(train)} val={len(val)} test={len(test)}")
    dl = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4,
                                     persistent_workers=True, drop_last=True)
    model = load_model(args.base, new_head=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr * 10, total_steps=args.epochs * len(dl),
                                                pct_start=0.1)
    best, hist = -1, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = F.interpolate(model(pixel_values=x).logits, size=y.shape[-2:], mode="bilinear",
                                   align_corners=False)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(model, val)
        hist.append({"epoch": ep, "loss": round(tot / len(dl), 4), **m})
        print(f"ep{ep:3d} loss {tot/len(dl):.4f} val {m} ({time.time()-t0:.0f}s)", flush=True)
        if m["iou_foam"] > best:
            best = m["iou_foam"]
            model.save_pretrained(args.out)
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
    model = load_model(args.out)
    res = evaluate(model, test)
    json.dump(res, open(os.path.join(args.out, "test_metrics.json"), "w"), indent=1)
    print("best val foam IoU", best, "| test:", res)


if __name__ == "__main__":
    main()
