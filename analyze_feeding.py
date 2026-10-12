"""
給餌イベントの解析 (レポート用)

timeline: 赤いコップ給餌の各日について、元画像 (archive/、5 秒間隔) にトリガーと魚群モデルを順に適用し、
          トリガー時刻・魚群の割合・狙い点の時系列を出す
scan    : 全日の日中を一定間隔で走査し、トリガーが給餌以外で誤って成立した回数を数える

  .venv/bin/python analyze_feeding.py timeline
  .venv/bin/python analyze_feeding.py scan --step 30
出力: analysis/timeline_<日付>.csv, analysis/timeline_summary.csv, analysis/trigger_scan.csv, analysis/frames/
"""
import argparse
import csv
import glob
import os
import time

import cv2
import numpy as np

from fish import FishModelSegmenter, FishSchoolSegmenter
from trigger import RedCupTrigger, draw_state
from foam import FoamSegmenter

SIZE = (1280, 720)
# 赤いコップ給餌 (events.csv の A、ユーザー確認済み) の解析範囲
WINDOWS = {
    "2026-10-05": ("17:10", "17:50"),
    "2026-10-06": ("14:50", "16:15"),
    "2026-10-07": ("09:55", "10:40"),
    "2026-10-08": ("11:35", "12:20"),
    "2026-10-09": ("15:40", "16:25"),
}
FEED_EVENTS = {  # 実際に赤いコップで給餌していた区間 (誤検出の判定用)
    "2026-10-05": [("17:21", "17:24")],
    "2026-10-06": [("15:03", "15:14"), ("15:24", "15:47"), ("15:52", "15:54")],
    "2026-10-07": [("10:08", "10:14")],
    "2026-10-08": [("11:47", "11:52")],
    "2026-10-09": [("15:51", "15:57")],
}


def tsec(hms):
    p = [int(v) for v in hms.replace("-", ":").split(":")]
    return p[0] * 3600 + p[1] * 60 + (p[2] if len(p) > 2 else 0)


def hms(t):
    return f"{int(t) // 3600:02d}:{int(t) % 3600 // 60:02d}:{int(t) % 60:02d}"


def day_frames(day, t0, t1, step=5):
    out, last = [], -1e9
    for p in sorted(glob.glob(f"archive/{day}/*.jpg")):
        t = tsec(os.path.basename(p)[:-4])
        if t0 <= t <= t1 and t - last >= step:
            out.append((t, p))
            last = t
    return out


def load(p):
    return cv2.resize(cv2.imread(p), SIZE, interpolation=cv2.INTER_AREA)


def timeline(args):
    os.makedirs("analysis/frames", exist_ok=True)
    seg = FishModelSegmenter(args.model)
    roi = FoamSegmenter("foam_roi.json").roi_mask((SIZE[1], SIZE[0]))
    summary = []
    for day, (a, b) in WINDOWS.items():
        clock = {"t": 0.0}
        trig = RedCupTrigger(roi, clock=lambda: clock["t"])
        frames = day_frames(day, tsec(a), tsec(b))
        rows, t_trig, prev_aim = [], None, None
        for t, p in frames:
            clock["t"] = t
            img = load(p)
            feeding, info = trig(img)
            r = seg(img)
            if feeding and t_trig is None:
                t_trig = t
                out = draw_state(FishSchoolSegmenter.draw(img, r), True, "redcup", info)
                cv2.imwrite(f"analysis/frames/trigger_{day}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            aim = r["aim"]
            jump = None if aim is None or prev_aim is None else float(np.hypot(aim[0] - prev_aim[0], aim[1] - prev_aim[1]))
            prev_aim = aim
            rows.append({"time": hms(t), "t": t, "cup": int(info["hit"]), "feeding": int(feeding),
                         "fish_ratio": round(r["fish_ratio"], 4), "foam_ratio": round(r["foam_ratio"], 4),
                         "aim_x": None if aim is None else round(aim[0], 1), "aim_y": None if aim is None else round(aim[1], 1),
                         "aim_jump": None if jump is None else round(jump, 1)})
        for row in rows:
            row["sec_from_trigger"] = None if t_trig is None else row["t"] - t_trig
        with open(f"analysis/timeline_{day}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["time", "sec_from_trigger", "cup", "feeding", "fish_ratio", "foam_ratio",
                                   "aim_x", "aim_y", "aim_jump"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        fr = np.array([r["fish_ratio"] for r in rows])
        after = [r for r in rows if t_trig is not None and r["t"] >= t_trig]
        first = next((r for r in after if r["fish_ratio"] >= 0.05), None)
        peak = max(after, key=lambda r: r["fish_ratio"]) if after else None
        jumps = [r["aim_jump"] for r in after if r["aim_jump"] is not None and r["fish_ratio"] >= 0.05]
        feed_start = tsec(FEED_EVENTS[day][0][0])
        s = {"date": day, "trigger": None if t_trig is None else hms(t_trig),
             "trigger_delay_sec": None if t_trig is None else t_trig - feed_start,
             "school_appear_sec": None if first is None else first["t"] - t_trig,
             "peak_fish_ratio": None if peak is None else peak["fish_ratio"],
             "peak_sec": None if peak is None else peak["t"] - t_trig,
             "school_frames": int((fr >= 0.05).sum()), "frames": len(rows),
             "aim_jump_median_px": None if not jumps else round(float(np.median(jumps)), 1)}
        summary.append(s)
        print(s, flush=True)
    with open("analysis/timeline_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, summary[0].keys())
        w.writeheader()
        w.writerows(summary)


def scan(args):
    """全日 06-18 時を step 秒ごとに走査し、トリガー (給餌開始) の成立を数える"""
    roi = FoamSegmenter("foam_roi.json").roi_mask((SIZE[1], SIZE[0]))
    rows = []
    for day in sorted(os.path.basename(d) for d in glob.glob("archive/*")):
        clock = {"t": 0.0}
        trig = RedCupTrigger(roi, clock=lambda: clock["t"], off_sec=args.off_sec)
        prev = False
        for t, p in day_frames(day, 6 * 3600, 18 * 3600 + 3599, args.step):
            clock["t"] = t
            feeding, info = trig(load(p))
            if feeding and not prev:
                ev = FEED_EVENTS.get(day, [])
                true = any(tsec(a) - 120 <= t <= tsec(b) + 120 for a, b in ev)
                rows.append({"date": day, "time": hms(t), "true_feeding": int(true), "file": p})
                print(rows[-1], flush=True)
            prev = feeding
    with open("analysis/trigger_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["date", "time", "true_feeding", "file"])
        w.writeheader()
        w.writerows(rows)
    days_with_feed = {d for d in FEED_EVENTS}
    hit_days = {r["date"] for r in rows if r["true_feeding"]}
    print(f"trigger onsets: {len(rows)}  true: {sum(r['true_feeding'] for r in rows)}  "
          f"false: {sum(1 - r['true_feeding'] for r in rows)}  feeding days detected: {len(hit_days)}/{len(days_with_feed)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timeline", "scan"])
    ap.add_argument("--model", default="models/fish_segformer")
    ap.add_argument("--step", type=int, default=30, help="scan の間隔 [秒]")
    ap.add_argument("--off-sec", type=float, default=300.0, help="scan で給餌終了とみなす無検出時間 [秒]")
    args = ap.parse_args()
    t0 = time.time()
    timeline(args) if args.cmd == "timeline" else scan(args)
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
