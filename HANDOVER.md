# 引き継ぎ書: PiCam 水槽画像の泡・魚群セグメンテーション

作成: 2026-10-09 / 前回の作業環境: Mac Studio (Apple Silicon, MPS), Python 3.11.5

---

## 1. 目的と現状

PiCam (Raspberry Pi カメラ) で撮影している養殖水槽 (手前の円形水槽) の映像から、

- **水面の白い泡** の領域・割合・重心をリアルタイムに計測する
- **給餌シーン** を過去画像から抽出し、給餌時に水面へ集まる **魚群** の領域を計測する

ことを目指している。

| 項目 | 状態 |
|---|---|
| ライブ映像の泡検出 (閾値処理) + 割合・重心・CSV 記録 | ✅ 動作確認済み (約 23-33 fps) |
| YOLO インスタンスセグメンテーションとの同時実行 | ✅ (`--mode foam,instance`、約 9-14 fps) |
| 泡の学習モデル (SegFormer-b0) | ✅ 学習済み。日中・通常運転時のみ有効 |
| 過去画像から給餌シーン抽出 | ✅ 7 日分を走査済み (給餌が明確 9 件、赤いコップ給餌 3 件) |
| 魚群セグメンテーション (閾値処理) | ⚠️ 試作。**魚かどうかの検証が未完了** (§6) |

---

## 2. 新しい PC でのセットアップ

```bash
git clone https://github.com/ba2vwbus13/picam-foam-segmentation.git
cd picam-foam-segmentation
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env && chmod 600 .env     # PICAM_URL / PICAM_USER / PICAM_PASS を記入
```

- `.env` は git 管理外。カメラ URL は `https://<カメラのホスト>/stream` 形式 (前の PC の `.env` からコピー)
- カメラは **Tailscale のネットワーク内** にあるため、新しい PC も同じ tailnet に参加している必要がある
- YOLO (`yolo11m-seg.pt`) と SegFormer の重みは初回実行時に自動ダウンロード
- GPU: Apple Silicon は MPS、NVIDIA は CUDA を自動選択 (`stream_segmentation.py` の `DEVICE`)。
  CUDA 環境では `requirements.txt` の torch を CUDA 版に入れ替えること
- 仮想環境に入る場合: `source .venv/bin/activate` (以下のコマンドは `python` だけでよくなる)

動作確認 (カメラ不要):

```bash
.venv/bin/python stream_segmentation.py --image samples/full/f1.jpg --mode foam
# -> foam {'foam_ratio': 0.4707, ...} と samples/full/f1_foam.jpg が出れば OK
```

---

## 3. カメラ・データについて

- ライブ: MJPEG `/stream` (1280×720)、HTTP Basic 認証
- アーカイブ API (同じ認証): 5 秒間隔、1920×1080。2026-10-09 時点で 10/03〜10/09 の 7 日分があった (保持期間の仕様は未確認)
  - `/api/calendar/days` → 日付一覧
  - `/api/calendar/hours?date=YYYY-MM-DD` → 時間一覧
  - `/api/calendar/images?date=...&hour=HH&interval=0` → 画像一覧
  - `/api/calendar/image?date=...&hour=...&filename=...` → 画像本体
  - 実効速度は約 1〜3 枚/秒 (サーバー側が律速。並列数を増やしても速くならない)。時々タイムアウトする
- **画角はほぼ固定** (日によって最大約 48px ずれる程度)。ずれは `dataset_tools.ViewAligner` (ORB 特徴点 + 射影変換) で補正できる
- 週末 (10/03 土・10/04 日) は人の出入りなし

### リポジトリに含めていないデータ (前の PC にのみ存在)

| フォルダ | 内容 | 容量 | 再作成方法 |
|---|---|---|---|
| `dataset/` | 泡学習用 204 枚 + 疑似ラベル | 84MB | `python build_dataset.py` (※古い日付がアーカイブから消えていれば同じ画像は再取得できない) |
| `feeding/` | 人が写る 1,563 フレーム、給餌抽出結果、魚群結果 | 845MB | `find_feeding.py` → `find_red_cup.py --src feeding/feeding_clear` → `fish.py` |
| `samples/turbid/` | 濁った水の比較画像 | 小 | — |

職員の顔が写っているため公開リポジトリには入れていない。必要なら前の PC から外付けドライブ・学内ストレージ等で直接コピーすること。

---

## 4. ファイル構成と使い方

| ファイル | 役割 |
|---|---|
| `stream_segmentation.py` | メイン。ストリーム受信、`--mode semantic/instance/foam` (カンマで組合せ)、表示・CSV・mp4 保存 |
| `foam.py` | 泡: `FoamSegmenter` (閾値処理)、`FoamModelSegmenter` (学習モデル) |
| `foam_roi.json` | 現在の画角での水面楕円 (手で内縁 10 点からフィット) |
| `fish.py` | 魚群: `FishSchoolSegmenter` |
| `dataset_tools.py` | アーカイブ API (`Archive`)、画角合わせ (`ViewAligner`) |
| `build_dataset.py` / `train_foam.py` | 泡モデルのデータ作成・学習・評価 |
| `find_feeding.py` / `find_red_cup.py` | 人がいる場面 / 赤いコップ給餌の抽出 |
| `models/foam_segformer/` | 学習済み泡モデル (14MB) と学習履歴・評価値 |

よく使うコマンド:

```bash
# ライブ: 泡 (閾値処理) + CSV + 動画
.venv/bin/python stream_segmentation.py --mode foam --csv foam_log.csv --save foam_live.mp4
# ライブ: 泡 (学習モデル) + YOLO で人だけ
.venv/bin/python stream_segmentation.py --mode foam,instance --foam-model models/foam_segformer --yolo-classes person
# 給餌シーン抽出 (全日 06-18 時、1 分間隔 → 候補の前後 1 分を 5 秒間隔で) 約 1 時間
.venv/bin/python find_feeding.py
.venv/bin/python find_red_cup.py --src feeding/feeding_clear --dst feeding/red_cup
.venv/bin/python fish.py --src feeding/red_cup --dst feeding/fish
```

ウィンドウ操作: `q` 終了、`s` スナップショット。

---

## 5. これまでの結果と分かったこと

### 泡 (閾値処理)
- 水面楕円 ROI 内で「Lab の L が Otsu 閾値より明るい & HSV 彩度 < 60」を泡とする
- ライブで泡割合 約 45〜51%、重心は水面中心よりやや上 (奥)
- 自動 ROI 推定 (SegFormer の tank + 沈殿物の凸包) は外壁の影を拾い不安定 → 手動 ROI を採用

### 泡の学習モデル
- 疑似ラベル (閾値処理の出力) で SegFormer-b0 を学習。train 10/03-07 (149 枚)、val 10/08 (25)、test 10/09 (30)
- test: 泡 IoU 0.81、水面 IoU 0.87、泡割合の誤差 平均 1.3 pt。推論 19 fps
- **閾値処理の模倣なので、閾値処理の弱点 (縁の誤検出・濁り時の過大評価) も受け継いでいる**

### 給餌シーン (`feeding/events.csv`)
- 人が水槽付近にいたイベント 75 件 → 目視で分類: 給餌が明確 9 / 給餌の可能性 23 / 通行 43
- **赤いコップでの給餌** (ユーザー確認済みの給餌方法): 10/07 10:08–10:13、10/08 11:47–11:51、10/09 15:51–15:56
- 黄色いバケツの作業 (10/07 13:30–13:42, 14:18–14:30) は濁りを伴わず、給餌かどうか未確認
- **給餌直後に水が茶色く濁り、泡と茶色い模様が一時的に消える**。約 1 時間後 (10/08 は 13:00 頃) に元の模様に戻る

### 魚群 (`fish.py`)
- 給餌中、魚は「暗く黄色みが少ない」(L 52–64, b−128 ≈ 1–6)、茶色い水は「明るく黄色い」(L 71–107, b−128 ≈ 12–24)
- 判定: `L < 68` かつ `b−128 < 9` の画素密度 > 0.5 の領域。泡と人 (YOLO) は除外
- 給餌中の魚群割合: 10/07 12–23%、10/08 10–22%、10/09 22–41%。給餌前は 0%、通常時 0–5%

---

## 6. 未解決の課題・要確認事項 (優先度順)

1. **魚群判定が本当に魚か未検証**
   - 10/07・10/08 は魚群が毎回「手前 (カメラ側) の帯」に出る。手前は水面の映り込みが少なく暗く見えるため、魚がいなくても暗い可能性がある
   - 給餌後に人が去っても魚群割合は 20% → 11% (30 分) → 0% (1 時間) と緩やかに減る。魚が残っているのか、見え方の問題か区別できていない
   - 10/09 の中央〜右側は魚の背中が目視でも確認でき、こちらは魚とみてよさそう
   - → 現場を知る人に、数枚の画像で「魚がいる範囲」を確認してもらうのが最優先
2. **通常時の「茶色い模様」は沈殿物か魚群か**
   - これまで沈殿物と仮定してきたが、拡大すると魚の密集にも見える。魚群であれば泡割合の解釈 (= 魚のいない部分の割合?) が変わる
3. **魚群判定の閾値が固定値** — カメラの自動露出で画面全体の明るさが変わると判定がぶれる (10/09 15:55 に 24% → 0% へ急落、同時に水面の明るさ中央値 79 → 99)。白い縁の明るさで正規化するなどの対策が必要
4. **泡判定の弱点**
   - Otsu は常に二分するため、泡がほぼ無い/濁った水面では泡を過大評価 (薄い茶色の水や縁の内壁を泡と判定)
   - 給餌中の画像の泡割合 (`foam_ratio`) は使わないこと
   - 対策案: 明るさの下限を固定で設ける、学習データに濁った状態を手で修正して加える
5. **重心は画像座標 (斜め視点) のまま** — 水槽平面上の位置にするには楕円→円の射影補正が必要
6. 夜間 (18 時以降) のデータは未検討

---

## 7. 次の一手 (提案)

1. 課題 1・2 を現場で確認し、魚群/泡/沈殿物の定義を決める
2. 評価用に 20〜30 枚を手でラベル付け (CVAT / Label Studio)。閾値処理と学習モデルを「本当の正解」で比較する
3. 魚群も泡と同様に疑似ラベル → 手修正 → SegFormer で学習し、クラスを「水面外 / 水 / 泡 / 魚群」に拡張
4. 給餌時刻 (赤いコップ検出) と泡割合・魚群割合の時系列を並べて解析
5. 魚群を `stream_segmentation.py` に `--mode fish` として組み込み、ライブで計測

---

## 8. 注意事項

- **このリポジトリは公開 (Public)**。`.env`、人物が写る画像、カメラの URL はコミットしないこと (`.gitignore` 済み)
- 以前のコミットからカメラ URL は除去済み
- アーカイブの保持期間は未確認 (確認時は 7 日分のみ)。残したい期間の画像は早めに取得しておくこと
