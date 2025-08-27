# ============================================
# src/make_asr_dataset.py
# CSV( pipeline出力 ) → 学習用クリップ + train/valid/test.csv を生成
# 入力CSV 例の必須列:
#   file,start_sec,end_sec,transcript
# ※ file は元音声のファイル名（相対/絶対は --audio_root と組み合わせて解決）
#
# 実行例:
# docker compose run --rm app \
#   python -m src.make_asr_dataset \
#     --input_csv data/out/result32s.csv \
#     --audio_root samples/amagasaki \
#     --out_dir data/asr \
#     --val_ratio 0.1 --test_ratio 0.1 \
#     --min_sec 0.6 --max_sec 20
#
# できるもの:
# data/asr/
# ├── clips/               # ex_000001.wav 等（16kHz/mono）
# ├── train.csv            # audio_path,text  （out_dir起点の相対パス）
# ├── valid.csv
# └── test.csv

# ============================================

from __future__ import annotations
import argparse, os, csv, math, random, re, unicodedata
from typing import List, Tuple
from pydub import AudioSegment


def normalize_text_ja(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_args():
    p = argparse.ArgumentParser("Build ASR dataset from pipeline CSV")
    p.add_argument("--input_csv", required=True, help="pipeline出力CSVのパス")
    p.add_argument(
        "--audio_root", required=True, help="元音声ファイルのルート（samples 等）"
    )
    p.add_argument("--out_dir", default="data/asr", help="出力先ディレクトリ")
    p.add_argument(
        "--clips_dirname", default="clips", help="クリップ保存ディレクトリ名"
    )
    p.add_argument("--min_sec", type=float, default=0.6, help="短すぎる区間は除外")
    p.add_argument("--max_sec", type=float, default=30.0, help="長すぎる区間は除外")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument(
        "--export_sr", type=int, default=16000, help="出力wavのサンプリングレート"
    )
    return p.parse_args()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def main():
    args = parse_args()
    random.seed(args.random_seed)

    out_root = os.path.abspath(args.out_dir)
    clips_dir = os.path.join(out_root, args.clips_dirname)
    ensure_dir(out_root)
    ensure_dir(clips_dir)

    rows = load_rows(args.input_csv)

    examples: List[Tuple[str, str]] = []  # (audio_path, text)

    # バッファ: 元音声をキャッシュ（同一ファイルの複数区間に対応）
    audio_cache = {}

    for i, r in enumerate(rows):
        try:
            src_fname = r["file"]  # 例: amagasaki__2014_10_28_32s.mp3
            s = float(r["start_sec"])
            e = float(r["end_sec"])
            text = (r.get("transcript") or "").strip()
            if not text:
                continue

            dur = e - s
            if dur < args.min_sec or dur > args.max_sec:
                continue

            # 元音声の実体パスを解決
            src_path = (
                os.path.join(args.audio_root, src_fname)
                if not os.path.isabs(src_fname)
                else src_fname
            )
            if not os.path.isfile(src_path):
                # サブフォルダ構成の差異に備えて多めに探す
                alt = os.path.join(args.audio_root, os.path.basename(src_fname))
                if os.path.isfile(alt):
                    src_path = alt
                else:
                    print(f"[WARN] missing audio: {src_path}")
                    continue

            # 音声ロード（キャッシュ）
            if src_path not in audio_cache:
                audio_cache[src_path] = AudioSegment.from_file(src_path)
            audio = audio_cache[src_path]

            # 区間切り出し
            clip = audio[int(s * 1000) : int(e * 1000)]

            # 16kHz/mono に正規化して保存
            clip = clip.set_frame_rate(args.export_sr).set_channels(1)
            out_wav = os.path.join(clips_dir, f"ex_{i:06d}.wav")
            clip.export(out_wav, format="wav")

            # テキスト正規化（必要に応じて拡張）
            text = normalize_text_ja(text)

            # 例のリストへ
            rel_path = os.path.relpath(
                out_wav, out_root
            )  # CSVは out_root からの相対にする
            examples.append((rel_path, text))

        except Exception as ex:
            print(f"[WARN] skip row {i}: {ex}")

    # シャッフル & 分割
    random.shuffle(examples)
    n = len(examples)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test = examples[:n_test]
    val = examples[n_test : n_test + n_val]
    train = examples[n_test + n_val :]

    def save_csv(name: str, data: List[Tuple[str, str]]):
        path = os.path.join(out_root, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["audio_path", "text"])
            w.writerows(data)
        print(f"[OK] {name}: {len(data)} rows")

    save_csv("train.csv", train)
    save_csv("valid.csv", val)
    save_csv("test.csv", test)

    print(f"[DONE] out_dir = {out_root}")
    print(f"       clips  = {clips_dir}")


if __name__ == "__main__":
    main()
