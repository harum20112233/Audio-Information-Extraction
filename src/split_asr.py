"""
split_asr.py

例）in.csv を 8:1:1 に分割、出力は train_data/asr_simple/、シード値は42
docker compose run --rm app python -m src.split_asr \
  --input_csv in.csv \
  --out_dir train_data/asr_simple \
  --val_ratio 0.1 --test_ratio 0.1 \
  --seed 42

  csv構成:
  - audio_path: 音声ファイルのパス（相対パスまたは絶対パス）
  - text: 音声に対応するテキスト

"""

import argparse, csv, os, random


def parse_args():
    p = argparse.ArgumentParser("Split ASR CSV (audio_path,text) into train/valid/test")
    p.add_argument(
        "--input_csv", required=True, help="入力CSV（audio_path,text の2列）"
    )
    p.add_argument("--out_dir", required=True, help="出力ディレクトリ")
    p.add_argument("--val_ratio", type=float, default=0.1, help="valid 比率")
    p.add_argument("--test_ratio", type=float, default=0.1, help="test 比率")
    p.add_argument("--seed", type=int, default=42, help="シャッフル乱数シード")
    p.add_argument(
        "--check_audio",
        action="store_true",
        help="存在しない音声行を除外（audio_path を CSV からの相対で解決）",
    )
    return p.parse_args()


# CSV読み込み、必須カラム確認utf-8-sigでBOM対応
def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        # 必須カラム確認
        need = {"audio_path", "text"}
        if set(r.fieldnames or []) & need != need:
            raise ValueError(
                f"CSVヘッダに {need} が必要です（見つかった列: {r.fieldnames}）"
            )
        return list(r)


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "text"])
        for r in rows:
            w.writerow([r["audio_path"], r["text"]])


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(args.input_csv))
    rows = read_rows(args.input_csv)

    # 音声存在チェック（任意）
    if args.check_audio:
        kept = []
        for r in rows:
            rel = r["audio_path"]
            p = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
            if os.path.isfile(p):
                kept.append(r)
            else:
                print(f"[WARN] missing audio: {p} → この行を除外")
        rows = kept

    random.seed(args.seed)
    random.shuffle(rows)

    n = len(rows)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test = rows[:n_test]
    valid = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val :]

    out_root = os.path.abspath(args.out_dir)
    write_rows(os.path.join(out_root, "train.csv"), train)
    write_rows(os.path.join(out_root, "valid.csv"), valid)
    write_rows(os.path.join(out_root, "test.csv"), test)

    print(f"[OK] total={n}  train={len(train)}  valid={len(valid)}  test={len(test)}")
    print(f"[OUT] {out_root}")


if __name__ == "__main__":
    main()
