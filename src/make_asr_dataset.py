"""
============================================
src/make_asr_dataset.py

CSV（pipeline出力）→ クリップ生成 → train/valid/test.csv を作るユーティリティ。
必須列: file, start_sec, end_sec, transcript

使い方（例）:
docker compose run --rm app \
  python -m src.make_asr_dataset \
    --input_csv data/out/result2min.csv \
    --audio_root samples/amagasaki \
    --out_dir train_data/asr \
    --val_ratio 0.1 --test_ratio 0.1 \
    --min_sec 0.6 --max_sec 20

出力構成:
out_dir/
├── clips/        # ex_000001.wav など（16kHz/mono）
├── train.csv     # audio_path,text  （out_dirからの相対パス）
├── valid.csv
└── test.csv
============================================
"""

from __future__ import (
    annotations,
)  # 前方参照の型ヒントを有効化（Python 3.7+）  # noqa: E402

import argparse  # コマンドライン引数の定義・解析  # noqa: E402
import os  # パス操作・ディレクトリ作成等  # noqa: E402
import csv  # CSVの読み書き  # noqa: E402
import math  # ここでは未使用だが、閾値計算などを拡張する余地  # noqa: E402
import random  # データ分割時のシャッフルに使用  # noqa: E402
import re  # テキスト正規化時の正規表現置換に使用  # noqa: E402
import unicodedata  # 全角半角の統一（NFKC）に使用  # noqa: E402
from typing import List, Tuple  # 型ヒントのためのList/Tuple  # noqa: E402
from pydub import (
    AudioSegment,
)  # 任意フォーマット読み込み・WAV出力・リサンプルに使用  # noqa: E402


def normalize_text_ja(s: str) -> str:  # 日本語テキストの最小限の正規化を行う関数
    s = unicodedata.normalize("NFKC", s)  # 全角・半角の揺れを互換分解・整形で吸収
    s = re.sub(
        r"\s+", " ", s
    ).strip()  # 連続空白・改行等を単一スペースに畳み、前後空白を除去
    return s  # 正規化済み文字列を返す


def parse_args():  # CLI引数の定義とパースを行う関数
    p = argparse.ArgumentParser("Build ASR dataset from pipeline CSV")  # ヘルプ用説明文
    p.add_argument(
        "--input_csv", required=True, help="pipeline出力CSVのパス"
    )  # 入力CSV
    p.add_argument(  # 元音声の探索ルート（サブフォルダ差異にも対応しやすくする）
        "--audio_root", required=True, help="元音声ファイルのルート（samples 等）"
    )
    p.add_argument(
        "--out_dir", default="data/asr", help="出力先ディレクトリ"
    )  # 出力ルート
    p.add_argument(  # クリップ保存先のサブディレクトリ名
        "--clips_dirname", default="clips", help="クリップ保存ディレクトリ名"
    )
    p.add_argument(
        "--min_sec", type=float, default=0.6, help="短すぎる区間は除外"
    )  # 長さ下限
    p.add_argument(
        "--max_sec", type=float, default=30.0, help="長すぎる区間は除外"
    )  # 長さ上限
    p.add_argument("--val_ratio", type=float, default=0.1)  # 検証用比率
    p.add_argument("--test_ratio", type=float, default=0.1)  # テスト用比率
    p.add_argument("--random_seed", type=int, default=42)  # 乱数シード（再現性担保）
    p.add_argument(  # クリップ出力時のサンプリングレート（Whisper等の標準16kを既定）
        "--export_sr", type=int, default=16000, help="出力wavのサンプリングレート"
    )
    return p.parse_args()  # パース結果（Namespace）を返す


def ensure_dir(p: str):  # ディレクトリが無ければ作成（再帰的）
    os.makedirs(p, exist_ok=True)  # 既存でもOK（例外を出さない）


def load_rows(path: str) -> List[dict]:  # 入力CSVを読み込み、各行を辞書として返す
    with open(path, newline="", encoding="utf-8") as f:  # UTF-8で開く（BOM無し想定）
        r = csv.DictReader(f)  # ヘッダ行をキーにして辞書化
        return list(r)  # 全行をメモリに展開（データ量が大きい場合はストリーム化も検討）


def main():  # エントリポイント
    args = parse_args()  # CLI引数を取得
    random.seed(args.random_seed)  # シャッフルの再現性確保

    out_root = os.path.abspath(args.out_dir)  # 出力ルートを絶対パス化（相対揺れ防止）
    clips_dir = os.path.join(out_root, args.clips_dirname)  # クリップ保存先のフルパス
    ensure_dir(out_root)  # 出力ルート作成
    ensure_dir(clips_dir)  # クリップフォルダ作成

    rows = load_rows(args.input_csv)  # 入力CSVをメモリに読む

    examples: List[Tuple[str, str]] = (
        []
    )  # 後でCSVに書く（audio_path, text）のペア格納用

    audio_cache: dict[str, AudioSegment] = (
        {}
    )  # 同一ファイルの複数区間切り出しに備えキャッシュ

    for i, r in enumerate(rows):  # 各行（区間）をループ処理
        try:  # 行単位で例外を握りつぶし、処理継続性を担保
            src_fname = r["file"]  # 元音声のファイル名（相対or絶対を想定）
            s = float(r["start_sec"])  # 区間開始（秒）
            e = float(r["end_sec"])  # 区間終了（秒）
            text = (r.get("transcript") or "").strip()  # 文字起こし（空ならスキップ）
            if not text:  # テキストが欠損・空白のみなら
                continue  # 学習サンプルとしては不適なので除外

            dur = e - s  # 区間の長さ（秒）
            if dur < args.min_sec or dur > args.max_sec:  # 長さが許容外なら
                continue  # 除外して次行へ

            # 元音声の実体パスを解決（絶対指定か、audio_rootと結合するか）
            src_path = (
                os.path.join(args.audio_root, src_fname)
                if not os.path.isabs(src_fname)
                else src_fname
            )
            if not os.path.isfile(src_path):  # 想定パスに実ファイルが無い場合
                # サブフォルダ構成の差異に備えて、basenameのみで探索（例: CSVが相対/絶対混在）
                alt = os.path.join(args.audio_root, os.path.basename(src_fname))
                if os.path.isfile(alt):  # 代替パスが存在するなら
                    src_path = alt  # そちらを採用
                else:  # それでも見つからない場合
                    print(f"[WARN] missing audio: {src_path}")  # 警告を出して
                    continue  # この行はスキップ

            # 音声ファイルの読み込み（都度読むとI/Oが重いのでキャッシュ活用）
            if src_path not in audio_cache:  # 未ロードなら
                audio_cache[src_path] = AudioSegment.from_file(src_path)  # 任意拡張子OK
            audio = audio_cache[src_path]  # キャッシュから取得

            # ミリ秒単位スライスで区間切り出し（pydubはミリ秒指定）
            clip = audio[int(s * 1000) : int(e * 1000)]  # 半開区間で抽出

            # 16kHz/monoへ正規化（ASR学習での一般的設定）
            clip = clip.set_frame_rate(args.export_sr).set_channels(
                1
            )  # SR/チャネル変換
            out_wav = os.path.join(clips_dir, f"ex_{i:06d}.wav")  # 連番で保存先パス作成
            clip.export(out_wav, format="wav")  # WAVとしてエクスポート

            # テキストを正規化（NFKC・空白畳み。辞書・表記揺れ等は用途に応じ拡張）
            text = normalize_text_ja(text)  # 簡易正規化

            # 出力CSVは out_root 起点の相対パスで記録（移植性を高める）
            rel_path = os.path.relpath(out_wav, out_root)  # 相対化
            examples.append((rel_path, text))  # サンプルとして追加

        except Exception as ex:  # 想定外エラーが出ても学習データ生成全体は継続
            print(f"[WARN] skip row {i}: {ex}")  # 行番号付きで原因を出力

    # データをシャッフル（分割に偏りが出ないようにする）
    random.shuffle(examples)  # 乱数シードにより再現可能

    n = len(examples)  # 総サンプル数
    n_test = int(n * args.test_ratio)  # テスト件数（切り捨て）
    n_val = int(n * args.val_ratio)  # 検証件数（切り捨て）
    test = examples[:n_test]  # 先頭からテスト
    val = examples[n_test : n_test + n_val]  # 続いてバリデーション
    train = examples[n_test + n_val :]  # 残りを学習用

    def save_csv(name: str, data: List[Tuple[str, str]]):  # CSV書き出しのヘルパ
        path = os.path.join(out_root, name)  # 出力パス
        with open(path, "w", newline="", encoding="utf-8") as f:  # 追記ではなく上書き
            w = csv.writer(f)  # シンプルなCSVライタ
            w.writerow(["audio_path", "text"])  # ヘッダ
            w.writerows(data)  # 本体
        print(f"[OK] {name}: {len(data)} rows")  # 進捗ログ

    save_csv("train.csv", train)  # 学習データを書き出し
    save_csv("valid.csv", val)  # 検証データを書き出し
    save_csv("test.csv", test)  # テストデータを書き出し

    print(f"[DONE] out_dir = {out_root}")  # 出力ルートを表示
    print(f"       clips  = {clips_dir}")  # クリップ保存先を表示


if __name__ == "__main__":  # スクリプトとして直接実行された場合にのみmain()を呼ぶ
    main()  # メイン処理を開始
