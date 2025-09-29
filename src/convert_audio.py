# save as: src/convert_audio.py
# 使い方（例）:
#   docker compose run --rm app python -m src.convert_audio --in ./data/in/example --out ./data/in/converted_example --to wav --sr 16000 --mono --recursive --keep-structure

"""
===============================================================================
【概要】
  本スクリプトは、指定フォルダ配下にある多数の .m4a 音声ファイルを一括で
  .wav または .mp3 に変換するためのユーティリティです。
  サブフォルダの再帰探索、出力サンプリングレート変更（例: 16kHz）、
  モノラル化（チャネル=1）、mp3ビットレート指定、ディレクトリ構造の維持、
  既存ファイルの上書き可否などをオプションで制御できます。

【想定用途】
  - 学習用データ作成のために、m4a を一括で wav(16kHz/mono) に揃える
  - 端末間での持ち回り用に、m4a を mp3(192kbps) に一括変換する
  - フォルダ構造を維持したまま、別の出力ルートに変換結果を保存する

【前提】
  - 変換処理には pydub と FFmpeg を使用します。
    pydub は FFmpeg バイナリに依存するため、実行環境に FFmpeg が必要です。
    - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
    - Windows    : 公式配布の FFmpeg を PATH に通すか、pydub.AudioSegment.converter へパス指定
    - macOS      : Homebrew などで `brew install ffmpeg`

【使い方（コマンド例）】
  1) すべてを 16kHz/モノラルの WAV へ（サブフォルダも含め、フォルダ構造を維持）
     python -m src.convert_audio --in ./m4a_folder --out ./converted \
       --to wav --sr 16000 --mono --recursive --keep-structure

  2) すべてを 192kbps の MP3 へ（同名ファイルがあれば上書き）
     python -m src.convert_audio --in ./m4a_folder --out ./converted \
       --to mp3 --bitrate 192k --overwrite

【注意点】
  - 入力拡張子は .m4a 固定です（mp4/aac なども対象にしたい場合は、find_m4a_files を拡張）。
  - --overwrite を付けない場合、出力先に同名ファイルが存在するとスキップします。
  - 非常に大きなファイルでも扱えるよう、逐次読み込み→逐次書き出しを行いますが、
    マシンのメモリ・CPU・I/O 性能に依存します。
  - 変換失敗（コーデック不一致、破損ファイル等）はエラーとして記録し、処理は続行します。

【出力ログ】
  - 変換開始時に、対象ファイル数と主な設定を表示
  - 変換完了後に、成功件数および失敗件数を要約
  - 失敗・スキップ（上書き無効で既存ファイル）した項目は、簡易メッセージを表示

===============================================================================
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# pydub は内部で FFmpeg を呼び出して変換を行うため、環境に FFmpeg が必要です。
# 例）Ubuntu: sudo apt-get install -y ffmpeg
from pydub import AudioSegment

try:
    # 進捗バー表示用（インストールされていない場合はダミーで代替します）
    from tqdm import tqdm
except Exception:
    # tqdm が無い環境でもスクリプトが落ちないよう、同等インタフェースのダミー関数を定義
    def tqdm(x, **kwargs):
        return x


def find_m4a_files(root: Path, recursive: bool) -> List[Path]:
    """
    指定ディレクトリ直下、または再帰的に .m4a ファイルを列挙する。

    引数:
      root (Path)       : 入力のルートディレクトリ
      recursive (bool)  : True の場合、サブフォルダも含めて探索

    戻り値:
      List[Path] : 見つかった .m4a ファイルのパス一覧
    """
    if recursive:
        # rglob はサブディレクトリも含めて再帰的に探索
        return [p for p in root.rglob("*.m4a") if p.is_file()]
    else:
        # glob は直下のみを対象に探索
        return [p for p in root.glob("*.m4a") if p.is_file()]


def make_out_path(
    in_path: Path, in_root: Path, out_root: Path, to_ext: str, keep_structure: bool
) -> Path:
    """
    出力先ファイルの保存パスを組み立てるヘルパー。

    - keep_structure=True:
        入力ルートからの相対パス構造を出力側にそのまま再現し、拡張子のみ変換。
        例）in_root/a/b/c.m4a -> out_root/a/b/c.wav

    - keep_structure=False:
        出力先直下にフラットに並べる。重複回避のため親フォルダ名を接頭辞に付与。
        例）in_root/a/b/c.m4a -> out_root/b__c.wav

    引数:
      in_path (Path)       : 入力ファイルのフルパス
      in_root (Path)       : 入力ルートディレクトリ
      out_root (Path)      : 出力ルートディレクトリ
      to_ext (str)         : 出力拡張子（"wav" or "mp3"）
      keep_structure (bool): ディレクトリ構造維持フラグ

    戻り値:
      Path : 出力ファイルのフルパス
    """
    if keep_structure:
        # 入力ルートからの相対パスを計算し、拡張子を目的のものに差し替え
        rel = in_path.relative_to(in_root).with_suffix("." + to_ext)
        return (out_root / rel).resolve()
    else:
        # フラット出力。親ディレクトリ名 + 元ファイル名 で衝突しにくくする
        rel_name = f"{in_path.parent.name}__{in_path.stem}.{to_ext}"
        return (out_root / rel_name).resolve()


def convert_one(
    src: Path,
    dst: Path,
    to_fmt: str,
    sr: int | None,
    mono: bool,
    bitrate: str | None,
    overwrite: bool,
) -> Tuple[bool, str]:
    """
    単一ファイルを指定フォーマットへ変換する。

    処理の流れ:
      1) 出力ファイルが既に存在し、overwrite=False の場合はスキップ
      2) 入力ファイルを pydub.AudioSegment で読み込み（FFmpeg 経由）
      3) サンプリングレート指定があれば set_frame_rate で変更
      4) モノラル指定があれば set_channels(1) でチャネル数を変更
      5) export() で指定フォーマットへ書き出し（mp3 の場合はビットレート指定可）
      6) 成功/失敗メッセージを返す（成功: True, 失敗: False）

    引数:
      src (Path)       : 入力ファイルパス
      dst (Path)       : 出力ファイルパス
      to_fmt (str)     : 出力フォーマット（"wav" or "mp3"）
      sr (int|None)    : 出力サンプリングレート（None の場合は元を維持）
      mono (bool)      : True ならモノラル（チャネル=1）へ変換
      bitrate (str|None): mp3 時のビットレート（例: "128k", "192k"）
      overwrite (bool) : True なら既存ファイルを上書き

    戻り値:
      (bool, str) : (成功フラグ, ログメッセージ)
    """
    try:
        # 既存ファイルがあり、上書き不可なら処理をスキップ
        if dst.exists() and not overwrite:
            return True, f"skip (exists): {dst}"

        # 出力ディレクトリを作成（多階層でも OK）
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 入力ファイル（m4a）を読み込み。
        # ここで FFmpeg が見つからない場合は例外が発生するため、事前にインストール確認を。
        audio = AudioSegment.from_file(src)

        # 出力サンプリングレート指定（例: 16000 で ASR 向けに揃える）
        if sr:
            audio = audio.set_frame_rate(sr)

        # モノラル化指定（ASR 学習や推論でチャネル=1を要求する場合が多い）
        if mono:
            audio = audio.set_channels(1)

        # export 時の追加引数を組み立て（主に mp3 のビットレート）
        export_kwargs = {}
        if to_fmt == "mp3" and bitrate:
            export_kwargs["bitrate"] = bitrate

        # 目的フォーマットで書き出し
        audio.export(dst, format=to_fmt, **export_kwargs)

        # 相対表示（2階層上を基準に短く見せる工夫。失敗時は絶対パスのまま）
        return (
            True,
            f"ok: {src.name} -> {dst.relative_to(dst.parents[1]) if dst.parents else dst}",
        )
    except Exception as e:
        # 例外は握りつぶさずメッセージ化。上流で集計してまとめて表示する。
        return False, f"error: {src} -> {dst} ({e.__class__.__name__}: {e})"


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を定義・解析する。
    主なオプション:
      --in            : 入力フォルダ（.m4a が大量に入っているルート）
      --out           : 出力フォルダ
      --to            : 出力フォーマット "wav" または "mp3"（デフォルト "wav"）
      --sr            : 出力サンプリングレート（例: 16000）。未指定なら元を維持
      --mono          : モノラル化フラグ（付与するとチャネル=1）
      --bitrate       : mp3 のビットレート（例: "128k", "192k", "256k"）
      --recursive     : サブフォルダも再帰的に変換
      --keep-structure: 入力側のディレクトリ構造を出力でも維持
      --overwrite     : 既存ファイルがあっても上書き
    """
    p = argparse.ArgumentParser(
        "大量の .m4a ファイルを .wav または .mp3 に一括変換するツール"
    )
    p.add_argument(
        "--in",
        dest="in_dir",
        required=True,
        help="入力フォルダ（.m4a が大量に入ったルート）",
    )
    p.add_argument("--out", dest="out_dir", required=True, help="出力フォルダ")
    p.add_argument(
        "--to", choices=["wav", "mp3"], default="wav", help="出力フォーマット"
    )
    p.add_argument(
        "--sr", type=int, default=None, help="出力サンプリングレート（例: 16000）"
    )
    p.add_argument("--mono", action="store_true", help="モノラル化（チャネル=1）")
    p.add_argument(
        "--bitrate", default="192k", help="mp3時のビットレート（例: 128k, 192k, 256k）"
    )
    p.add_argument(
        "--recursive", action="store_true", help="サブフォルダも再帰的に変換"
    )
    p.add_argument(
        "--keep-structure",
        action="store_true",
        help="入力のディレクトリ構造を出力側に維持",
    )
    p.add_argument("--overwrite", action="store_true", help="既存ファイルを上書き")
    return p.parse_args()


def main():
    """
    エントリポイント。
    - 入力/出力ディレクトリの存在確認
    - 対象 .m4a ファイルの列挙
    - 指定オプションに従って順次変換
    - 変換結果の要約を標準出力に表示
    """
    args = parse_args()
    in_root = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    # 入力フォルダの存在チェック（存在しない場合は AssertError を投げて終了）
    assert in_root.is_dir(), f"入力フォルダが見つかりません: {in_root}"
    # 出力フォルダが存在しない場合は作成
    out_root.mkdir(parents=True, exist_ok=True)

    # .m4a を列挙（--recursive 有無で探索範囲が変わる）
    files = find_m4a_files(in_root, recursive=args.recursive)
    if not files:
        print(f"[INFO] .m4a が見つかりませんでした: {in_root}")
        sys.exit(0)

    to_ext = args.to.lower()
    ok_count = 0
    err_count = 0
    logs: List[str] = []

    # 変換開始メッセージ（主な設定を明示しておくと再現性が高い）
    print(
        f"[START] {len(files)} files | to={to_ext} sr={args.sr or 'keep'} mono={args.mono} "
        f"{'bitrate='+args.bitrate if to_ext=='mp3' else ''} keep_structure={args.keep_structure}"
    )

    # 各ファイルを順次変換。tqdm が存在すれば進捗バー表示、無ければ通常ループ。
    for src in tqdm(files, desc="Converting"):
        dst = make_out_path(src, in_root, out_root, to_ext, args.keep_structure)
        success, log = convert_one(
            src=src,
            dst=dst,
            to_fmt=to_ext,
            sr=args.sr,
            mono=args.mono,
            bitrate=args.bitrate if to_ext == "mp3" else None,
            overwrite=args.overwrite,
        )
        logs.append(log)
        if success:
            ok_count += 1
        else:
            err_count += 1

    # 結果要約の表示
    print("\n[SUMMARY]")
    print(f"  converted: {ok_count}")
    print(f"  failed   : {err_count}")
    if err_count:
        print("\n[FAILED OR SKIPPED ITEMS]")
        # 失敗のみを抽出して列挙（スキップはあえて除外。必要なら条件を変更）
        for line in logs:
            if line.startswith("error"):
                print("  -", line)


if __name__ == "__main__":
    main()
