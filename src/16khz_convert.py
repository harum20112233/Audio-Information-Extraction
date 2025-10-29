"""
16kHz に変換するためのユーティリティ

概要:
    - m4a/mp3/wav/flac/ogg/webm 等の音声ファイルを 16kHz にリサンプリングします。
    - 既定では「出力は WAV」に強制変換します（拡張子 .wav で保存）。
    - `--out-ext` で出力拡張子を指定できます（例: wav/flac/mp3 等）。
    - 指定したファイルまたはディレクトリ内の音声ファイルを一括で変換します。
    - サブフォルダの再帰処理や、対象拡張子の指定、モノラル化の有無、ドライラン（実行せずに対象だけ確認）を選べます。

注意:
    - 変換には FFmpeg を使用します（pydub 経由）。コンテナ/Dockerfile では ffmpeg を導入済みです。
    - ロスレスでない形式（m4a/mp3/ogg等）は、再エンコードにより劣化する可能性があります。

使用例:
    - フォルダ配下の wav, m4a, mp3 をすべて 16kHz・WAV に（再帰的・モノラル化）
                    python -m src.16khz_convert --in ./data/in --recursive --mono

    - 出力を FLAC にしたい場合
                    python -m src.16khz_convert --in ./data/in --out-ext flac

    - 特定拡張子だけ対象（wav と flac のみ）を 16kHz に（再帰なし・ドライラン）
                    python -m src.16khz_convert --in ./data/in --ext wav flac --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _get_tqdm():
    try:  # 遅延インポート（環境に無い場合はダミーを返す）
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:  # pragma: no cover

        def _tqdm(x: Iterable, **kwargs):
            return x

        return _tqdm


DEFAULT_EXTS = [
    "wav",
    "m4a",
    "mp3",
    "flac",
    "ogg",
    "webm",
]


def ext_to_ffmpeg_format(ext: str) -> str | None:
    """拡張子から ffmpeg のフォーマット名へ変換。

    pydub/export で利用する format 名を返します。未対応は None。
    代表例:
      - .m4a はコンテナ mp4 として扱う
      - .aac は adts だが、本ツールのデフォルト対象からは外す
    """

    ext = ext.lower().lstrip(".")
    mapping = {
        "wav": "wav",
        "mp3": "mp3",
        "m4a": "mp4",
        "mp4": "mp4",
        "flac": "flac",
        "ogg": "ogg",
        "webm": "webm",
        # "aac": "adts",  # 必要なら対象に追加
    }
    return mapping.get(ext)


def find_audio_files(target: Path, recursive: bool, exts: List[str]) -> List[Path]:
    """対象パスから処理対象の音声ファイル一覧を抽出。"""
    if target.is_file():
        return [target]

    if not target.is_dir():
        raise FileNotFoundError(f"入力パスが存在しません: {target}")

    patterns = [f"*.{e.lower().lstrip('.')}" for e in exts]
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(p for p in target.rglob(pat) if p.is_file())
    else:
        for pat in patterns:
            files.extend(p for p in target.glob(pat) if p.is_file())
    # 重複排除 & ソート
    return sorted(set(files))


def convert_to_ext(
    src: Path,
    out_ext: str = "wav",
    sr: int = 16000,
    mono: bool = False,
    overwrite: bool = True,
) -> Tuple[bool, str]:
    """単一ファイルを 16kHz にリサンプリングし、指定拡張子で保存する。

    - デフォルトは WAV で保存（同ディレクトリ・同名で拡張子のみ変更）
    - out_ext が入力拡張子と同じなら実質上書きと同じ挙動
    - 失敗時は (False, reason)
    """

    try:
        from pydub import AudioSegment  # type: ignore

        fmt_out = ext_to_ffmpeg_format(out_ext)
        if fmt_out is None:
            return False, f"unsupported output ext: .{out_ext}"

        dst = src.with_suffix("." + out_ext.lstrip("."))
        if dst.exists() and not overwrite:
            return True, f"skip(exists): {dst}"

        audio = AudioSegment.from_file(src)
        if sr:
            audio = audio.set_frame_rate(sr)
        if mono:
            audio = audio.set_channels(1)

        audio.export(dst, format=fmt_out)
        return True, f"ok: {dst}"

    except Exception as e:
        return False, f"error: {src} -> .{out_ext} ({e.__class__.__name__}: {e})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "音声ファイルを 16kHz に一括変換し、既定で WAV に保存します（--out-ext で変更可）。"
        )
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="入力パス（ファイルまたはディレクトリ）",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="ディレクトリ指定時、サブフォルダも再帰的に探索",
    )
    p.add_argument(
        "--ext",
        nargs="*",
        default=DEFAULT_EXTS,
        help=f"対象拡張子（デフォルト: {' '.join(DEFAULT_EXTS)}）",
    )
    p.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="出力サンプリングレート（デフォルト: 16000）",
    )
    p.add_argument("--mono", action="store_true", help="モノラル化（チャネル=1）")
    p.add_argument(
        "--out-ext",
        default="wav",
        help=("出力拡張子（既定: wav）。例: wav, flac, mp3, ogg, webm, m4a。"),
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="出力先が存在する場合はスキップ（既定は上書き）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には変換せず、対象一覧のみ表示",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.in_path).resolve()
    exts = [e.lower().lstrip(".") for e in args.ext]

    files = find_audio_files(target, recursive=args.recursive, exts=exts)
    if not files:
        print(f"[INFO] 対象ファイルが見つかりませんでした: {target}")
        sys.exit(0)

    print(
        f"[START] {len(files)} files | sr={args.sr} mono={args.mono} out-ext={args.out_ext} overwrite={not args.no_overwrite}"
    )

    if args.dry_run:
        for p in files:
            print("-", p)
        print("[DRY-RUN] 変換は行っていません")
        return

    ok = 0
    ng = 0
    for p in _get_tqdm()(files, desc="Resampling 16kHz"):
        success, msg = convert_to_ext(
            p,
            out_ext=args.out_ext,
            sr=args.sr,
            mono=args.mono,
            overwrite=(not args.no_overwrite),
        )
        if success:
            ok += 1
        else:
            ng += 1
        # tqdm 進捗に混ざると読みにくいので、失敗のみ逐次表示
        if not success:
            print(msg)

    print("\n[SUMMARY]")
    print(f"  converted: {ok}")
    print(f"  failed   : {ng}")


if __name__ == "__main__":
    main()
