"""
16kHz に変換するためのユーティリティ（上書き・同名保存）

概要:
  - 指定したファイルまたはディレクトリ内の音声ファイルを一括で 16kHz にリサンプリングします。
  - 変換後は「元の場所・同じ拡張子・同じファイル名」で上書き保存します（バックアップは取りません）。
  - サブフォルダの再帰処理や、対象拡張子の指定、モノラル化の有無、ドライラン（実行せずに対象だけ確認）を選べます。

注意:
  - 変換には FFmpeg を使用します（pydub 経由）。コンテナ/Dockerfile では ffmpeg を導入済みです。
  - ロスレスでない形式（m4a/mp3/ogg等）は、再エンコードにより劣化する可能性があります。

使用例:
  - フォルダ配下の wav, m4a, mp3 をすべて 16kHz に（再帰的・モノラル化）
          python -m src.16khz_convert --in ./data/in --recursive --mono

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


def convert_in_place(
    src: Path,
    sr: int = 16000,
    mono: bool = False,
) -> Tuple[bool, str]:
    """単一ファイルを 16kHz（既定）へリサンプリングし、同じ場所へ上書き保存する。

    - 形式は拡張子に合わせて維持（例: .m4a -> mp4 コンテナで書き戻し）
    - 失敗時は (False, reason)
    """

    try:
        from pydub import AudioSegment  # type: ignore

        fmt = ext_to_ffmpeg_format(src.suffix)
        if fmt is None:
            return False, f"unsupported ext: {src.name}"

        audio = AudioSegment.from_file(src)
        if sr:
            audio = audio.set_frame_rate(sr)
        if mono:
            audio = audio.set_channels(1)

        # 上書き保存。pydub は出力先から format を推測しないため明示する
        audio.export(src, format=fmt)
        return True, f"ok: {src}"

    except Exception as e:
        return False, f"error: {src} ({e.__class__.__name__}: {e})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "音声ファイルを 16kHz に一括変換して上書き保存します（拡張子は維持）。"
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
        f"[START] {len(files)} files | sr={args.sr} mono={args.mono} overwrite=in-place"
    )

    if args.dry_run:
        for p in files:
            print("-", p)
        print("[DRY-RUN] 変換は行っていません")
        return

    ok = 0
    ng = 0
    for p in _get_tqdm()(files, desc="Resampling 16kHz"):
        success, msg = convert_in_place(p, sr=args.sr, mono=args.mono)
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
