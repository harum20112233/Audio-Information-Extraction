#!/usr/bin/env python3
"""
data/raw 以下の音声を一括で文字起こしし、SBV2用の filelists を出力します。
バックエンドは OpenAI whisper か Hugging Face Transformers のどちらか（自動判定）。

出力:
  <out_dir>/<speaker>_train.txt
  <out_dir>/<speaker>_val.txt
  形式: <abs_wav>|<speaker>|<text>

例:
docker compose run --rm app \
  python -m src.whisper_to_filelist \
    --audio_dir data/raw \
    --speaker myvoice \
    --language ja \
    --whisper_model large-v3   # 例: openai-whisper の "medium" 等
    --device auto \
    --out_dir filelists
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch


# ===== backend 判定（元コード流用） =====
def detect_whisper_backend(model_name_or_path: str) -> str:
    """
    'hf' or 'openai' を返す。
    ローカルディレクトリに config.json があれば Hugging Face 形式('hf')とみなす。
    """
    p = Path(model_name_or_path)
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        return "hf"
    return "openai"


# ===== OpenAI whisper =====
def load_asr_openai(model_name: str, use_cuda: bool):
    import whisper as openai_whisper

    device = "cuda" if use_cuda else "cpu"
    print(f"[Whisper/openai] loading model: {model_name} on {device}")
    model = openai_whisper.load_model(model_name, device=device)
    return model


def transcribe_openai(model, wav_path: str, language: str) -> str:
    kwargs = {"fp16": model.device.type == "cuda", "task": "transcribe"}
    if language and language.lower() != "auto":
        kwargs["language"] = language
    out = model.transcribe(wav_path, **kwargs)
    return (out.get("text") or "").strip()


# ===== HF Transformers whisper =====
def load_asr_hf(model_name_or_path: str, use_cuda: bool, language: str):
    from transformers import pipeline as hf_asr_pipeline

    device_index = 0 if use_cuda else -1
    torch_dtype = torch.float16 if use_cuda else None
    # ★ 翻訳に振れないよう task="transcribe" を明示
    generate_kwargs = {}
    if language and language.lower() != "auto":
        lang_kw = (
            "japanese" if language.lower() in {"ja", "jpn", "japanese"} else language
        )
        generate_kwargs = {"language": lang_kw, "task": "transcribe"}

    print(
        f"[Whisper/HF] loading: {model_name_or_path} on {'cuda' if use_cuda else 'cpu'}"
    )
    asr = hf_asr_pipeline(
        task="automatic-speech-recognition",
        model=model_name_or_path,
        device=device_index,
        torch_dtype=torch_dtype,
        return_timestamps=False,
        generate_kwargs=generate_kwargs or None,
    )
    return asr


def transcribe_hf(asr_pipeline, wav_path: str) -> str:
    out = asr_pipeline(wav_path)
    if isinstance(out, dict):
        return (out.get("text") or "").strip()
    return str(out).strip()


# ===== ユーティリティ =====
def list_audio_files(root: Path, exts: List[str]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext.lower()}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    # 実ファイル & ソート重複排除
    files = sorted(set(p for p in files if p.is_file()))
    return files


# ===== 引数 =====
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Transcribe directory -> SBV2 filelists (no faster-whisper)"
    )
    p.add_argument(
        "--audio_dir", required=True, help="音声のルートディレクトリ（再帰検索）"
    )
    p.add_argument("--speaker", default="myvoice", help="filelistsに書く話者名")
    p.add_argument("--language", default="ja", help="言語（ja 推奨）")
    p.add_argument(
        "--whisper_model",
        default="small",
        help="openai-whisperモデル名 or HFローカル/リポジトリパス",
    )
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--out_dir", default="filelists", help="filelistsの出力先")
    p.add_argument("--val_ratio", type=float, default=0.05, help="検証データの比率")
    p.add_argument("--shuffle_seed", type=int, default=42, help="シャッフルseed")
    p.add_argument(
        "--exts",
        default=".wav,.flac,.m4a,.mp3,.ogg,.opus",
        help="対象拡張子（カンマ区切り）",
    )
    p.add_argument("--progress_every", type=int, default=50, help="進捗ログ間隔")
    return p.parse_args()


def main():
    args = parse_args()

    # デバイス判定（torchはHFでしか使わないが、存在しなくてもOK）
    use_cuda = (
        torch.cuda.is_available()
        if args.device.lower() == "auto"
        else (args.device.lower() == "cuda")
    )
    print(f"[Device] cuda={use_cuda}")

    audio_root = Path(args.audio_dir)
    assert audio_root.is_dir(), f"Not a directory: {audio_root}"

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    wavs = list_audio_files(audio_root, exts)
    if not wavs:
        print(f"[ERR] 音声が見つかりません: {audio_root} (exts={exts})")
        sys.exit(1)
    print(f"[Info] found {len(wavs)} files under {audio_root}")

    # backend 準備
    backend = detect_whisper_backend(args.whisper_model)
    print(f"[Whisper] backend={backend}  model={args.whisper_model}")

    if backend == "openai":
        model = load_asr_openai(args.whisper_model, use_cuda)

        def transcribe_fn(pth: str) -> str:
            return transcribe_openai(model, pth, args.language)

    else:
        asr = load_asr_hf(args.whisper_model, use_cuda, args.language)

        def transcribe_fn(pth: str) -> str:
            return transcribe_hf(asr, pth)

    # 一括で filelists 行を作成
    from random import Random

    rnd = Random(args.shuffle_seed)
    rows: List[str] = []
    pe = max(1, args.progress_every)

    for i, wav in enumerate(wavs, 1):
        wav_abs = str(wav.resolve())
        try:
            text = transcribe_fn(wav_abs)
        except Exception as e:
            print(f"[ERR] {wav_abs}: {e}")
            text = ""
        if text:
            # <abs_wav>|<speaker>|<text>
            rows.append(f"{wav_abs}|{args.speaker}|{text}")
        if i % pe == 0:
            print(f"[Progress] {i}/{len(wavs)} processed")

    if not rows:
        print(
            "[ERR] 文字起こし結果が0件でした。モデル/言語/ファイルを確認してください。"
        )
        sys.exit(3)

    # train/val に分割して出力
    rnd.shuffle(rows)
    n = len(rows)
    n_val = max(1, math.floor(n * args.val_ratio))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_train = out_dir / f"{args.speaker}_train.txt"
    p_val = out_dir / f"{args.speaker}_val.txt"

    p_val.write_text("\n".join(rows[:n_val]) + "\n", encoding="utf-8")
    p_train.write_text("\n".join(rows[n_val:]) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {p_train} ({n - n_val} rows)")
    print(f"[OK] wrote: {p_val}   ({n_val} rows)")


if __name__ == "__main__":
    main()
