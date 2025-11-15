# src/whisper_wwer_cer_eval.py

"""
input_csv に記載された複数の音声ファイルを Whisper で文字起こしし、
（参照テキストがあれば）CER/WERを計算して出力CSVに保存する。
audio_path 列,text 列を持つCSVを入力とし、
出力CSVには audio_path, pred_text, ref_text, cer, wer 列が含まれる。
例
docker compose run --rm app \
  python -m src.whisper_wer_cer_eval \
    --input_csv data/raw/name_raw_in/name_eval_input.csv \
    --audio_root . \
    --whisper_model models/whisper-small-name-raw \
    --language ja \
    --device auto \
    --out_csv data/out/name_raw_eval_small.csv

"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch


# ============== Whisper backend 判定 ==============
def detect_whisper_backend(model_name_or_path: str) -> str:
    """
    'hf' or 'openai' を返す。
    ローカルディレクトリに config.json があれば Hugging Face 形式('hf')とみなす。
    """
    p = Path(model_name_or_path)
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        return "hf"
    return "openai"


# ============== 文字起こし実行 ==============
def load_asr_openai(model_name: str, use_cuda: bool):
    import whisper as openai_whisper

    device = "cuda" if use_cuda else "cpu"
    print(f"[Whisper/openai] loading model: {model_name} on {device}")
    model = openai_whisper.load_model(model_name, device=device)
    return model


def transcribe_openai(model, wav_path: str, language: str) -> str:
    kwargs = {"fp16": model.device.type == "cuda"}
    if language and language.lower() != "auto":
        kwargs["language"] = language
    out = model.transcribe(wav_path, **kwargs)
    return (out.get("text") or "").strip()


def load_asr_hf(model_name_or_path: str, use_cuda: bool, language: str):
    from transformers import pipeline as hf_asr_pipeline

    device_index = 0 if use_cuda else -1
    torch_dtype = torch.float16 if use_cuda else None
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


# ============== CER/WER（参照テキストがある場合のみ） ==============
def _levenshtein(a: List[str], b: List[str]) -> int:
    """汎用レーベンシュタイン距離（a,b はトークン列 または 文字列のリスト）"""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1, dp[j - 1] + 1, prev + cost  # deletion  # insertion
            )  # substitution
            prev = cur
    return dp[m]


def normalize_ja(s: str) -> str:
    # ざっくり：全角空白→半角、前後空白除去。必要なら正規化を追加。
    return s.replace("\u3000", " ").strip()


def cer(ref: str, hyp: str) -> Optional[float]:
    ref = normalize_ja(ref)
    hyp = normalize_ja(hyp)
    if len(ref) == 0:
        return None
    dist = _levenshtein(list(ref), list(hyp))
    return dist / len(ref)


def wer(ref: str, hyp: str) -> Optional[float]:
    # 日本語はスペース区切りが弱いので参考値扱い。必要なら分かち書きに差し替え可。
    ref = normalize_ja(ref)
    hyp = normalize_ja(hyp)
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if len(ref_tokens) == 0:
        return None
    dist = _levenshtein(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


# ============== CSV I/O ==============
def read_manifest(csv_path: str) -> List[Tuple[str, Optional[str]]]:
    """
    必須列: audio_path
    任意列: text（参照テキスト）
    """
    rows: List[Tuple[str, Optional[str]]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        assert "audio_path" in r.fieldnames, "CSVに 'audio_path' 列が必要です"
        for row in r:
            ap = (row.get("audio_path") or "").strip()
            tx = (row.get("text") or "").strip() if "text" in row else None
            if ap:
                rows.append((ap, tx if tx else None))
    return rows


def write_results(out_csv: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    header = ["audio_path", "pred_text", "ref_text", "cer", "wer"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "audio_path": r["audio_path"],
                    "pred_text": r["pred_text"],
                    "ref_text": r.get("ref_text", ""),
                    "cer": None if r.get("cer") is None else round(r["cer"], 4),
                    "wer": None if r.get("wer") is None else round(r["wer"], 4),
                }
            )
    print(f"[OK] wrote: {out_csv} (rows={len(rows)})")


# ============== メイン ==============
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch transcribe short WAVs (optional CER/WER)")
    p.add_argument(
        "--input_csv", required=True, help="音声リストCSV（列: audio_path[, text]）"
    )
    p.add_argument(
        "--audio_root",
        default=None,
        help="CSVの audio_path が相対パスの場合のルート（任意）",
    )
    p.add_argument(
        "--whisper_model",
        default="small",
        help="openai-whisperモデル名 or HFローカル/リポジトリパス",
    )
    p.add_argument(
        "--language", default="ja", help="言語。日本語は 'ja' 推奨（デフォルト）"
    )
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument(
        "--out_csv",
        required=True,
        help="出力CSV（列: audio_path, pred_text, ref_text, cer, wer）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # デバイス判定
    use_cuda = (
        torch.cuda.is_available()
        if args.device.lower() == "auto"
        else (args.device.lower() == "cuda")
    )
    print(f"[Device] cuda={use_cuda}")

    manifest = read_manifest(args.input_csv)
    assert len(manifest) > 0, "入力CSVに行がありません"

    # backend 準備
    backend = detect_whisper_backend(args.whisper_model)
    print(f"[Whisper] backend={backend}")

    if backend == "openai":
        model = load_asr_openai(args.whisper_model, use_cuda)

        def transcribe_fn(pth: str) -> str:
            return transcribe_openai(model, pth, args.language)

    else:
        asr = load_asr_hf(args.whisper_model, use_cuda, args.language)

        def transcribe_fn(pth: str) -> str:
            return transcribe_hf(asr, pth)

    # 一括処理
    results: List[dict] = []
    base = Path(args.audio_root) if args.audio_root else None
    for i, (ap, ref) in enumerate(manifest, 1):
        wav_path = str((base / ap) if base else Path(ap))
        if not os.path.isfile(wav_path):
            print(f"[WARN] not found: {wav_path}  -> skip")
            continue
        try:
            pred = transcribe_fn(wav_path)
        except Exception as e:
            print(f"[ERR] {wav_path}: {e}")
            pred = ""

        row = {"audio_path": ap, "pred_text": pred, "ref_text": ref or ""}
        if ref:
            c = cer(ref, pred)
            w = wer(ref, pred)
            row["cer"] = c
            row["wer"] = w
        results.append(row)

        if i % 50 == 0:
            print(f"[Progress] {i}/{len(manifest)} files processed")

    write_results(args.out_csv, results)
    # 参考：全体CER/WER（refがある行のみ）
    cer_vals = [r["cer"] for r in results if r.get("cer") is not None]
    wer_vals = [r["wer"] for r in results if r.get("wer") is not None]
    if cer_vals:
        print(
            f"[Summary] mean CER = {sum(cer_vals)/len(cer_vals):.4f}  (n={len(cer_vals)})"
        )
    if wer_vals:
        print(
            f"[Summary] mean WER = {sum(wer_vals)/len(wer_vals):.4f}  (n={len(wer_vals)})"
        )


if __name__ == "__main__":
    main()
