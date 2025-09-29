## dockerfile

```
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

```

## requirements.txt

```
openai-whisper==20231117      # 音声認識（文字起こし）ライブラリ。ffmpegと一緒に使うことが多い
transformers==4.56.1          # 感情分析などで使うHugging Faceのモデル運用ライブラリ
pyannote.audio==3.1.1         # 話者分離/話者ダイアライゼーション向けの定番ライブラリ
soundfile==0.13.1               # ベースイメージに libsndfile が入っていればOKですが、Python側の soundfile を入れておくと安心。
pydub==0.25.1                 # シンプルな音声の分割/結合などをPythonで行うための補助ライブラリ
pandas==2.3.2                 # 結果をCSV出力する際などに便利なデータフレーム処理
accelerate==1.10.1            # GPU/CPU切替や分散などHugging Face系の実行を簡単にする補助
huggingface_hub==0.34.4       # モデルのダウンロード/認証（HFトークン利用）などに使用
# llm-book/bert-base-japanese-v3-wrime-sentiment は MeCab系トークナイザ（MecabTokenizer） を使うので、Python 版の MeCab ラッパー fugashi と 辞書（ipadic か unidic-lite） が必須
fugashi==1.3.2                # 形態素解析ライブラリ（日本語処理）
unidic-lite==1.0.8            # 軽量辞書（ipadic でも可）
debugpy==1.8.16                # VSCode などのデバッガーを使うためのライブラリ
datasets==4.0.0                # データセット管理ライブラリ
peft==0.17.1                     # LoRA/PEFT
evaluate==0.4.5                       # モデル評価用ライブラリ
jiwer==3.0.4                       # WER/CER 計算
torchcodec==0.1.0                     # 音声コーデック系ライブラリ


# conda が ruamel-yaml<0.18 を要求するため固定
ruamel.yaml==0.17.40
ruamel.yaml.clib==0.2.8
numpy==2.0.2
```

## コード

```
"""
============================================
src/pipeline.py 〜 超ていねいコメント版 〜
--------------------------------------------
これは「音声 → (区間分割) → 文字起こし → 感情分析 → CSV保存」
という一直線の処理“パイプライン”を 1 本のスクリプトで実行します。

【全体像】
  入力音声
    ├─ ① 区間分割（話者分離pyannote.audio or 無音検出pyDub）
    ├─ ② 文字起こし（Whisper）
    ├─ ③ 感情分析（Transformers）
    └─ ④ CSV 出力

【使い方（コンテナ内での実行例）】
  docker compose run --rm app python -m src.pipeline \
    --in samples/amagasaki/amagasaki__2014_10_28_2min.mp3 \
    --out data/out/result-small.csv \
    --whisper_model small \
    --language ja \
    --device auto


  ※ --language ja を付けると日本語の精度が安定しやすいです。
  ※ --sentiment_model は任意。指定しない場合は
     1) 日本語モデルを試す（要ネット/場合によりHFトークン）
     2) ダメなら英語モデル（SST-2）にフォールバックします。
  【話者分離（pyannote）を使いたい場合】
  - 環境変数 USE_PYANNOTE=1 にする
  - HUGGINGFACE_TOKEN を .env に設定（モデルによっては同意/認証が必要）
  → 上記が満たされていれば pyannote で話者分離を試み、
     失敗したら自動で「無音検出（単一話者）」にフォールバックします。
  【前提ライブラリ】
  whisper, transformers, pyannote.audio, pydub, pandas, torch
  （Dockerfile/requirements.txt でインストール済み）
============================================
"""

from __future__ import annotations  # ｜ 型ヒントの前方参照（Python 3.10系互換のため）
import argparse  # ｜ コマンドライン引数のパース
import os  # ｜ OSパス/環境変数など
import csv  # ｜ CSV書き出し
from datetime import datetime  # ｜ タイムスタンプ用
import tempfile  # ｜ 一時ファイル（区間音声の切り出しに使う）
import warnings  # ｜ モデル取得失敗時などの警告表示

# ========= 数値計算 / GPU利用（PyTorch） =========
import torch  # ｜ PyTorch（CUDAが使えるかの判定や、Whisperのデバイス指定で使用）

# ========= 音声入出力 / 無音検出（pydub） =========
from pydub import AudioSegment  # ｜ 音声ファイルを読み書き・切り出し
from pydub.silence import detect_nonsilent  # ｜ 非無音（発話っぽい）区間の検出

# # ========= 文字起こし（OpenAI Whisper） =========
# import whisper  # ｜ Whisper公式実装（pipパッケージ）

# ========= 感情分析（Hugging Face Transformers） =========
from transformers import pipeline as hf_pipeline  # ｜ 便利な推論パイプラインAPI

# 学習したローカルのライブラリを用いたいときに用いる
from pathlib import Path

# ========= 環境変数での動作切り替え（pyannote利用可否など） =========
USE_PYANNOTE: bool = (
    os.environ.get("USE_PYANNOTE", "0") == "1"
)  # ｜ "1"で pyannote を使う
HF_TOKEN: str | None = os.environ.get("HUGGINGFACE_TOKEN")  # ｜ Hugging Face のトークン

# ========= VAD（無音検出）の既定パラメータ =========
#   ここを調整するだけで“細かめに切る/大まかに切る”の変更ができます。
VAD_MIN_SILENCE_LEN_MS: int = 400  # ｜ これ以上続く静けさを“無音”とみなす（ms）
VAD_SILENCE_MARGIN_DB: int = 16  # ｜ 平均音量(dBFS)から何dB下を“無音”とみなすか
VAD_KEEP_SILENCE_MS: int = 100  # ｜ 切り出し時に前後へ加える余白（ms）
VAD_SEEK_STEP_MS: int = 10  # ｜ 検出の探索間隔（ms）小さいほど精密だが遅い


def detect_whisper_backend(model_name_or_path: str) -> str:
    """
    'hf'か'openai'を返す。ローカルディレクトリにkonfig.jsonがあれば'hf'。
    hfはローカルのモデルを使うときに指定する。
    openaiは公式のWhisperモデルを使うときに指定する。
    """
    p = Path(model_name_or_path)
    if p.exists() and p.is_dir() and (p / "config.json").exists():
        return "hf"  # Transformers形式
    return "openai"


# -------------------------------------------------
# ユーティリティ：出力ディレクトリが無ければ作成
# -------------------------------------------------
def ensure_dirs(path: str) -> None:
    """
    指定されたパスのディレクトリ部分を作成（存在していれば何もしない）。
    例: path = "data/out/result.csv" → "data/out" を作成
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


# -------------------------------------------------
# ① 区間分割：話者分離（pyannote） or 無音検出（pydub）
# -------------------------------------------------
def diarize_segments(
    wav_path: str, num_speakers: int | None = None
) -> list[tuple[float, float, str]]:
    """
    pyannote が使えれば pyannote（GPU対応）、ダメなら pydub VAD に自動フォールバック。
    """
    if USE_PYANNOTE and HF_TOKEN:
        try:
            from pyannote.audio import Pipeline

            model_name = "pyannote/speaker-diarization-3.1"
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=HF_TOKEN)

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda:0"))  # ← 重要
                print("[Diarization] method=pyannote device=cuda:0")
            else:
                print("[Diarization] method=pyannote device=cpu")

            print(f"[Diarization]  diarization start...")

            # 既知話者数があれば指定
            diarization = (
                pipeline(wav_path, num_speakers=num_speakers)
                if num_speakers
                else pipeline(wav_path)
            )

            print(f"[Diarization]  diarization completed.")
            # [start, end, speaker]のリスト
            segments: list[tuple[float, float, str]] = []  # 出力リストを初期化

            # 各発話区間と話者ラベルを列挙
            # track: トラック番号（同じ話者が同時に複数発話するときなどの区別用）は今回使わないので、_で無視
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # 秒単位にキャストしタプルでsegmentsリストに追加
                segments.append((float(turn.start), float(turn.end), str(speaker)))
            if segments:  # 1件以上得られたらそのまま返却。
                # 開始時刻でソートし時間順に整列したものを返す
                return sorted(segments, key=lambda x: x[0])
        except Exception as e:  # モデル未同意/ネット不通/VRAM不足等の例外処理
            warnings.warn(f"[pyannote fallback] {e}")

    # ---- ここから VAD (pydub) フォールバック ----
    # ログにフォールバックしたことを明示。
    print("[Diarization] method=VAD(pydub)")
    # pydubで音声ファイルを読み込む
    audio = AudioSegment.from_file(wav_path)
    # 無音と判定する閾値を設定（平均音量からのマージン）。
    silence_thresh_db = audio.dBFS - VAD_SILENCE_MARGIN_DB
    # 非無音（発話らしい）区間のリストを取得。
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=VAD_MIN_SILENCE_LEN_MS,
        silence_thresh=silence_thresh_db,
        seek_step=VAD_SEEK_STEP_MS,
    )
    # VAD結果を格納するリストを初期化
    segments: list[tuple[float, float, str]] = []
    # 取得した各非無音区間に対して処理
    for start_ms, end_ms in nonsilent:
        # 先頭に余白を付与（負方向に出ないようmax）
        s = max(0, start_ms - VAD_KEEP_SILENCE_MS)
        # 末尾に余白を付与（音声が途切れないようmin）
        e = min(len(audio), end_ms + VAD_KEEP_SILENCE_MS)
        # 秒に変換し単一話者ラベルで保存。
        segments.append((s / 1000.0, e / 1000.0, "SPEAKER_00"))
    if not segments:  # まったく非無音が見つからなかった場合
        # 全体を一つの区間として扱う
        segments = [(0.0, len(audio) / 1000.0, "SPEAKER_00")]
    # [(start,end,speaker)]形式のリストを返す
    return segments


# -------------------------------------------------
# ② 文字起こし（Whisper）
#   - Whisper のモデルは最初に 1 回だけロード（高速化）
#   - 各区間ごとに一時WAVへ書き出して transcribe
# -------------------------------------------------
def transcribe_segments(
    wav_path: str,
    segments: list[tuple[float, float, str]],
    model_name: str,
    language: str,
    device: str,
) -> list[tuple[float, float, str, str]]:
    """
    引数:
      wav_path  : 元の音声ファイルパス
      segments  : diarize_segments() が返す [(start, end, speaker), ...]
      model_name: Whisper のモデル名（tiny/base/small/medium/large-v2 等）
      language  : 'auto' なら言語自動判定。'ja' などで固定可
      device    : 'auto' / 'cpu' / 'cuda'（Whisperをどこで動かすか）
    戻り値:
      [(start, end, speaker, text), ...]
    """
    """
    Whisper推論（openai-whisper or Transformers）を自動選択して区間ごとに文字起こし。
    """
    # --- デバイス決定 ---
    if device.lower() == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = device.lower() == "cuda"
    device_index = 0 if use_cuda else -1

    backend = detect_whisper_backend(model_name)
    print(f"[Whisper] backend={backend}")

    # 共通の切り出し準備
    audio = AudioSegment.from_file(wav_path)
    results: list[tuple[float, float, str, str]] = []

    if backend == "openai":
        import whisper as openai_whisper

        print(f"[Whisper] model load... / model = {model_name}")
        model = openai_whisper.load_model(
            model_name, device="cuda" if use_cuda else "cpu"
        )
        print(f"[Whisper] completed / device = {model.device}")

        for s, e, spk in segments:
            clip = audio[int(s * 1000) : int(e * 1000)]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                clip.export(tmp.name, format="wav")
                kwargs = {"fp16": use_cuda}
                if language and language.lower() != "auto":
                    kwargs["language"] = language
                out = model.transcribe(tmp.name, **kwargs)
                text = (out.get("text") or "").strip()
                results.append((s, e, spk, text))
        return results

    # ---- ここから Transformers backend ----
    from transformers import pipeline as hf_asr_pipeline

    # Whisper は生成時に言語/タスク指定が可能（Transformers 4.44+）
    # ja固定時は language="japanese", task="transcribe"
    generate_kwargs = {}
    if language and language.lower() != "auto":
        # Whisperの言語指定は英語名かISOコード。日本語なら "japanese" が分かりやすい
        lang_kw = (
            "japanese" if language.lower() in {"ja", "jpn", "japanese"} else language
        )
        generate_kwargs = {"language": lang_kw, "task": "transcribe"}

    # GPUなら半精度で軽く（小〜中モデルで有効）
    torch_dtype = torch.float16 if use_cuda else None

    print(f"[Whisper/HF] loading model from: {model_name}")
    asr = hf_asr_pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=device_index,
        torch_dtype=torch_dtype,
        return_timestamps=False,
        generate_kwargs=generate_kwargs or None,
    )
    print(f"[Whisper/HF] loaded. device={'cuda:0' if use_cuda else 'cpu'}")

    for s, e, spk in segments:
        clip = audio[int(s * 1000) : int(e * 1000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            clip.export(tmp.name, format="wav")
            out = asr(tmp.name)
            text = (
                (out.get("text") or "").strip()
                if isinstance(out, dict)
                else str(out).strip()
            )
            results.append((s, e, spk, text))

    return results


def build_sentiment_pipeline(model_name: str | None = None):
    """日本語モデル優先 → 候補が全滅なら英語SST-2"""
    device = 0 if torch.cuda.is_available() else -1

    def _make(name: str):
        print(f"[Sentiment] model={name}")
        return hf_pipeline("sentiment-analysis", model=name, device=device)

    # 明示指定が最優先
    if model_name:
        return _make(model_name)

    # 日本語モデルの候補（順にトライ）
    ja_candidates = [
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",  # 多クラス
        "llm-book/bert-base-japanese-v3-wrime-sentiment",  # 推奨（WRIME）
        "jarvisx17/japanese-sentiment-analysis",  # 2値POS/NEG
    ]
    for name in ja_candidates:
        try:
            return _make(name)
        except Exception as e:
            warnings.warn(f"[sentiment candidate failed] {name}: {e}")

    # 英語にフォールバック
    warnings.warn("[sentiment fallback to English] all Japanese models failed")
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )


def analyze_sentiments(
    transcripts: list[tuple[float, float, str, str]], sent_pipe
) -> list[tuple[float, float, str, str, str, float]]:
    """
    文字起こし結果（(start, end, speaker, text)）に対して、
    感情ラベルとスコアを付与して返す。
    戻り値:
      [(start, end, speaker, text, label, score), ...]
    """
    outputs: list[tuple[float, float, str, str, str, float]] = []

    for s, e, spk, text in transcripts:
        if text.strip():
            # transformers のパイプラインは [{label:.., score:..}] の形で返す
            res = sent_pipe(text, truncation=True)[0]
            label = str(res.get("label"))
            score = float(res.get("score", 0.0))
        else:
            # テキストが空（無音/音楽等）の場合は neutral/0.0 を付与（運用方針次第で変えてOK）
            label, score = "neutral", 0.0

        outputs.append((s, e, spk, text, label, score))

    return outputs


# -------------------------------------------------
# 引数パーサ（CLI引数の定義）
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="音声 → 区間分割 → 文字起こし → 感情分析 → CSV の一括実行スクリプト"
    )
    # 入力/出力
    p.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="入力音声ファイルのパス（wav/mp3 など）",
    )
    p.add_argument(
        "--out",
        dest="output_csv",
        required=True,
        help="出力CSVのパス（例: data/out/result.csv）",
    )

    # Whisper の挙動
    p.add_argument(
        "--whisper_model",
        default="small",
        help="Whisperモデル名（tiny/base/small/medium/large-v2 など）",
    )
    p.add_argument(
        "--language",
        default="auto",
        help="発話言語（'auto' で自動判定 / 'ja' などで固定）",
    )
    p.add_argument(
        "--device", default="auto", help="Whisperの実行デバイス（auto/cpu/cuda）"
    )

    # 感情モデル指定（任意）
    p.add_argument(
        "--sentiment_model",
        default=None,
        help="感情分析に使うHugging Faceモデル名（未指定で日本語→英語の順に自動）",
    )
    # 話者数を指定したいとき用
    p.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="既知の話者数（未指定なら自動推定）",
    )

    return p.parse_args()


# -------------------------------------------------
# ④ CSV 書き出し
# -------------------------------------------------
def write_csv(
    out_csv_path: str,
    input_file: str,
    analyzed: list[tuple[float, float, str, str, str, float]],
) -> None:
    """
    analyzed を CSV に保存する。列は以下の通り：
      file, start_sec, end_sec, speaker, transcript, sentiment, score, created_at
    """
    ensure_dirs(out_csv_path)  # ｜ 出力先ディレクトリが無ければ作成

    header = [
        "file",
        "start_sec",
        "end_sec",
        "speaker",
        "transcript",
        "sentiment",
        "score",
        "created_at",
    ]
    rows = []
    for s, e, spk, text, label, score in analyzed:
        rows.append(
            [
                os.path.basename(input_file),  # ｜ 入力ファイル名のみ
                round(s, 3),  # ｜ 小数3桁に丸め（見やすさのため）
                round(e, 3),
                spk,
                text,
                label,
                round(score, 4),  # ｜ スコアは4桁に丸め
                datetime.now().isoformat(timespec="seconds"),  # ｜ 実行時刻
            ]
        )

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# -------------------------------------------------
# メイン（ここから実質の“パイプライン”が動く）
# -------------------------------------------------
def main() -> None:
    # 1) 引数を読む
    args = parse_args()

    # 2) 入力の存在チェック（無ければ早期に落として分かりやすくする）
    assert os.path.isfile(args.input_path), f"入力が見つかりません: {args.input_path}"

    # 3) 区間分割（話者分離 or 無音検出）
    segments = diarize_segments(args.input_path, num_speakers=args.num_speakers)
    # 何セグメント切れたか表示
    print(f"[Diarization] segments={len(segments)}")

    #    ここで得られるのは [(start_sec, end_sec, speaker_label), ...]
    #    無音が多いと区間数は少なめ、会話が続くと細かく刻まれます。

    # 4) 文字起こし（Whisper）— モデルは最初に1回ロードし、各区間を順に処理
    transcripts = transcribe_segments(
        args.input_path,
        segments,
        model_name=args.whisper_model,
        language=args.language,
        device=args.device,
    )
    #    transcripts は [(start, end, speaker, text), ...] の形になります。

    # 5) 感情分析（Transformers）— まず日本語モデルを試し、ダメなら英語にフォールバック
    sent_pipe = build_sentiment_pipeline(model_name=args.sentiment_model)
    print(f"[Sentiment] device = {sent_pipe.model.device}")  # cpuかcudaか確認
    analyzed = analyze_sentiments(transcripts, sent_pipe)
    #    analyzed は [(start, end, speaker, text, label, score), ...] です。

    # 6) CSV に保存
    write_csv(args.output_csv, args.input_path, analyzed)

    # 7) 完了ログ（行数を表示）
    print(f"[OK] CSV出力: {args.output_csv}  行数={len(analyzed)}")


# スクリプトとして直接呼ばれたときだけ main() を実行（モジュール import 時は実行しない）
if __name__ == "__main__":
    main()

```

## transformers でフルファインチューニングしたモデルで文字起こしをしようとするとエラーに

```
╭─harum@masaX ~/audio-ie ‹main›
╰─$ docker compose run --rm app python -m src.pipeline \                                                                                                                                                                           130 ↵
  --in samples/amagasaki/amagasaki__2014_10_28_2min.mp3 \
  --out data/out/result-small.csv \
  --whisper_model models/whisper-small-ja-full \
  --language ja \
  --device auto

/opt/conda/lib/python3.10/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
/opt/conda/lib/python3.10/site-packages/pyannote/audio/core/io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
config.yaml: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:00<00:00, 2.14MB/s]
/opt/conda/lib/python3.10/site-packages/pyannote/audio/pipelines/speaker_verification.py:43: UserWarning: torchaudio._backend.get_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  backend = torchaudio.get_audio_backend()
/opt/conda/lib/python3.10/site-packages/pyannote/audio/pipelines/speaker_verification.py:45: UserWarning: Module 'speechbrain.pretrained' was deprecated, redirecting to 'speechbrain.inference'. Please update your script. This is a change from SpeechBrain 1.0. See: https://github.com/speechbrain/speechbrain/releases/tag/v1.0.0
  from speechbrain.pretrained import (
/opt/conda/lib/python3.10/site-packages/pyannote/audio/pipelines/speaker_verification.py:53: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend(backend)
/opt/conda/lib/python3.10/site-packages/pyannote/audio/tasks/segmentation/mixins.py:37: UserWarning: `torchaudio.backend.common.AudioMetaData` has been moved to `torchaudio.AudioMetaData`. Please update the import path.
  from torchaudio.backend.common import AudioMetaData
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.91M/5.91M [00:01<00:00, 3.93MB/s]
config.yaml: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 399/399 [00:00<00:00, 2.12MB/s]
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26.6M/26.6M [00:00<00:00, 51.2MB/s]
config.yaml: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 221/221 [00:00<00:00, 817kB/s]
[Diarization] method=pyannote device=cuda:0
[Diarization]  diarization start...
/opt/conda/lib/python3.10/site-packages/pyannote/audio/utils/reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
It can be re-enabled by calling
   >>> import torch
   >>> torch.backends.cuda.matmul.allow_tf32 = True
   >>> torch.backends.cudnn.allow_tf32 = True
See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.

  warnings.warn(
/opt/conda/lib/python3.10/site-packages/pyannote/audio/models/blocks/pooling.py:104: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at /opt/conda/conda-bld/pytorch_1716905979055/work/aten/src/ATen/native/ReduceOps.cpp:1807.)
  std = sequences.std(dim=-1, correction=1)
[Diarization]  diarization completed.
[Diarization] segments=45
[Whisper] backend=hf
[Whisper/HF] loading model from: models/whisper-small-ja-full
`torch_dtype` is deprecated! Use `dtype` instead!
Device set to use cuda:0
[Whisper/HF] loaded. device=cuda:0
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/conda/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/work/src/pipeline.py", line 479, in <module>
    main()
  File "/work/src/pipeline.py", line 455, in main
    transcripts = transcribe_segments(
  File "/work/src/pipeline.py", line 268, in transcribe_segments
    out = asr(tmp.name)
  File "/opt/conda/lib/python3.10/site-packages/transformers/pipelines/automatic_speech_recognition.py", line 275, in __call__
    return super().__call__(inputs, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/transformers/pipelines/base.py", line 1459, in __call__
    return next(
  File "/opt/conda/lib/python3.10/site-packages/transformers/pipelines/pt_utils.py", line 126, in __next__
    item = next(self.iterator)
  File "/opt/conda/lib/python3.10/site-packages/transformers/pipelines/pt_utils.py", line 271, in __next__
    processed = self.infer(next(self.iterator), **self.params)
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 631, in __next__
    data = self._next_data()
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 675, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 32, in fetch
    data.append(next(self.dataset_iter))
  File "/opt/conda/lib/python3.10/site-packages/transformers/pipelines/pt_utils.py", line 188, in __next__
    processed = next(self.subiterator)
  File "/opt/conda/lib/python3.10/site-packages/transformers/pipelines/automatic_speech_recognition.py", line 374, in preprocess
    import torchcodec
  File "/opt/conda/lib/python3.10/site-packages/torchcodec/__init__.py", line 10, in <module>
    from . import decoders, samplers  # noqa
  File "/opt/conda/lib/python3.10/site-packages/torchcodec/decoders/__init__.py", line 7, in <module>
    from ._core import VideoStreamMetadata
  File "/opt/conda/lib/python3.10/site-packages/torchcodec/decoders/_core/__init__.py", line 8, in <module>
    from ._metadata import (
  File "/opt/conda/lib/python3.10/site-packages/torchcodec/decoders/_core/_metadata.py", line 15, in <module>
    from torchcodec.decoders._core.video_decoder_ops import (
  File "/opt/conda/lib/python3.10/site-packages/torchcodec/decoders/_core/video_decoder_ops.py", line 12, in <module>
    from torch.library import get_ctx, register_fake
ImportError: cannot import name 'register_fake' from 'torch.library' (/opt/conda/lib/python3.10/site-packages/torch/library.py)

```
