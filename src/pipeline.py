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
    'hf'か'openai'を返す。ローカルディレクトリにconfig.jsonがあれば'hf'。
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
