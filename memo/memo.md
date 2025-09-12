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
# train_asr.py (torchcodec 非依存 / Whisper fine-tune 完成版)

"""
============================================
- OpenAI Whisper を Hugging Face Transformers で追加学習
- フルFT / LoRA(PEFT) 両対応
- 16 kHz 推奨。librosa があれば自動リサンプリング
- DataCollatorSpeechSeq2SeqWithPadding 採用（音声入力に必須）
- WER / CER を評価。最良モデルを保存

CSV/TSV 仕様:
  - 必須列: audio_path, text
  - audio_path は CSV/TSV の場所からの相対パスまたは絶対パス

使用例:
  docker compose run --rm app python -m src.train_asr

  この例は今のところ動かないが、以下のように実行する想定
  docker compose run --rm app \
    python -m src.train_asr \
      --base_model openai/whisper-small \
      --train_csv train_data/asr_2min/train.csv \
      --valid_csv train_data/asr_2min/valid.csv \
      --output_dir models/whisper-small-ja-lora \
      --use_lora \
      --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
      --num_train_epochs 5 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 2 \
      --learning_rate 1e-5 \
      --weight_decay 0.01 \
      --warmup_ratio 0.1 \
      --fp16 \
      --gradient_checkpointing \
      --merge_lora_and_save
============================================
"""
# ファイルのエンコーディングや将来の型ヒント互換のための宣言を先に行う
# ここでは前方参照型注釈（Python 3.7+）を有効化して、文字列型注釈の扱いを改善する
from __future__ import annotations

# 標準ライブラリの argparse を使ってコマンドライン引数を定義・解析する
import argparse

# OS パス操作や環境設定（ディレクトリ作成など）に用いる
import os

# 学習ステップ数等の計算で使用（切り上げなど）
import math

# 実行時の軽微な警告を制御（必要に応じて無視）するためのモジュール
import warnings

# 型ヒントに Dict, Any などを使うために typing をインポート
from typing import Dict, Any

# 数値・信号処理での配列計算に使用（音声配列の変換・平均化等）
import numpy as np

# PyTorch 本体（テンソル、学習、AMP、GPU 利用など）
import torch

# Hugging Face Datasets（CSV/TSV の読み込みや map による前処理で使用）
import datasets

# load_dataset を名前空間から直接使うためのインポート（今回は使っていないが参照として）
from datasets import load_dataset

# Transformers ライブラリから Whisper と学習関連のクラス群をインポート
from transformers import (
    WhisperForConditionalGeneration,  # Whisper 本体の seq2seq 生成モデル
    WhisperProcessor,  # 特徴抽出＋トークナイザを一体に扱うプロセッサ
    Seq2SeqTrainingArguments,  # Trainer 用の学習設定
    Seq2SeqTrainer,  # seq2seq 専用の Trainer 実装
    EarlyStoppingCallback,  # 早期終了（評価指標の悪化が続くと学習停止）
)

# 評価指標（WER/CER）をロードする evaluate ライブラリ
import evaluate

# 音声ファイルの読み込み（wav/flac など）に用いる SoundFile
import soundfile as sf

# ---- LoRA（任意） ----
# LoRA（PEFT: Parameter-Efficient Fine-Tuning）の設定とラッパー
from peft import LoraConfig, get_peft_model

# librosa は任意（16kHz 以外を自動リサンプリングするために使う）
try:
    # librosa は音声のリサンプリング等で便利だが、未インストールでも動くように try/except
    import librosa  # type: ignore

    # 利用可能フラグを True に
    _HAS_LIBROSA = True
except Exception:
    # インポート失敗時は False（16kHz 以外の入力を弾く）
    _HAS_LIBROSA = False

# デバッグ・保守向け：オブジェクトがどこで定義されているか等を調べる際に使える
import inspect


# ---- Whisper用コラトラ（バックポート） -----------------------
# DataCollatorSpeechSeq2SeqWithPadding: 音声タスク（Whisper）向けの最小限コラトラ
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Whisper向けの最小限データコラトラ。
    入力要素: {"input_features": np.ndarray(80, T) もしくは (T, 80), "labels": list[int]}
    出力:
      - input_features: FloatTensor [B, 80, Tmax]
      - labels: LongTensor [B, Lmax] （pad_tokenは -100 に変換）
    """

    # 初期化で processor（特徴抽出＋トークナイザ）と decoder_start_token_id（任意）を受け取る
    def __init__(
        self, processor: WhisperProcessor, decoder_start_token_id: int | None = None
    ):
        # 特徴抽出・トークナイザ処理に使う WhisperProcessor を保持
        self.processor = processor
        # デコーダ開始トークン ID（未使用だが将来拡張に備えて受け取り）
        self.decoder_start_token_id = decoder_start_token_id
        # パディングに用いるトークン ID（tokenizer 側の pad_token_id）
        self.pad_token_id = processor.tokenizer.pad_token_id

    # バッチの list[dict] を受け取り、パディング済みテンソル群を返す
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # 1) 音声特徴のパディング準備
        #    processor.feature_extractor.pad が期待する {"input_features": Tensor} の形に整形
        input_feats = []
        for f in features:
            # 各サンプルの "input_features" を取り出してテンソル化
            x = f["input_features"]
            x = torch.tensor(x)
            # 形が (T, 80) の場合は (80, T) に転置（Whisper は [80, T] を想定）
            if x.ndim == 2 and x.shape[0] != 80 and x.shape[1] == 80:
                x = x.transpose(0, 1)
            # pad 用に {"input_features": Tensor} 形式で蓄積
            input_feats.append({"input_features": x})

        # feature_extractor.pad を用いて、時系列長を最大長に合わせてパディング（バッチ化）
        batch_inputs = self.processor.feature_extractor.pad(
            input_feats, return_tensors="pt"
        )

        # 2) ラベル列のパディング
        #    tokenizer.pad で長さを揃えた後、pad_token_id を -100 に変換（損失計算で無視）
        label_ids = [f["labels"] for f in features]
        labels = self.processor.tokenizer.pad(
            {"input_ids": label_ids}, padding=True, return_tensors="pt"
        )["input_ids"]
        # 交差エントロピー計算で無視されるよう、pad 部分を -100 に置換
        labels = labels.masked_fill(labels == self.pad_token_id, -100)

        # 学習で必要なテンソルを返す（型は PyTorch Tensor）
        return {
            "input_features": batch_inputs["input_features"].to(torch.float32),
            "labels": labels.long(),
        }


# --------------------------------------------------------------


# =============================
#  argparse
# =============================


# コマンドライン引数の定義関数
def parse_args():
    # スクリプト概要のヘルプ文言を与えて ArgumentParser を生成
    p = argparse.ArgumentParser("Fine-tune Whisper (full/LoRA)")
    # 学習のベースとなる Whisper モデル名（Hugging Face のリポジトリ指定など）
    p.add_argument("--base_model", default="openai/whisper-small")
    # 学習データ（CSV/TSV）へのパス
    p.add_argument("--train_csv", default="train_data/asr_2min/train.csv")
    # 検証データ（CSV/TSV）へのパス
    p.add_argument("--valid_csv", default="train_data/asr_2min/valid.csv")
    # 学習結果の保存先ディレクトリ
    p.add_argument("--output_dir", default="models/whisper-small-ja-lora")

    # LoRA 利用の有無（指定時に LoRA での学習）
    p.add_argument("--use_lora", action="store_true")
    # LoRA のランク r（低ランクの次元）
    p.add_argument("--lora_r", type=int, default=16)
    # LoRA のスケーリング係数 alpha
    p.add_argument("--lora_alpha", type=int, default=32)
    # LoRA のドロップアウト率
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # 学習エポック数
    p.add_argument("--num_train_epochs", type=int, default=5)
    # 1 デバイス当たりのトレーニング時バッチサイズ
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    # 1 デバイス当たりの評価時バッチサイズ
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    # 勾配累積ステップ数（大きなバッチ相当を実現）
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    # 学習率
    p.add_argument("--learning_rate", type=float, default=1e-5)
    # L2 正則化（Weight Decay）の係数
    p.add_argument("--weight_decay", type=float, default=0.01)
    # ウォームアップ比率（総ステップに対する割合）
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    # 省メモリ/精度関連フラグ（半精度など）
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    # 勾配チェックポイント（活性値の再計算で VRAM 節約）
    p.add_argument("--gradient_checkpointing", action="store_true")

    # 評価・保存・ロギング頻度など
    p.add_argument("--eval_strategy", default="epoch")
    p.add_argument("--save_strategy", default="epoch")  # eval_strategyと一致させる
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)
    # 早期終了の忍耐回数（改善が見られない評価回数）
    p.add_argument("--patience", type=int, default=5)

    # Whisper の言語とタスク（transcribe/translate など）
    p.add_argument("--language", default="ja")
    p.add_argument("--task", default="transcribe")

    # 生成時（評価時に predict_with_generate=True のため使用）
    p.add_argument("--generation_max_length", type=int, default=225)
    p.add_argument("--generation_num_beams", type=int, default=1)

    # データローダのワーカ数（並列前処理）
    p.add_argument("--num_workers", type=int, default=4)

    # LoRA 学習時、学習後に LoRA を元モデルへマージし通常モデルとして保存するか
    p.add_argument(
        "--merge_lora_and_save",
        action="store_true",
        help="LoRA をベースにマージして通常モデルとして保存",
    )

    # 解析した引数オブジェクトを返す
    return p.parse_args()


# =============================
#  Dataset loader
# =============================


# CSV/TSV を Hugging Face Datasets として読み込むユーティリティ
def load_csv_as_dataset(path: str):
    # 拡張子から区切り文字を判定（.tsv はタブ、それ以外はカンマ）
    ext = os.path.splitext(path)[1].lower()
    sep = "\t" if ext == ".tsv" else ","
    # datasets.load_dataset を使って、単一 split="train" としてロード
    return datasets.load_dataset("csv", data_files=path, delimiter=sep, split="train")


# WhisperProcessor による前処理関数を生成する（datasets.map 用）
def build_prepare_fn(processor: WhisperProcessor, base_dir: str):
    """CSV の行を Whisper 入力/ラベルへ変換する関数を返す。
    - input_features: log-Mel (numpy)
    - labels: list[int]
    """

    # 実際に 1 レコードを受け取り、辞書を返す内部関数
    def prepare(batch: Dict[str, Any]) -> Dict[str, Any]:
        # CSV の audio_path 列から実ファイルパスを解決（相対なら base_dir と結合）
        rel = batch["audio_path"]
        path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        # 音声ファイルの存在チェック（なければ明示的にエラー）
        if not os.path.exists(path):
            raise FileNotFoundError(f"audio file not found: {path}")

        # SoundFile で音声を読み込む（np.ndarray: (T,) もしくは (T, C)）
        audio, sr = sf.read(path)  # np.ndarray (T,) or (T, C)
        # ステレオ等の多チャンネルの場合は平均してモノラル化
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # モノラル化

        # サンプリングレートが 16kHz 以外ならリサンプリング（librosa が無い場合はエラー）
        if sr != 16000:
            if _HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                raise RuntimeError(
                    f"Expected 16kHz but got {sr}. Install librosa or provide 16k audio."
                )

        # Whisper の feature_extractor で log-Mel スペクトログラムを生成（テンソル返却）
        inputs = processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )

        # ラベル側は tokenizer で ID 列へ（<|ja|><|transcribe|> 等は forced_decoder_ids で自動付与）
        label_ids = processor.tokenizer(
            batch["text"], add_special_tokens=True
        ).input_ids

        # datasets.map が扱いやすいよう、numpy/list で返却
        return {
            "input_features": inputs.input_features[0].numpy(),  # numpy のまま
            "labels": label_ids,  # list[int] のまま
        }

    # 上記の内部関数を返す（map から呼ばれる）
    return prepare


# =============================
#  Metrics (WER / CER)
# =============================


# Trainer から呼ばれる評価関数を作る（WER/CER を同時計算）
def build_metrics_fn(processor: WhisperProcessor):
    # word error rate（単語誤り率）メトリクスをロード
    wer_metric = evaluate.load("wer")
    # character error rate（文字誤り率）メトリクスをロード
    cer_metric = evaluate.load("cer")

    # Trainer から渡される pred オブジェクトを受け、辞書を返す関数
    def compute_metrics(pred):
        # 予測 ID（logits などのタプルで返ることもあるため先頭要素を取り出す）
        pred_ids = pred.predictions
        # pred.predictions が (logits, ...) の場合に備えて先頭を予測 ID とみなす
        if isinstance(pred_ids, (tuple, list)):
            pred_ids = pred_ids[0]

        # ID 列からテキストへ復号（特殊トークンをスキップ）
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # 参照ラベル側も -100 を pad_token_id に戻してからテキスト復号
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # evaluate に渡して WER/CER を同時計算して返す
        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    # 上記 compute_metrics を返す
    return compute_metrics


# =============================
#  Main
# =============================


# エントリポイント（引数処理→前処理→学習→保存）をまとめる
def main():
    # コマンドライン引数を解析
    args = parse_args()

    # 出力ディレクトリを作成（既にある場合は何もしない）
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Processor / Model の用意
    #   WhisperProcessor は feature_extractor と tokenizer を内包しており、
    #   言語とタスクを指定すると生成側プロンプト取得にも使える
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task=args.task
    )
    # Whisper の seq2seq モデル本体をプリトレからロード
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # tokenizer に pad_token が無い場合があるので、EOS を pad として流用（標準的対応）
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # モデル設定にも pad_token_id を反映
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # 生成時の強制デコーダトークン（言語とタスクに対応するプロンプト ID）
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    # 初期抑制トークンを無効化（必要に応じて空集合に）
    model.generation_config.suppress_tokens = []

    # 勾配チェックポイントを使う場合、キャッシュは無効化（再計算を許可）し、メモリ節約
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # 2) データ読み込み（train/valid）と前処理の map
    train_ds = load_csv_as_dataset(args.train_csv)
    valid_ds = load_csv_as_dataset(args.valid_csv)

    # 相対パス解決の起点となる基準ディレクトリを決定（CSV の所在ディレクトリ）
    train_base = os.path.dirname(os.path.abspath(args.train_csv))
    valid_base = os.path.dirname(os.path.abspath(args.valid_csv))

    # 並列前処理のためのプロセス数を確保（OS の CPU 論理コア数と num_workers から安全に決定）
    num_proc = min(os.cpu_count() or 1, max(1, args.num_workers))

    # 学習データに対して前処理（音声→log-Mel、テキスト→ID 列）を適用
    train_ds = train_ds.map(
        build_prepare_fn(processor, train_base),
        remove_columns=train_ds.column_names,  # 元の列は不要になるため削除
        num_proc=num_proc,  # 並列度
        desc="prepare-train",  # 進捗バーに表示する説明
    )
    # 検証データに対しても同様の前処理
    valid_ds = valid_ds.map(
        build_prepare_fn(processor, valid_base),
        remove_columns=valid_ds.column_names,
        num_proc=num_proc,
        desc="prepare-valid",
    )

    # 3) LoRA（任意設定）
    if args.use_lora:
        # LoRA のターゲット層を Whisper の注意機構・FFN に合わせて指定
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        # モデルを LoRA 対応ラッパに差し替え（学習可能パラメータを削減）
        model = get_peft_model(model, lora_cfg)
        # 学習可能パラメータ数を標準出力へ表示（デバッグ・確認用）
        model.print_trainable_parameters()

    # 4) 評価関数（WER/CER）の構築
    compute_metrics = build_metrics_fn(processor)

    # 5) 音声タスク向けデータコラトラを用意（pad→-100 などを処理）
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 6) 学習設定（Seq2SeqTrainingArguments）の構築
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,  # 出力フォルダ
        per_device_train_batch_size=args.per_device_train_batch_size,  # 学習バッチ
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # 評価バッチ
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 勾配累積
        learning_rate=args.learning_rate,  # 学習率
        weight_decay=args.weight_decay,  # L2 正則化
        warmup_ratio=args.warmup_ratio,  # ウォームアップ比
        num_train_epochs=args.num_train_epochs,  # 総エポック
        eval_strategy=args.eval_strategy,  # 評価戦略（epoch/steps）
        save_strategy=args.save_strategy,  # 保存戦略（eval_strategyと一致させる）
        eval_steps=args.eval_steps,  # 評価間隔
        save_steps=args.save_steps,  # 保存間隔
        logging_steps=args.logging_steps,  # ログ出力間隔
        save_total_limit=2,  # 保存世代の上限
        predict_with_generate=True,  # 評価時に生成を有効化
        fp16=args.fp16,  # 半精度 (fp16) 使用
        bf16=args.bf16,  # bfloat16 使用（ハード依存）
        gradient_checkpointing=args.gradient_checkpointing,  # 勾配チェックポイント
        report_to="none",  # ロガー（wandb 等）への報告無効
        load_best_model_at_end=True,  # 最高スコアのモデル読込
        metric_for_best_model="wer",  # ベスト判定の指標
        greater_is_better=False,  # WER は小さいほど良い
        # 音声タスクでは未使用列を自動削除して DataCollator に委ねる
        remove_unused_columns=True,
        # 長さ別グルーピングを無効化（可読・再現性重視／必要に応じて変更可）
        group_by_length=False,
        # DataLoader のワーカ数（Python マルチプロセス）
        dataloader_num_workers=max(0, args.num_workers),
        # 生成設定（評価時の最大長やビーム数）
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
    )

    # 7) Trainer の構築（モデル・データ・評価・コールバック等を結合）
    trainer = Seq2SeqTrainer(
        model=model,  # 学習対象モデル（LoRA ラップ済み可）
        args=training_args,  # 上記の学習設定
        data_collator=data_collator,  # 音声向けコラトラ
        tokenizer=processor,  # 生成時にデコード用として使用
        train_dataset=train_ds,  # 学習データ
        eval_dataset=valid_ds,  # 検証データ
        compute_metrics=compute_metrics,  # WER/CER 計算関数
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience)
        ],  # 早期終了
    )

    # 8) 学習の実行（内部でループ・評価・保存などを管理）
    trainer.train()

    # 9) 保存フェーズ
    if args.use_lora and args.merge_lora_and_save:
        # LoRA アダプタ重みをベースモデルへマージし、通常のモデルとして保存
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
    else:
        # 通常保存：LoRA アダプタを含む（もしくはフル FT 後のモデル）として保存
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    # 完了メッセージ（保存先を表示）
    print("[OK] saved:", args.output_dir)


# スクリプトが直接実行されたときだけ main() を起動（モジュール import では起動しない）
if __name__ == "__main__":
    main()

```

## 動かなかった方のコマンド

```
╭─harum@masaX ~/audio-ie ‹feature/whisper-ft●›
╰─$
docker compose run --rm app \
  python -m src.train_asr \
    --base_model openai/whisper-small \
    --train_csv train_data/asr_2min/train.csv \
    --valid_csv train_data/asr_2min/valid.csv \
    --output_dir models/whisper-small-ja-lora \
    --use_lora \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --fp16 \
    --gradient_checkpointing \
    --merge_lora_and_save

/opt/conda/lib/python3.10/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
prepare-train (num_proc=4): 100%|███████████████████████| 24/24 [00:05<00:00,  4.04 examples/s]
num_proc must be <= 3. Reducing num_proc to 3 for dataset of size 3.
prepare-valid (num_proc=3): 100%|█████████████████████████| 3/3 [00:04<00:00,  1.34s/ examples]
trainable params: 6,488,064 || all params: 248,222,976 || trainable%: 2.6138047752678624
Using the latest cached version of the module from /root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--wer/e41eaa77ca7152430cd94704de20946c1b004b5b488ab5d20b26fb81c6c15506 (last modified on Wed Aug 27 11:47:35 2025) since it couldn't be found locally at evaluate-metric--wer, or remotely on the Hugging Face Hub.
Using the latest cached version of the module from /root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--cer/87ffabb91696e0a8647c7685d414f639d31fb1eed705afe32e5ab03f4692245c (last modified on Wed Aug 27 11:47:37 2025) since it couldn't be found locally at evaluate-metric--cer, or remotely on the Hugging Face Hub.
  0%|                                                                    | 0/5 [00:00<?, ?it/s]Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/conda/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/work/src/train_asr.py", line 504, in <module>
    main()
  File "/work/src/train_asr.py", line 485, in main
    trainer.train()
  File "/opt/conda/lib/python3.10/site-packages/transformers/trainer.py", line 1885, in train
    return inner_training_loop(
  File "/opt/conda/lib/python3.10/site-packages/transformers/trainer.py", line 2216, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
  File "/opt/conda/lib/python3.10/site-packages/transformers/trainer.py", line 3238, in training_step
    loss = self.compute_loss(model, inputs)
  File "/opt/conda/lib/python3.10/site-packages/transformers/trainer.py", line 3264, in compute_loss
    outputs = model(**inputs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/accelerate/utils/operations.py", line 819, in forward
    return model_forward(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/accelerate/utils/operations.py", line 807, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
  File "/opt/conda/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 16, in decorate_autocast
    return func(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/peft/peft_model.py", line 1326, in forward
    return self.base_model(
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/lib/python3.10/site-packages/peft/tuners/tuners_utils.py", line 161, in forward
    return self.model.forward(*args, **kwargs)
TypeError: WhisperForConditionalGeneration.forward() got an unexpected keyword argument 'input_ids'
  0%|                                                                    | 0/5 [00:01<?, ?it/s]

```

## 動いた方のコマンド

```
╭─harum@masaX ~/audio-ie ‹main●›
╰─$ docker compose run --rm app python -m src.train_asr

/opt/conda/lib/python3.10/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 9664.29it/s]
num_proc must be <= 3. Reducing num_proc to 3 for dataset of size 3.
/work/src/train_asr.py:471: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
  trainer = Seq2SeqTrainer(
  0%|                                                                                                                                                                                                             | 0/10 [00:00<?, ?it/s]You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
 20%|███████████████████████████████████████▍                                                                                                                                                             | 2/10 [00:06<00:23,  2.90s/it]You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
{'eval_loss': 1.8311270475387573, 'eval_wer': 1.0, 'eval_cer': 0.11764705882352941, 'eval_runtime': 1.3825, 'eval_samples_per_second': 2.17, 'eval_steps_per_second': 0.723, 'epoch': 1.0}
 20%|███████████████████████████████████████▍                                                                                                                                                             | 2/10 [00:08<00:23,  2.90s/it/opt/conda/lib/python3.10/site-packages/transformers/modeling_utils.py:4034: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 448, 'suppress_tokens': [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362], 'begin_suppress_tokens': [220, 50257]}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
  warnings.warn(
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
 40%|██████████████████████████████████████████████████████████████████████████████▊                                                                                                                      | 4/10 [00:23<00:33,  5.57s/it]You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
{'eval_loss': 1.4648158550262451, 'eval_wer': 1.0, 'eval_cer': 0.18823529411764706, 'eval_runtime': 1.3849, 'eval_samples_per_second': 2.166, 'eval_steps_per_second': 0.722, 'epoch': 2.0}
 40%|██████████████████████████████████████████████████████████████████████████████▊                                                                                                                      | 4/10 [00:24<00:33,  5.57s/itYou're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
 60%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                              | 6/10 [00:36<00:22,  5.61s/it]You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
{'eval_loss': 1.3121376037597656, 'eval_wer': 1.0, 'eval_cer': 0.2, 'eval_runtime': 1.2767, 'eval_samples_per_second': 2.35, 'eval_steps_per_second': 0.783, 'epoch': 3.0}
 60%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                              | 6/10 [00:37<00:22,  5.61s/itYou're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                       | 8/10 [00:53<00:13,  6.52s/it]You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
{'eval_loss': 1.2276040315628052, 'eval_wer': 1.0, 'eval_cer': 0.12941176470588237, 'eval_runtime': 1.4492, 'eval_samples_per_second': 2.07, 'eval_steps_per_second': 0.69, 'epoch': 4.0}
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                       | 8/10 [00:55<00:13,  6.52s/itYou're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:09<00:00,  6.56s/it]You're using a WhisperTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
{'eval_loss': 1.1893086433410645, 'eval_wer': 1.0, 'eval_cer': 0.12941176470588237, 'eval_runtime': 1.4026, 'eval_samples_per_second': 2.139, 'eval_steps_per_second': 0.713, 'epoch': 5.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:10<00:00,  6.56s/itThere were missing keys in the checkpoint model loaded: ['proj_out.weight'].
{'train_runtime': 90.5454, 'train_samples_per_second': 1.325, 'train_steps_per_second': 0.11, 'train_loss': 2.4354724884033203, 'epoch': 5.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:30<00:00,  9.05s/it]
[OK] saved: models/whisper-small-ja-lora
```
