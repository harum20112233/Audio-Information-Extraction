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

from __future__ import annotations
import argparse
import os
import math
import warnings
from typing import Dict, Any

import numpy as np
import torch
import datasets
from datasets import load_dataset

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorSpeechSeq2SeqWithPadding,
)

import evaluate
import soundfile as sf

# ---- LoRA（任意） ----
from peft import LoraConfig, get_peft_model

# librosa は任意（16kHz以外の入力を自動で16kへ）
try:
    import librosa  # type: ignore

    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


# =============================
#  argparse
# =============================


def parse_args():
    p = argparse.ArgumentParser("Fine-tune Whisper (full/LoRA)")
    p.add_argument("--base_model", default="openai/whisper-small")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--valid_csv", required=True)
    p.add_argument("--output_dir", required=True)

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # 学習
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    # 省メモリ/精度
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # 評価/保存/ロギング
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)

    # Whisper 言語/タスク
    p.add_argument("--language", default="ja")
    p.add_argument("--task", default="transcribe")

    # 生成系（評価時）
    p.add_argument("--generation_max_length", type=int, default=225)
    p.add_argument("--generation_num_beams", type=int, default=1)

    # I/O
    p.add_argument("--num_workers", type=int, default=4)

    # LoRA をマージして通常モデルとして保存するか
    p.add_argument(
        "--merge_lora_and_save",
        action="store_true",
        help="LoRA をベースにマージして通常モデルとして保存",
    )

    return p.parse_args()


# =============================
#  Dataset loader
# =============================


def load_csv_as_dataset(path: str):
    ext = os.path.splitext(path)[1].lower()
    sep = "\t" if ext == ".tsv" else ","
    return datasets.load_dataset("csv", data_files=path, delimiter=sep, split="train")


def build_prepare_fn(processor: WhisperProcessor, base_dir: str):
    """CSV の行を Whisper 入力/ラベルへ変換する関数を返す。
    - input_features: log-Mel (numpy)
    - labels: list[int]
    """

    def prepare(batch: Dict[str, Any]) -> Dict[str, Any]:
        rel = batch["audio_path"]
        path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"audio file not found: {path}")

        audio, sr = sf.read(path)  # np.ndarray (T,) or (T, C)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # モノラル化

        if sr != 16000:
            if _HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                raise RuntimeError(
                    f"Expected 16kHz but got {sr}. Install librosa or provide 16k audio."
                )

        # 入力特徴量（log-Mel）
        inputs = processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )

        # ラベル（forced_decoder_ids により <|ja|><|transcribe|> は自動付与）
        label_ids = processor.tokenizer(
            batch["text"], add_special_tokens=True
        ).input_ids

        return {
            "input_features": inputs.input_features[0].numpy(),  # numpy のまま
            "labels": label_ids,  # list[int] のまま
        }

    return prepare


# =============================
#  Metrics (WER / CER)
# =============================


def build_metrics_fn(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        # pred.predictions が (logits, ...) の場合に備える
        if isinstance(pred_ids, (tuple, list)):
            pred_ids = pred_ids[0]

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    return compute_metrics


# =============================
#  Main
# =============================


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Processor / Model
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task=args.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # pad token 明示設定
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # 推論時プロンプト（日本語/転写を固定）
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    model.generation_config.suppress_tokens = []

    # gradient checkpointing を使うなら use_cache=False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # 2) データ読み込み
    train_ds = load_csv_as_dataset(args.train_csv)
    valid_ds = load_csv_as_dataset(args.valid_csv)

    train_base = os.path.dirname(os.path.abspath(args.train_csv))
    valid_base = os.path.dirname(os.path.abspath(args.valid_csv))

    num_proc = min(os.cpu_count() or 1, max(1, args.num_workers))

    train_ds = train_ds.map(
        build_prepare_fn(processor, train_base),
        remove_columns=train_ds.column_names,
        num_proc=num_proc,
        desc="prepare-train",
    )
    valid_ds = valid_ds.map(
        build_prepare_fn(processor, valid_base),
        remove_columns=valid_ds.column_names,
        num_proc=num_proc,
        desc="prepare-valid",
    )

    # 3) LoRA（任意）
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # 4) Metrics
    compute_metrics = build_metrics_fn(processor)

    # 5) Data collator（音声専用）
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 6) TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # 音声タスクでは必須
        remove_unused_columns=False,
        # 速度・省メモリ
        group_by_length=True,
        dataloader_num_workers=max(0, args.num_workers),
        # 生成まわり（評価時）
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
    )

    # 7) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=processor,  # processor を渡すのが最も安全
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # 8) Train
    trainer.train()

    # 9) Save
    if args.use_lora and args.merge_lora_and_save:
        # LoRA 重みをベースへマージして通常モデルとして保存
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)  # アダプタ含む/もしくはフルFT済
        processor.save_pretrained(args.output_dir)

    print("[OK] saved:", args.output_dir)


if __name__ == "__main__":
    main()
