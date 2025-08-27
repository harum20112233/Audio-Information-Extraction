# ============================================
# train_asr.py
# --------------------------------------------
# ASR = Automatic Speech Recognition（自動音声認識）
# 音声 → テキスト（文字起こし）を行うモデルを学習します。
#
# このスクリプトでは Hugging Face Transformers を用いて
# OpenAI Whisper モデルをベースに追加学習を行います。
#
# 主な用途:
#   - 専門用語や固有名詞の認識精度を改善するためのファインチューニング
#   - 小規模データでのドメイン適応（医療・自治体・教育など）
#
# 対応する学習方式:
#   1) フルファインチューニング（全パラメータ更新）
#   2) LoRA/PEFT による軽量学習（VRAM節約・高速）
#
# 入力:
#   - CSV/TSV: audio_path と text の列を持つデータ
#       audio_path: 音声ファイルへのパス（wav等、16kHz推奨）
#       text      : 音声に対応する文字起こし（正解ラベル）
#
# 出力:
#   - 学習済みモデル（Hugging Face Transformers形式）
#   - 評価指標: 単語誤り率 (WER)、文字誤り率 (CER)
#
# 使用例:
# 例: small を LoRA で微調整し、最終保存は LoRAマージ版（通常モデル互換）
# docker compose run --rm app \
#   python -m src.train_asr \
#     --base_model openai/whisper-small \
#     --train_csv data/asr/train.csv \
#     --valid_csv data/asr/valid.csv \
#     --output_dir models/whisper-small-med-ft \
#     --use_lora \
#     --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-5 \
#     --fp16 --gradient_checkpointing \
#     --merge_lora_and_save
#
#
# ============================================


from __future__ import annotations
import argparse, os, math
import torch
import datasets
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
import evaluate

# ---- LoRA（任意） ----
from peft import LoraConfig, get_peft_model, PeftModel


def parse_args():
    p = argparse.ArgumentParser("Fine-tune Whisper (full/LoRA)")
    p.add_argument("--base_model", default="openai/whisper-small")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--valid_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--language", default="ja")  # 日本語固定
    p.add_argument("--task", default="transcribe")  # translate ではなく transcribe
    p.add_argument(
        "--merge_lora_and_save",
        action="store_true",
        help="LoRAをベースにマージして通常モデルとして保存",
    )
    return p.parse_args()


def load_csv_as_dataset(path: str):
    ext = os.path.splitext(path)[1].lower()
    sep = "\t" if ext == ".tsv" else ","
    return datasets.load_dataset("csv", data_files=path, delimiter=sep, split="train")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Processor / Model 読み込み
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task=args.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # 推論時に <|ja|><|transcribe|> を自動で入れるよう設定
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    model.config.suppress_tokens = []

    # 2) データ読み込み
    train_ds = load_csv_as_dataset(args.train_csv)
    valid_ds = load_csv_as_dataset(args.valid_csv)

    # 16k へ統一
    sampling_rate = 16000
    train_ds = train_ds.cast_column("audio_path", Audio(sampling_rate=sampling_rate))
    valid_ds = valid_ds.cast_column("audio_path", Audio(sampling_rate=sampling_rate))

    def prepare(batch):
        # 音声テンソル
        audio = batch["audio_path"]["array"]
        inputs = processor.feature_extractor(
            audio, sampling_rate=sampling_rate, return_tensors="pt"
        )
        with processor.as_target_processor():
            labels = processor.tokenizer(batch["text"])
        batch_out = {
            "input_features": inputs.input_features[0],
            "labels": torch.tensor(labels.input_ids),
        }
        return batch_out

    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names, num_proc=1)
    valid_ds = valid_ds.map(prepare, remove_columns=valid_ds.column_names, num_proc=1)

    # 3) LoRA オプション
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # 4) 評価指標（WER/CER）
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        # whisperは generate 出力のため logits ではなく token_ids が来る想定
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    # 5) Trainer 設定
    data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model)

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
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,  # 入力側の正規化をtokenizerの代わりに使う
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    # 6) 保存
    if args.use_lora and args.merge_lora_and_save:
        # LoRAをベースにマージして通常モデルとして保存（推論が簡単になる）
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
    else:
        # そのまま保存（LoRAはアダプタとして保存／フルFTは通常保存）
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    print("[OK] saved:", args.output_dir)


if __name__ == "__main__":
    main()
