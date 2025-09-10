flowchart TD
A[CLI 引数 parse_args()] --> B[出力ディレクトリ作成]
B --> C[Processor/Model 読み込み<br/>WhisperProcessor.from_pretrained<br/>WhisperForConditionalGeneration.from_pretrained]
C --> C2[pad_token/EOS 設定<br/>forced_decoder_ids 設定<br/>suppress_tokens 解除]
C2 --> D[CSV/TSV 読み込み<br/>load_csv_as_dataset(train/valid)]
D --> E[prepare 関数を map 適用<br/>音声読込 →(必要なら)16kHz リサンプリング<br/>log-Mel 抽出 / tokenizer]
E --> F{--use_lora ?}
F -- Yes --> G[LoRA 構成: LoraConfig<br/>target: q/k/v/out_proj, fc1, fc2<br/>get_peft_model でラップ]
F -- No --> H[そのままモデルを使用]
G --> I[DataCollatorSpeechSeq2SeqWithPadding<br/>(-100 で pad 無視)]
H --> I
I --> J[Seq2SeqTrainingArguments 構築]
J --> K[Seq2SeqTrainer 構築<br/>compute_metrics(WER/CER), EarlyStopping]
K --> L[trainer.train() 実行]
L --> M{--merge_lora_and_save ?}
M -- Yes(LoRA 時) --> N[merge_and_unload() でベースへマージ → 保存]
M -- No --> O[trainer.save_model / processor.save_pretrained]
N --> P[[[ 完了 / 出力Dir]]]
O --> P

flowchart LR
subgraph Input
A[audio_path, text<br/>from CSV/TSV]
end

    A --> B[prepare関数]
    B -->|sf.read| C[np.float32 波形 (T,) or (T,C)]
    C -->|多ch平均| D[モノラル化]
    D -->|sr!=16k & librosa有| E[librosa.resample → 16kHz]
    D -->|sr==16k or librosa無| E2[そのまま/エラー]

    E --> F[processor.feature_extractor<br/>→ log-Mel (80×T)]
    E2 --> F

    A --> B2[tokenizer(text) → label_ids]
    F --> G[map後の例:<br/>{"input_features": np(80×T), "labels": list[int]}]

    subgraph Collator
        G --> H[DataCollatorSpeechSeq2SeqWithPadding]
        H --> I[input_features: FloatTensor (B,80,Tmax)]
        H --> J[labels: LongTensor (B,Lmax), pad→-100]
    end

    I --> K[Model.forward]
    J --> K

    K --> L[Loss (CrossEntropy, -100は無視)]
    K --> M[Eval時 generate() → tokens]
    M --> N[processor.batch_decode → 予測文字列]

sequenceDiagram
participant U as ユーザ/CLI
participant T as Trainer
participant M as Model(Whisper/LoRA)
participant C as Collator
participant Met as compute_metrics(WER/CER)

    U->>T: trainer.train()
    loop 各ステップ (train)
        T->>C: バッチ取り出し & pad(-100)
        C-->>T: input_features, labels
        T->>M: forward (学習)
        M-->>T: loss
        T->>T: 勾配降下/累積/更新
    end

    Note over T: evaluation_strategy="steps"<br/>eval_steps ごとに評価
    T->>T: evalタイミング到来
    loop 検証ループ (eval)
        T->>C: validバッチ作成
        C-->>T: input_features, labels
        T->>M: generate(max_length, num_beams)
        M-->>T: pred_ids
        T->>Met: batch_decode(pred/label) → WER/CER
        Met-->>T: {wer, cer}
    end
    T->>T: metric_for_best_model="wer"<br/>ベスト更新なら保存
    T->>T: EarlyStoppingCallback(patience) 判定

    alt use_lora=True かつ merge_lora_and_save=True
        T->>M: merge_and_unload()
        M-->>T: マージ済み通常モデル
        T->>U: save_pretrained(output_dir)
    else
        T->>U: save_model / save_pretrained(output_dir)
    end
