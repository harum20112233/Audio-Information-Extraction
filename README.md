# 改訂版：統合型音声分析システム(話者分離+感情推定+Whisper 微調整) 要件定義書

## 1. 概要（更新）

- **目的**: 複数話者の音声から「話者分離・文字起こし・感情分析」を実行し、**領域固有語（専門用語・商品名・略語等）に強い**文字起こしを**Whisper のファインチューニング（FT）**で実現する。FT は**Transformers ベース**で行い、少量データでも精度向上を狙う。必要に応じて**初期プロンプト**や**語彙辞書**も併用。

## 2. 機能要件（更新）

### 2.1 音声入力

- 対応：WAV（必須）、MP3（望ましい）
- サンプルレート：**16kHz/mono**を推奨（学習・推論の一貫性確保）

### 2.2 話者分離

- 既存通り：`pyannote.audio`（利用可なら）→ 失敗時`pydub`無音検出へフォールバック

### 2.3 文字起こし（更新）

- **バックエンド二択**
  - **A Transformers**（推奨）：`WhisperForConditionalGeneration` + `AutoProcessor`
  - **B openai/whisper**：FT 済み**HF モデルを OpenAI-Whisper 形式へ変換**して読み込む（互換維持）
- **初期プロンプト**（専門語のヒント）併用可。ただし**224 トークン程度**の制約あり → 語彙が多い領域では FT を基本とする。

### 2.4 感情分析

- 既存通り：Transformers の日本語モデルを優先 → 失敗時英語 SST-2 にフォールバック
- 将来的拡張：多クラス（joy/anger/sadness/neutral 等）

### 2.5 結果出力

- 必須：CSV（既存項目）
- 追加：**モデル情報**（`asr_model_id`, `asr_model_sha`, `asr_backend`, `asr_prompt_used` など）をメタ列に付与（再現性）

### 2.6 新規：Whisper ファインチューニング・モジュール

- **入力**：音声–文字ペア（Train/Valid/Test に分割）。自前音声／既存録音／**合成音声(Tacotron2 等)** の併用を許容
- **実行**：Hugging Face Transformers 公式手順で微調整（LoRA 等の軽量 FT も選択可）
- **出力**：
  - HF 形式のモデル（`<repo_or_path>`）
  - （B 案）**OpenAI-Whisper 形式への変換済み重み**（既存推論コードのまま使用可）
- **評価**：日本語は**CER（文字誤り率）**を必須、必要に応じて**形態素分割後の WER**も併用

## 3. 非機能要件（更新）

### 3.1 可搬性/再現性

- Docker で**学習・推論の両方**を再現可能に。`requirements-train.txt` と `requirements-infer.txt` を分離

### 3.2 性能

- 学習：単 GPU 24GB 以上推奨（LoRA なら**VRAM 軽減**可）
- 推論は従来通り

### 3.3 精度

- **数時間規模のラベル付き音声**でも改善が見込めることを明示（例：5 時間程度でも効果）

### 3.4 セキュリティ

- 学習用データはローカル保管。秘匿テキストは暗号化保存。HF へのモデル公開は**Private**既定

### 3.5 ログ/エラー

- 学習ログ（loss/cer/wer）、モデルカード、自動保存（`checkpoint_*`）
- 失敗時は**ベースモデルで推論にフォールバック**

## 4. 実行環境・技術スタック（更新）

- 追加（学習用）：`transformers`, `datasets`, `accelerate`, `evaluate`, `jiwer`, `peft`, `soundfile`, `librosa`
- 日本語 WER 用（任意）：`fugashi`, `ipadic`（or `ja-ginza`）

## 5. 成果物（更新）

- `src/train_whisper.py`（Transformers/LoRA 対応の学習スクリプト）
- `src/convert_hf_to_openai_whisper.py`（HF→OpenAI-Whisper 変換）
- `models/asr/<tag>/`（学習成果：HF 形式 or 変換後重み）
- `EVAL.md`（CER/WER、ドメイン語の事例比較）
- **必須フォーマット**: `CSV`
- **望ましいフォーマット**: `JSON`（オプション機能として拡張可能）
- **データ項目**:
  - `start_time` (発話開始時刻)
  - `end_time` (発話終了時刻)
  - `speaker_id` (話者 ID)
  - `text` (文字起こしされた発話内容)
  - `emotion` (感情ラベル)
  - `score` (感情分析の信頼度スコア)

---

# 実行方法

このリポジトリは **話者分離 + 文字起こし + 感情分析** をワンコマンドで実行するパイプラインです。  
Docker が入っていればセットアップ不要で動きます（GPU/CPU 両対応）。

---

## 0 前提

- OS: Linux / macOS / Windows（**WSL2 推奨**）
- 必須: [Docker](https://docs.docker.com/get-docker/) / Docker Compose
- （任意・推奨）GPU: NVIDIA + CUDA（Windows は WSL2 + NVIDIA ドライバ/WSL 用 CUDA）

> **WSL2 ユーザー向け:**  
> `docker context use default` を実行して、WSL 内の Docker デーモン（または Windows 側の Docker Desktop）を使える状態にしてください。

---

## 1 リポジトリ取得

```bash
git clone https://github.com/harum20112233/Audio-Information-Extraction audio-ie
cd audio-ie
```

## 2 hugging face トークンを設定

`.env.example`を参照し，`.env`を作成してください

```
# .env
USE_PYANNOTE=1
HUGGINGFACE_TOKEN=hf_********************************

```

## 3 Dockerfile があるディレクトリ上でビルド

```
docker compose build
```

# 4 実行

docker内でgpuが使えるかを確認するコマンド
```
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

```
docker compose run --rm app \
  python -m src.pipeline \
    --in <音声ファイルのパス> \
    --out data/out/result.csv
```

# 5 主なオプション

```
--in <path>                入力音声（WAV/MP3）
--out <path>               出力CSVの保存先
--language <auto|ja|en…>   Whisper言語 'auto' で自動判定、'ja' 固定など
--num_speakers <int>       話者数ヒント
--sentiment_model <name>   HFモデル名
--verbose                  詳細ログ
--whisper_model: Whisperモデル（tiny/base/small/…）
--device: 'auto'|'cpu'|'cuda'（Whisperの実行先）
--sentiment_model: 感情モデル名（未指定なら日本語候補→英語に自動フォールバック）
--num_speakers: 既知の話者数（pyannoteに渡すヒント）
```
