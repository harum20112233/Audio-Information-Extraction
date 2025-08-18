# =========================
# ベースイメージを指定します。PyTorch + CUDA 12.1 の実行環境が入っています
# - これによりGPUでPyTorchを動かせる土台が一発で整います
# =========================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# =========================
# 非対話モードでのapt実行時に余計な質問が出ないよう抑制します
# =========================
ARG DEBIAN_FRONTEND=noninteractive

# =========================
# OSパッケージをインストールします
# - ffmpeg        : 音声/動画の入出力（Whisper等で使う）
# - libsndfile1   : 音声ファイル読み込みで使われることが多い
# - git           : 将来、レポジトリからモデル/コードを引くときに役立つ
# - 最後にaptキャッシュを削除し、イメージサイズを小さくします
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# =========================
# コンテナ内の作業ディレクトリを /work にします
# 以降のファイルコピーやコマンドの実行は /work 基準になります
# =========================
WORKDIR /work

# =========================
# pip の余計なキャッシュを残さない・バージョン確認を抑制する設定です
# 小さな最適化ですが地味に効きます
# =========================
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# =========================
# Python依存パッケージの一覧をコンテナにコピーします
# 変更が少ない順にCOPY/RUNするとDockerビルドのキャッシュが効きやすくなります
# =========================
COPY requirements.txt .

# =========================
# pip を最新化し、requirements.txt に基づいて依存をインストールします
# - ここでWhisper / Transformers / pyannote.audio などが入ります
# =========================
RUN pip install --upgrade pip && pip install -r requirements.txt

# =========================
# スモークテスト用のPythonスクリプトをコピーします
# =========================
COPY smoke.py .

# =========================
# コンテナ起動時にデフォルトで実行するコマンドを指定します
# ここでは smoke.py を実行して環境確認ができるようにしています
# =========================
CMD ["python", "smoke.py"]
