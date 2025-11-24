# =========================
# ベースイメージを指定します。PyTorch + CUDA 12.6 の実行環境が入っています
# - これによりGPUでPyTorchを動かせる土台が一発で整います
# pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtimeが元のやつ、不具合があったら戻す
# =========================
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# =========================
# 【追加】uv を公式イメージからコピー (これが一番手軽で確実です)
# /usr/local/bin/uv に配置されるため、パスが通った状態で使えます
# =========================
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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
# 環境変数の設定
# UV_SYSTEM_PYTHON=1: 仮想環境を作らず、コンテナのPython環境(System Python)に直接インストールする設定
# UV_COMPILE_BYTECODE=1: インストール時にバイトコンパイルを行い、起動速度を少し上げる
# =========================
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# =========================
# Python依存パッケージの一覧をコンテナにコピーします
# 変更が少ない順にCOPY/RUNするとDockerビルドのキャッシュが効きやすくなります
# =========================
COPY requirements.txt .


# =========================
# 【変更】uv を使ってインストール
# --system: システムのPython環境にインストール (Dockerではこれが基本)
# --no-cache: Dockerイメージサイズ削減のためキャッシュを残さない
# =========================
RUN uv pip install --system --no-cache -r requirements.txt

# =========================
# スモークテスト用のPythonスクリプトをコピーします
# =========================
COPY smoke.py .

# =========================
# コンテナ起動時にデフォルトで実行するコマンドを指定します
# ここでは smoke.py を実行して環境確認ができるようにしています
# =========================
CMD ["python", "smoke.py"]
