# ============================================
# smoke.py
# 目的:
#   - コンテナ内でPython/ライブラリ/GPUが正しく使えるかを一気に確認する
#   - 実際の処理前の“健康診断”のようなスクリプト
# 使い方:
#   - コンテナ起動時に自動で実行される（DockerfileのCMDで指定）
# ============================================

# 標準ライブラリの読み込み
import sys  # Pythonのバージョンなどの情報を確認するため
import shutil  # ffmpegコマンドの存在チェックに使う
import os  # 環境変数(HFトークン)を読むため
import time  # ベンチマーク（簡易計測）用

# PyTorchを読み込み、バージョンやGPUの可用性を確認します
import torch

# --- Python / Torch の基本情報を表示 ---
print("Python version          :", sys.version.split()[0])  # 例: 3.10.14
print("PyTorch version         :", torch.__version__)  # 例: 2.3.1
print("CUDA available (PyTorch):", torch.cuda.is_available())  # TrueならGPU使用可能

# --- GPUが使えるなら、GPU名と簡単な行列積ベンチを実行 ---
if torch.cuda.is_available():
    print(
        "CUDA device name        :", torch.cuda.get_device_name(0)
    )  # 例: NVIDIA RTX ****
    # 大きめの行列を用意してGPUで行列積（@）を実行し、時間を測定
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    t1 = time.time()
    print("CUDA matmul 4096x4096 s :", round(t1 - t0, 3))

# --- ffmpeg の有無を確認（Whisperなどで必要）---
print("ffmpeg found            :", bool(shutil.which("ffmpeg")))  # TrueならOK

# --- 主要ライブラリの import テスト ---
libs = ["whisper", "transformers", "pyannote.audio", "pydub", "pandas"]
for name in libs:
    try:
        __import__(name)
        print(f"import {name:<15}      : OK")
    except Exception as e:
        print(f"import {name:<15}      : FAIL -> {e}")

# --- Hugging Face のトークンが環境変数で渡っているか（任意）---
#   * モデルにより、ダウンロード時にトークンが必要な場合があります
#   * ここでは「設定されているか」を表示するのみで、実際に使いません
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
print("HF token via env         :", bool(hf_token))  # Trueなら設定済み

print("\nSmoke test completed.")  # 最後に完了メッセージ
