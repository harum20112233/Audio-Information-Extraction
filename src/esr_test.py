"""
esr_test.py
音声から感情を推定するためのテストスクリプト（SUPERB版・Safetensors強制・修正済み）
"""

import os
import glob
import torch
import librosa
import numpy as np
from transformers import pipeline

TARGET_SR = 16000
WAV_DIR = "./data/test_wav"

# 【確実性No.1】SUPERB ベンチマークのモデル
# 多くの研究で基準として使われるため、動作が最も安定しています
MODEL_NAME = "superb/wav2vec2-base-superb-er"


def main():
    # GPU設定
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    print(f"感情認識モデルの読み込み中...{MODEL_NAME}")

    try:
        # 【重要修正】use_safetensors=True を追加
        # これにより、古いpickle形式の読み込みエラーを回避し、安全なファイルを強制的に使います
        classifier = pipeline(
            "audio-classification",
            model=MODEL_NAME,
            device=device,
            use_safetensors=True,
        )
    except Exception as e:
        print(f"Pipeline読み込みエラー: {e}")
        return

    print("モデルの読み込みが完了しました。")

    wav_files = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
    if not wav_files:
        print("テスト用のWAVファイルが見つかりません。")
        return

    print(f"テスト用WAVファイル数は {len(wav_files)} 件です。解析を開始します。")
    print("-" * 65)
    print(f"{'ファイル名':<20} | {'推定感情':<10} | {'確信度'} | {'全スコア(上位3件)'}")
    print("-" * 65)

    # SUPERBモデルのラベル定義（英語→日本語）
    label_map = {
        "hap": "喜び",
        "neu": "中立",
        "ang": "怒り",
        "sad": "悲しみ",
    }

    for file_path in wav_files:
        try:
            filename = os.path.basename(file_path)

            # 音声をロード & 16kHzにリサンプリング
            # ※wav2vec2モデルは16kHz入力が必須要件です
            y, sr = librosa.load(file_path, sr=TARGET_SR)

            if len(y) < sr * 0.1:
                print(f"{filename:<20} | (スキップ: 音声が短すぎます)")
                continue

            # 推論実行
            results = classifier(y, top_k=None)

            # 1位の結果を取得
            top_result = results[0]
            label_en = top_result["label"]
            score = top_result["score"]

            # 日本語ラベルへの変換
            label_jp = label_map.get(label_en, label_en)

            # 詳細スコアの整形
            details = ", ".join(
                [
                    f"{label_map.get(r['label'], r['label'])}:{r['score']:.2f}"
                    for r in results[:3]
                ]
            )

            print(f"{filename:<20} | {label_jp:<10} | {score:.2%} | {details}")

        except Exception as e:
            print(f"❌ エラー: {filename} -> {e}")

    print("-" * 65)
    print("✅ 全ての処理が完了しました。")


if __name__ == "__main__":
    main()
