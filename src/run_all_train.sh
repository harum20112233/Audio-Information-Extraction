#!/bin/bash

# 実行するnameのリスト
NAMES=("amou" "fujii" "matsumoto" "nishimura" "suzuki" "yamaguchi" "yamamoto")
# 実行するtypeのリスト
TYPES=("raw" "make")

# ループ処理
for name in "${NAMES[@]}"; do
  for type in "${TYPES[@]}"; do
    
    # 実行するコマンドをログとして表示
    echo "--- 実行中: name=${name}, type=${type} ---"
    
    # コマンドを実行
    docker compose run --rm app \
      python -m src.train_asr \
        --base_model openai/whisper-small \
        --train_csv train_data/asr_${name}_${type}/train.csv \
        --valid_csv train_data/asr_${name}_${type}/valid.csv \
        --output_dir models/whisper-small-${name}-${type} \
        --audio_root /work
        
    # 1つの処理が終わったらログ表示
    echo "--- 完了: name=${name}, type=${type} ---"
    echo "" # 改行
    
  done
done

echo "全ての処理が完了しました。"