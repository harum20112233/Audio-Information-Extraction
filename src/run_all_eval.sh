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
      python -m src.whisper_wer_cer_eval \
        --input_csv data/in/${name}_WAV/${name}_eval_input.csv \
        --audio_root . \
        --whisper_model models/whisper-small-${name}-${type} \
        --language ja \
        --device auto \
        --out_csv data/out/${name}_${type}_eval_small.csv
        
    # 1つの処理が終わったらログ表示
    echo "--- 完了: name=${name}, type=${type} ---"
    echo "" # 改行
    
  done
done


for name in "${NAMES[@]}"; do
    # 実行するコマンドをログとして表示
    echo "--- 実行中: name=${name}, type=vanilla ---"
    
    # コマンドを実行
    docker compose run --rm app \
      python -m src.whisper_wer_cer_eval \
        --input_csv data/in/${name}_WAV/${name}_eval_input.csv \
        --audio_root . \
        --whisper_model small \
        --language ja \
        --device auto \
        --out_csv data/out/${name}_vanilla_eval_small.csv
        
    # 1つの処理が終わったらログ表示
    echo "--- 完了: name=${name}, type=vanilla ---"
    echo "" # 改行
done
echo "全ての処理が完了しました。"