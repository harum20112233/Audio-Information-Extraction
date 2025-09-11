# cut_audio.py
# samplesの音源をカットする用のスクリプト
"""
5000ms～10000ms(5～10秒)を抽出
sound1 = sound[5000:10000]

最後の10000ms(10秒)を抽出
sound2 = sound[-10000:]
"""


from pydub import AudioSegment

# mp3ファイルの読み込み
sound = AudioSegment.from_file(
    "samples/amagasaki/amagasaki__2014_10_28.mp3", format="mp3"
)

sound1 = sound[0:127000]

# 抽出した部分を出力
sound1.export("samples/amagasaki/amagasaki__2014_10_28_2min.mp3", format="mp3")
