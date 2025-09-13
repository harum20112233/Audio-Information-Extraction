# tools/eval_asr.py
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process

# 文字・単語誤り率
from jiwer import cer, wer, Compose, RemovePunctuation, ToLowerCase, Strip

# 軽い正規化（必要に応じて調整）
norm = Compose([ToLowerCase(), Strip(), RemovePunctuation()])

# 入力
csv_pred = Path("data/out/isha_baseline.csv")
gold_txt = Path("data/ref/isha_gold.txt")
term_txt = Path("data/ref/terms_medical.txt")

# 予測テキスト（全区間のtranscriptを連結）
df = pd.read_csv(csv_pred)
hyp = " ".join(str(x) for x in df["transcript"].fillna(""))

# 正解テキスト
ref = gold_txt.read_text(encoding="utf-8")

# 語彙（用語）リスト
terms = [
    t.strip() for t in term_txt.read_text(encoding="utf-8").splitlines() if t.strip()
]

# CER / WER
cer_val = cer(ref, hyp, truth_transform=norm, hypothesis_transform=norm)
wer_val = wer(ref, hyp, truth_transform=norm, hypothesis_transform=norm)


# 用語ヒット率（ゆるい部分一致：しきい値は70で仮設定）
def term_hit_rate(terms, text, threshold=70):
    hits = 0
    for t in terms:
        score = fuzz.partial_ratio(t, text)
        if score >= threshold:
            hits += 1
    return hits / max(1, len(terms))


hit_rate = term_hit_rate(terms, hyp)

print(f"BASELINE CER: {cer_val:.3f}")
print(f"BASELINE WER: {wer_val:.3f}")
print(f"BASELINE 用語ヒット率: {hit_rate*100:.1f}%")
