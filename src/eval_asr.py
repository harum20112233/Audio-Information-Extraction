# tools/eval_asr.py（改）
from pathlib import Path
import pandas as pd
import unicodedata, re
from rapidfuzz import fuzz
from jiwer import cer, wer, Compose, RemovePunctuation, ToLowerCase, Strip


# 1) 正規化：make_asr_dataset.pyと同じNFKC＋空白潰し
def normalize_text_ja(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# jiwer側の軽正規化（句読点や大小）
norm = Compose([ToLowerCase(), Strip(), RemovePunctuation()])

# 入力
pred_csv = Path("data/out/isha_baseline.csv")  # 予測（pipeline出力）
gold_csv = Path("data/ref/isha_gold.csv")  # 正解（CSV/区間ごと）推奨
terms_txt = Path("data/ref/terms_medical.txt")  # 医療用語リスト

# 2) 区間アライン評価（start/end で結合）※ズレが大きい場合は連結評価にフォールバック
pred = pd.read_csv(pred_csv)
gold = pd.read_csv(gold_csv)

# try: 近い時刻でjoin（±0.25s）
tol = 0.25
merged = pred.merge(gold, on="file", suffixes=("_pred", "_gold"))
merged = merged[
    (merged.start_sec_pred - merged.start_sec_gold).abs()
    <= tol & (merged.end_sec_pred - merged.end_sec_gold).abs()
    <= tol
].copy()


def nz(s):
    return normalize_text_ja(str(s or ""))


if len(merged) < max(3, 0.5 * len(gold)):  # マッチ少なすぎ→連結比較に切替
    hyp = normalize_text_ja(" ".join(str(x) for x in pred["transcript"].fillna("")))
    ref = normalize_text_ja(" ".join(str(x) for x in gold["transcript"].fillna("")))
    cer_val = cer(ref, hyp, truth_transform=norm, hypothesis_transform=norm)
    wer_val = wer(ref, hyp, truth_transform=norm, hypothesis_transform=norm)
    dur_w_cer = cer_val  # 近似
else:
    # 3) 時間重み付きCER：長い区間の寄与を大きく
    seg_scores = []
    for _, r in merged.iterrows():
        ref = nz(r["transcript_gold"])
        hyp = nz(r["transcript_pred"])
        dur = max(1e-3, float(r["end_sec_gold"] - r["start_sec_gold"]))
        c = cer(ref, hyp, truth_transform=norm, hypothesis_transform=norm)
        w = wer(ref, hyp, truth_transform=norm, hypothesis_transform=norm)
        seg_scores.append((dur, c, w))
    total_dur = sum(d for d, _, _ in seg_scores)
    dur_w_cer = sum(d * c for d, c, _ in seg_scores) / total_dur
    dur_w_wer = sum(d * w for d, _, w in seg_scores) / total_dur
    cer_val, wer_val = dur_w_cer, dur_w_wer

# 4) 用語評価：再現率/適合率/F1（ゆるい一致）
terms = [
    t.strip() for t in terms_txt.read_text(encoding="utf-8").splitlines() if t.strip()
]
hyp_all = normalize_text_ja(" ".join(str(x) for x in pred["transcript"].fillna("")))
ref_all = normalize_text_ja(" ".join(str(x) for x in gold["transcript"].fillna("")))


def fuzzy_in(term, text, th=80):  # しきい値は70→80推奨
    return fuzz.partial_ratio(term, text) >= th


tp = sum(1 for t in terms if fuzzy_in(t, hyp_all) and fuzzy_in(t, ref_all))
fn = sum(1 for t in terms if (not fuzzy_in(t, hyp_all)) and fuzzy_in(t, ref_all))
# FPは「正解にない“医療用語らしき文字列”」の抽出が難しいので、まずは用語集合に限定して近似
fp = sum(1 for t in terms if fuzzy_in(t, hyp_all) and not fuzzy_in(t, ref_all))

recall = tp / max(1, tp + fn)
precision = tp / max(1, tp + fp)
f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

print(f"CER (time-weighted): {cer_val:.3f}")
print(f"WER (time-weighted): {wer_val:.3f}")
print(
    f"Medical terms  Precision: {precision*100:.1f}%  Recall: {recall*100:.1f}%  F1: {f1*100:.1f}%"
)
