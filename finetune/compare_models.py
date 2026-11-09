"""So sánh các model trên một CSV (cột text, label; tùy chọn error_type) + McNemar khi có 2 model.

Ví dụ:
  python compare_models.py --data ../dataset/drafts/challenge_draft.csv \
      --models ../outputs/phobert_intent/final_model ../outputs/phobert_intent_v2/final_model
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from vncorenlp import VnCoreNLP

from common import ID2LABEL, VNCORENLP_JAR, segment_text
from predict import load_classifier, predict_batch


def evaluate(df, segmented, model_dir):
    tokenizer, model = load_classifier(Path(model_dir))
    ids, _ = predict_batch(segmented, tokenizer, model)
    pred = pd.Series([ID2LABEL[i] for i in ids], index=df.index)
    correct = pred == df.label
    unknown = df.label == "UNKNOWN"
    print(f"\n== {model_dir}")
    print(f"accuracy {correct.mean():.2%} | macro-F1 {f1_score(df.label, pred, average='macro'):.2%}"
          f" | recall UNKNOWN {correct[unknown].mean():.2%}")
    if "error_type" in df:
        print(correct.groupby([df.error_type, df.label]).mean().map("{:.0%}".format).to_string())
    return pred, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", help="Ghi CSV dự đoán của mọi model.")
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    if "review" in df:
        df = df[df["review"].fillna("") != "xoa"].reset_index(drop=True)
    with VnCoreNLP(str(VNCORENLP_JAR), annotators="wseg", max_heap_size="-Xmx1g") as segmenter:
        segmented = [segment_text(segmenter, t) for t in df.text]

    results = [evaluate(df, segmented, m) for m in args.models]

    if len(results) == 2:
        (_, a), (_, b) = results
        only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
        p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        print(f"\nChỉ model 1 đúng: {only_a} | chỉ model 2 đúng: {only_b} | McNemar p = {p:.3f}")

    if args.out:
        out = df.copy()
        for i, (pred, _) in enumerate(results, start=1):
            out[f"pred_{i}"] = pred
        out.to_csv(args.out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
