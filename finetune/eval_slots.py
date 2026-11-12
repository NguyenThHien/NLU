"""So sánh model slot theo target_obj.

--data: jsonl giống dataset gốc (lấy OBJECT từ slots) hoặc CSV cột text, target_obj (tùy chọn category).
Với jsonl chỉ chấm câu có intent cần slot (như API). Chia kết quả theo category nếu có.

  python eval_slots.py --data ../dataset/test.jsonl ../dataset/slots_v2/new_val.jsonl \
      --models ../outputs/phobert_slots/final_model ../outputs/phobert_slots_v2/final_model
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from vncorenlp import VnCoreNLP

from common import VNCORENLP_JAR, segment_text
from predict import extract_slots, load_slot_tagger, target_object

SLOT_INTENTS = {"READ_TEXT", "DESCRIBE_SCENE", "ANSWER_VISUAL_QUESTION"}


def load_rows(path):
    """→ [(text, danh sách OBJECT vàng, category)]."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        return [(r["text"], [s["value"] for s in r["slots"] if s["type"] == "OBJECT"], r.get("category", "-"))
                for r in rows if r["label"] in SLOT_INTENTS]
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    return [(r.text, [r.target_obj.lower()] if r.target_obj else [], getattr(r, "category", "-") or "-")
            for r in frame.itertuples()]


def score(rows, segmented, tokenizer, model):
    """Độ chính xác target_obj (OBJECT đầu tiên, kể cả đúng là không có) và F1 OBJECT, theo category."""
    stats = defaultdict(Counter)
    for (text, gold, category), seg in zip(rows, segmented):
        pred = [s["value"].lower() for s in extract_slots(text, seg, tokenizer, model) if s["type"] == "OBJECT"]
        for key in ("all", category):
            stats[key]["n"] += 1
            stats[key]["target_ok"] += target_object([{"type": "OBJECT", "value": v} for v in pred]) == (gold or [None])[0]
            stats[key]["tp"] += sum((Counter(gold) & Counter(pred)).values())
            stats[key]["pred"] += len(pred)
            stats[key]["gold"] += len(gold)
    table = {}
    for key, c in stats.items():
        precision = c["tp"] / c["pred"] if c["pred"] else 0.0
        recall = c["tp"] / c["gold"] if c["gold"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        table[key] = {"n": c["n"], "target_obj_acc": c["target_ok"] / c["n"], "object_f1": f1}
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    datasets = {path: load_rows(path) for path in args.data}
    with VnCoreNLP(str(VNCORENLP_JAR), annotators="wseg", max_heap_size="-Xmx1g") as segmenter:
        segmented = {path: [segment_text(segmenter, text) for text, _, _ in rows] for path, rows in datasets.items()}

    for model_dir in args.models:
        tokenizer, model = load_slot_tagger(Path(model_dir).resolve())
        print(f"\n== {model_dir}")
        for path, rows in datasets.items():
            table = score(rows, segmented[path], tokenizer, model)
            print(f"  {Path(path).name}")
            for key, m in sorted(table.items(), key=lambda item: item[0] != "all"):
                print(f"    {key:<12} n={m['n']:<4} target_obj {m['target_obj_acc']:.1%}  OBJECT F1 {m['object_f1']:.1%}")


if __name__ == "__main__":
    main()
