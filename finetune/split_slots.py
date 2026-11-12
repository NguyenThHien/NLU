"""Chia lại train/val/test riêng cho slot: vật thể (và mẫu câu nếu được) ở test không xuất hiện trong train.

Bộ chia gốc phục vụ intent nên 80% OBJECT ở test đã có trong train → điểm slot bị thổi phồng.
Chỉ dùng để ĐO; model chạy thật vẫn có thể train trên toàn bộ dữ liệu.

  python split_slots.py                 # chia theo vật thể + mẫu câu (nếu không gộp quá lớn)
  python train_slots.py --data-dir ../dataset/slots_split --run-name phobert_slots_split
"""

import argparse
import json
import random
from collections import Counter

from common import ROOT_DIR, SEED
from merge_drafts import core

SLOT_INTENTS = {"READ_TEXT", "DESCRIBE_SCENE", "ANSWER_VISUAL_QUESTION"}
FRACTIONS = {"test": 0.15, "val": 0.10}
MAX_GROUP_SHARE = 0.30


def template(row):
    """Câu sau khi thay mỗi slot bằng <LOẠI> và bỏ tiền tố khẩu ngữ: 'đọc <OBJECT> cho tôi'."""
    text = row["text"]
    for slot in sorted(row["slots"], key=lambda s: s["start"], reverse=True):
        text = text[:slot["start"]] + f"<{slot['type']}>" + text[slot["end"]:]
    return core(text)


def group_rows(rows, use_template):
    """Union-find: câu chung một OBJECT (hoặc chung mẫu câu) thuộc cùng nhóm."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for index, row in enumerate(rows):
        keys = [("obj", s["value"]) for s in row["slots"] if s["type"] == "OBJECT"]
        if use_template:
            keys.append(("tpl", template(row)))
        for key in keys:
            if key in owner:
                parent[find(index)] = find(owner[key])
            else:
                owner[key] = index
    return [find(i) for i in range(len(rows))]


def assign(groups, total, rng):
    """Xáo nhóm rồi lấp test, val theo tỉ lệ; phần còn lại vào train."""
    sizes = Counter(groups)
    order = list(sizes)
    rng.shuffle(order)
    split_of, filled = {}, Counter()
    for group in order:
        split = next((s for s, f in FRACTIONS.items() if filled[s] + sizes[group] <= f * total), "train")
        split_of[group] = split
        filled[split] += sizes[group]
    return [split_of[g] for g in groups]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="slots_split", help="Thư mục con trong dataset/.")
    args = parser.parse_args()
    rng = random.Random(SEED)

    rows = [json.loads(line) for split in ("train", "val", "test")
            for line in (ROOT_DIR / "dataset" / f"{split}.jsonl").open(encoding="utf-8")]
    slot_rows = [r for r in rows if r["label"] in SLOT_INTENTS]
    other_rows = [r for r in rows if r["label"] not in SLOT_INTENTS]

    use_template = True
    groups = group_rows(slot_rows, use_template)
    largest = max(Counter(groups).values()) / len(slot_rows)
    if largest > MAX_GROUP_SHARE:
        print(f"Nhóm theo mẫu câu gộp {largest:.0%} số câu vào một nhóm → chỉ chia theo vật thể.")
        use_template = False
        groups = group_rows(slot_rows, use_template)
    print(f"{len(slot_rows)} câu có slot, {len(set(groups))} nhóm, nhóm lớn nhất "
          f"{max(Counter(groups).values())} câu (tiêu chí: vật thể{' + mẫu câu' if use_template else ''})")

    splits = {"train": [], "val": [], "test": []}
    for row, split in zip(slot_rows, assign(groups, len(slot_rows), rng)):
        splits[split].append(row)
    rng.shuffle(other_rows)
    for index, row in enumerate(other_rows):
        position = index / len(other_rows)
        splits["test" if position < FRACTIONS["test"] else "val" if position < sum(FRACTIONS.values()) else "train"].append(row)

    objects = {s: {x["value"] for r in rows_ for x in r["slots"] if x["type"] == "OBJECT"} for s, rows_ in splits.items()}
    assert not objects["test"] & objects["train"], "vật thể test lọt vào train"
    assert not objects["val"] & objects["train"], "vật thể val lọt vào train"

    out = ROOT_DIR / "dataset" / args.name
    out.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        rng.shuffle(split_rows)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            file.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in split_rows)
        print(f"{split}: {len(split_rows)} câu, {len(objects[split])} vật thể, "
              f"{sum(r['label'] in SLOT_INTENTS for r in split_rows)} câu có intent cần slot")


if __name__ == "__main__":
    main()
